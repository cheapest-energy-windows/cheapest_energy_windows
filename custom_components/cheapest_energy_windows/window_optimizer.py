"""Window optimizer for automatic window count selection.

Uses a two-phase grid search to minimize planned_total_cost:
- Phase 1 (Coarse): ~585 iterations with step=1
- Phase 2 (Fine): ~27 iterations around the best coarse result

Total: ~600 iterations, designed for sub-5-second execution.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .calculation_engine import WindowCalculationEngine
from .const import LOGGER_NAME

_LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass
class OptimizationResult:
    """Result of window optimization."""
    optimal_charge_windows: int
    optimal_discharge_windows: int
    optimal_percentile: int
    planned_total_cost: float
    baseline_cost: float
    savings_vs_baseline: float
    iterations_checked: int
    optimization_time_ms: float
    result: Dict[str, Any]
    decision_tree: List[str] = field(default_factory=list)
    below_min_savings: bool = False
    # What optimizer found before threshold check (same as optimal_* if threshold met)
    found_charge_windows: int = 0
    found_discharge_windows: int = 0
    found_percentile: int = 25
    found_savings: float = 0.0
    # Total value (for maximize_value strategy)
    total_value: float = 0.0
    optimization_strategy: str = "minimize_cost"


class WindowOptimizer:
    """Optimizer using two-phase search: coarse grid then fine refinement.

    Respects physical constraints and operational rules while bypassing
    user preference thresholds to find the true optimal configuration.

    CRITICAL: Enforces energy balance constraint - you cannot discharge
    more energy than is available from (buffer + solar + charging).
    """

    # Phase 1: Grid search (step=1 for comprehensive coverage)
    PERCENTILE_COARSE = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    CHARGE_STEP_COARSE = 1
    DISCHARGE_STEP_COARSE = 1
    # Fallback limits when battery_capacity not configured
    FALLBACK_MAX_CHARGE = 12
    FALLBACK_MAX_DISCHARGE = 8

    # Phase 2: Fine refinement (+/- 1 around best)
    FINE_OFFSET = 1

    # Energy utilization safety margin (don't plan to use 100% of available)
    ENERGY_UTILIZATION_MARGIN = 0.95

    def __init__(self):
        self._calculation_engine = WindowCalculationEngine()

    def _get_window_duration_hours(self, config: Dict[str, Any]) -> float:
        """Get window duration in hours based on pricing mode."""
        pricing_mode = config.get("pricing_window_duration", "15_minutes")
        return 0.25 if pricing_mode == "15_minutes" else 1.0

    def _calculate_window_limits(self, config: Dict[str, Any]) -> Tuple[int, int]:
        """Calculate max charge/discharge windows based on battery config.

        Returns dynamic limits if battery_capacity is configured,
        otherwise falls back to hardcoded defaults.

        Fast Search toggle (default ON) uses fallback limits for quicker optimization.

        Returns:
            Tuple of (max_charge_windows, max_discharge_windows)
        """
        # Fast Search toggle (default ON) uses fallback limits for quicker optimization
        if config.get("fast_search", True):
            return self.FALLBACK_MAX_CHARGE, self.FALLBACK_MAX_DISCHARGE

        battery_capacity = float(config.get("battery_capacity", 100.0))

        # If using default placeholder (100 kWh), use fallback limits
        if battery_capacity >= 100.0:
            return self.FALLBACK_MAX_CHARGE, self.FALLBACK_MAX_DISCHARGE

        window_hours = self._get_window_duration_hours(config)
        charge_power_kw = float(config.get("charge_power", 2500)) / 1000
        discharge_power_kw = float(config.get("discharge_power", 2500)) / 1000
        battery_rte = float(config.get("battery_rte", 85)) / 100

        # kWh per charge window (usable after RTE)
        usable_per_charge = charge_power_kw * window_hours * battery_rte
        if usable_per_charge > 0:
            max_charge = math.ceil(battery_capacity / usable_per_charge)
        else:
            max_charge = self.FALLBACK_MAX_CHARGE

        # kWh per discharge window
        kwh_per_discharge = discharge_power_kw * window_hours
        if kwh_per_discharge > 0:
            max_discharge = math.ceil(battery_capacity / kwh_per_discharge)
        else:
            max_discharge = self.FALLBACK_MAX_DISCHARGE

        # Cap to reasonable maximum (avoid explosion with tiny power values)
        # 96 windows = 24 hours in 15-min mode
        max_charge = min(max_charge, 96)
        max_discharge = min(max_discharge, 96)

        return max_charge, max_discharge

    def _calculate_available_discharge_kwh(
        self,
        num_charge: int,
        config: Dict[str, Any],
        suffix: str = ""
    ) -> float:
        """Calculate maximum available energy for discharge.

        Available energy = buffer + net_solar_to_battery + (charge_windows × charge_power × RTE)

        IMPORTANT: Solar first covers base usage during daylight hours.
        Only excess solar reaches the battery.

        Args:
            num_charge: Number of charge windows
            config: Configuration dictionary
            suffix: "_tomorrow" or "" for config key suffix

        Returns:
            Available discharge energy in kWh
        """
        # Get battery buffer (energy already in battery)
        buffer_kwh = float(config.get("battery_buffer_kwh", 0))

        # Get expected solar production (only if solar forecast is enabled)
        solar_kwh = 0.0
        if config.get("use_solar_forecast", True):
            solar_key = f"expected_solar_kwh{suffix}" if suffix else "expected_solar_kwh"
            solar_kwh = float(config.get(solar_key, 0))

        # Calculate charge energy (accounting for RTE loss during storage)
        window_duration = self._get_window_duration_hours(config)
        charge_power_kw = float(config.get("charge_power", 800)) / 1000.0
        battery_rte = float(config.get("battery_rte", 85)) / 100.0

        # Energy stored from grid charging (after RTE loss)
        charge_kwh = num_charge * window_duration * charge_power_kw * battery_rte

        # Calculate NET solar that actually reaches battery (after base usage consumes its share)
        net_solar_kwh = 0.0
        if solar_kwh > 0:
            # Get solar window duration
            solar_start_str = config.get("solar_window_start", "09:00:00")
            solar_end_str = config.get("solar_window_end", "19:00:00")

            # Parse times and calculate hours
            try:
                from datetime import datetime
                start = datetime.strptime(solar_start_str, "%H:%M:%S")
                end = datetime.strptime(solar_end_str, "%H:%M:%S")
                solar_hours = (end - start).seconds / 3600
            except (ValueError, TypeError):
                solar_hours = 10.0  # Default fallback

            # Base usage during solar hours consumes solar first
            base_usage_kw = float(config.get("base_usage", 0)) / 1000.0
            base_usage_during_solar = base_usage_kw * solar_hours

            # Only excess solar goes to battery
            excess_solar = max(0, solar_kwh - base_usage_during_solar)

            # Apply RTE loss when solar is stored in battery
            net_solar_kwh = excess_solar * battery_rte

        # Total available for discharge
        available_kwh = buffer_kwh + net_solar_kwh + charge_kwh

        return available_kwh

    def _calculate_required_discharge_kwh(
        self,
        num_discharge: int,
        config: Dict[str, Any]
    ) -> float:
        """Calculate energy required for discharge windows.

        Args:
            num_discharge: Number of discharge windows
            config: Configuration dictionary

        Returns:
            Required discharge energy in kWh
        """
        window_duration = self._get_window_duration_hours(config)
        discharge_power_kw = float(config.get("discharge_power", 800)) / 1000.0

        return num_discharge * window_duration * discharge_power_kw

    def _is_energy_feasible(
        self,
        num_charge: int,
        num_discharge: int,
        config: Dict[str, Any],
        suffix: str = ""
    ) -> bool:
        """Check if configuration is physically feasible (energy balance).

        Core constraint: You cannot discharge more than you can source.

        Args:
            num_charge: Number of charge windows
            num_discharge: Number of discharge windows
            config: Configuration dictionary
            suffix: "_tomorrow" or "" for config key suffix

        Returns:
            True if configuration is feasible, False otherwise
        """
        if num_discharge == 0:
            return True  # No discharge = always feasible

        available_kwh = self._calculate_available_discharge_kwh(num_charge, config, suffix)
        required_kwh = self._calculate_required_discharge_kwh(num_discharge, config)

        # Apply safety margin - don't plan to use 100% of available energy
        return required_kwh <= available_kwh * self.ENERGY_UTILIZATION_MARGIN

    def _get_optimization_score(self, result: Dict[str, Any], strategy: str) -> float:
        """Get the score to optimize for based on strategy.

        Returns a value where HIGHER is better for both strategies.
        This allows unified comparison logic regardless of strategy.

        Args:
            result: Calculation result dictionary
            strategy: "minimize_cost" or "maximize_value"

        Returns:
            Score where higher is better
        """
        if strategy == "maximize_value":
            # Maximize total value = savings + EOD battery value
            # savings = baseline_cost - planned_total_cost (what we saved vs doing nothing)
            # EOD value = remaining battery value for tomorrow
            baseline_cost = result.get("baseline_cost", 0)
            planned_total_cost = result.get("planned_total_cost", float('inf'))
            eod_value = result.get("battery_state_end_of_day_value", 0)
            savings = baseline_cost - planned_total_cost
            return savings + eod_value
        else:
            # minimize_cost: Return NEGATIVE cost (so higher = better)
            return -result.get("planned_total_cost", float('inf'))

    def _bypass_thresholds(self, config: Dict[str, Any], suffix: str = "") -> Dict[str, Any]:
        """Bypass user preference thresholds - keep only physical constraints.

        Physical constraints (KEEP):
        - charge_power, discharge_power (hardware limits)
        - battery_rte (physics)
        - battery_capacity, battery_buffer_kwh (physical)

        Operational rules (KEEP - user's system setup):
        - base_usage_*_strategy (how their system operates)
        - solar_priority_strategy
        - limit_discharge_to_buffer

        User preference thresholds (BYPASS - we want true optimal):
        - min_profit_charge, min_profit_discharge
        - min_price_difference
        - use_min_sell_price, min_sell_price_bypass_spread
        """
        bypassed = config.copy()

        # Bypass profit thresholds - use very negative values to allow any profit level
        # (1-hour windows can have negative profit due to RTE losses)
        bypassed[f"min_profit_charge{suffix}"] = -1000
        bypassed[f"min_profit_discharge{suffix}"] = -1000
        bypassed["min_price_difference"] = -1000

        # Bypass min sell price restrictions
        bypassed["use_min_sell_price"] = False
        bypassed["min_sell_price_bypass_spread"] = True

        return bypassed

    def optimize(
        self,
        raw_prices: List[Dict[str, Any]],
        config: Dict[str, Any],
        strategy: str = "minimize_cost",
        is_tomorrow: bool = False,
        hass: Any = None
    ) -> OptimizationResult:
        """Find optimal window configuration via two-phase grid search.

        Args:
            raw_prices: Price data for the day
            config: Full configuration dictionary
            strategy: Optimization strategy (currently only "minimize_cost")
            is_tomorrow: Whether optimizing for tomorrow
            hass: Home Assistant instance for sensor access

        Returns:
            OptimizationResult with optimal configuration and metadata
        """
        start_time = time.time()
        iterations = 0
        decision_tree: List[str] = []
        best_by_percentile: Dict[int, Tuple[int, int, float]] = {}  # {percentile: (charge, discharge, cost)}

        suffix = "_tomorrow" if is_tomorrow and config.get("tomorrow_settings_enabled", False) else ""
        min_daily_savings = config.get("min_daily_savings", 0.50)

        # Bypass thresholds to find true optimal
        test_config = self._bypass_thresholds(config, suffix)

        # Get configuration values for decision tree header
        pricing_mode = config.get("pricing_window_duration", "15_minutes")
        pricing_label = "15-min" if pricing_mode == "15_minutes" else "hourly"
        # Calculate actual periods (hourly mode aggregates 15-min prices)
        num_prices = 24 if pricing_mode == "1_hour" else len(raw_prices)
        window_hours = self._get_window_duration_hours(config)
        charge_power = float(config.get("charge_power", 2500))
        discharge_power = float(config.get("discharge_power", 2500))
        charge_kwh_per_window = charge_power / 1000 * window_hours
        discharge_kwh_per_window = discharge_power / 1000 * window_hours
        battery_rte = float(config.get("battery_rte", 85))

        # Build decision tree header
        STRATEGY_NAMES = {
            "minimize_cost": "Minimize Cost",
            "maximize_value": "Maximize Value"
        }
        strategy_display = STRATEGY_NAMES.get(strategy, strategy)
        decision_tree.append(f"Strategy: {strategy_display}")
        decision_tree.append(f"Price periods: {num_prices} ({pricing_label})")
        decision_tree.append(f"Battery RTE: {int(battery_rte)}%")
        decision_tree.append(f"Charge: {int(charge_power)}W → {charge_kwh_per_window:.3f} kWh/window")
        decision_tree.append(f"Discharge: {int(discharge_power)}W → {discharge_kwh_per_window:.3f} kWh/window")

        # Show solar/base calculation for transparency
        solar_key = f"expected_solar_kwh{suffix}" if suffix else "expected_solar_kwh"
        expected_solar = float(config.get(solar_key, 0))
        base_usage_w = float(config.get("base_usage", 0))
        if expected_solar > 0 and base_usage_w > 0:
            solar_start_str = config.get("solar_window_start", "09:00:00")
            solar_end_str = config.get("solar_window_end", "19:00:00")
            try:
                from datetime import datetime
                start = datetime.strptime(solar_start_str, "%H:%M:%S")
                end = datetime.strptime(solar_end_str, "%H:%M:%S")
                solar_hours = (end - start).seconds / 3600
            except (ValueError, TypeError):
                solar_hours = 10.0
            base_usage_kw = base_usage_w / 1000.0
            base_during_solar = base_usage_kw * solar_hours
            net_solar = max(0, expected_solar - base_during_solar)
            net_solar_after_rte = net_solar * (battery_rte / 100.0)
            decision_tree.append(f"Solar: {expected_solar:.1f} kWh - {base_during_solar:.1f} kWh base = {net_solar:.1f} kWh → battery")
            decision_tree.append(f"  After RTE: {net_solar_after_rte:.1f} kWh available for discharge")
        elif expected_solar > 0:
            decision_tree.append(f"Solar: {expected_solar:.1f} kWh (no base usage configured)")
        decision_tree.append("")

        # Calculate baseline (0 windows = no battery action, just base usage)
        baseline_config = test_config.copy()
        baseline_config[f"charging_windows{suffix}"] = 0
        baseline_config[f"expensive_windows{suffix}"] = 0

        baseline_result = self._calculation_engine.calculate_windows(
            raw_prices, baseline_config, is_tomorrow, hass
        )
        baseline_cost = baseline_result.get("planned_total_cost", float('inf'))
        baseline_score = self._get_optimization_score(baseline_result, strategy)
        iterations += 1

        # Start with baseline as best
        best_score = baseline_score
        best_cost = baseline_cost  # Keep for savings calculation
        best_config: Tuple[int, int, int] = (0, 0, 25)  # (charge, discharge, percentile)
        best_result: Dict[str, Any] = baseline_result

        # Calculate dynamic window limits based on battery config
        max_charge, max_discharge = self._calculate_window_limits(config)
        _LOGGER.debug(f"Dynamic window limits: max_charge={max_charge}, max_discharge={max_discharge}")

        # Phase 1: Coarse grid search
        decision_tree.append("--- Phase 1: Coarse Grid Search ---")
        decision_tree.append("Search space:")
        decision_tree.append(f"  Charge: 0-{max_charge} windows")
        decision_tree.append(f"  Discharge: 0-{max_discharge} windows")
        decision_tree.append(f"  Percentile: {self.PERCENTILE_COARSE}")
        total_coarse = (max_charge + 1) * (max_discharge + 1) * len(self.PERCENTILE_COARSE)
        decision_tree.append(f"  Max configs: {total_coarse}")
        decision_tree.append("")
        decision_tree.append(f"Baseline (0C/0D): €{baseline_cost:.4f}")
        decision_tree.append("")

        _LOGGER.debug(f"Optimizer Phase 1: Coarse grid search (step={self.CHARGE_STEP_COARSE})")
        skipped_infeasible = 0

        for percentile in self.PERCENTILE_COARSE:
            for num_charge in range(0, max_charge + 1, self.CHARGE_STEP_COARSE):
                for num_discharge in range(0, max_discharge + 1, self.DISCHARGE_STEP_COARSE):
                    if num_charge == 0 and num_discharge == 0:
                        continue  # Already calculated as baseline

                    # CRITICAL: Skip infeasible configurations (energy balance constraint)
                    if not self._is_energy_feasible(num_charge, num_discharge, test_config, suffix):
                        skipped_infeasible += 1
                        continue

                    iter_config = test_config.copy()
                    iter_config[f"charging_windows{suffix}"] = num_charge
                    iter_config[f"expensive_windows{suffix}"] = num_discharge
                    iter_config[f"percentile_threshold{suffix}"] = percentile

                    result = self._calculation_engine.calculate_windows(
                        raw_prices, iter_config, is_tomorrow, hass
                    )
                    iterations += 1
                    score = self._get_optimization_score(result, strategy)
                    cost = result.get("planned_total_cost", float('inf'))

                    # Track best per percentile (using score for comparison)
                    if percentile not in best_by_percentile or score > best_by_percentile[percentile][2]:
                        best_by_percentile[percentile] = (num_charge, num_discharge, score)

                    if score > best_score:  # Higher score is better
                        best_score = score
                        best_cost = cost
                        best_config = (num_charge, num_discharge, percentile)
                        best_result = result

        _LOGGER.debug(f"Optimizer Phase 1: Skipped {skipped_infeasible} infeasible configs")

        # Add results by percentile to decision tree
        coarse_best = best_config
        metric_label = "value" if strategy == "maximize_value" else "cost"
        decision_tree.append("Results by percentile:")
        for p in self.PERCENTILE_COARSE:
            if p in best_by_percentile:
                c, d, p_score = best_by_percentile[p]
                marker = " ← best" if (c, d, p) == coarse_best else ""
                # Display score appropriately (negative for minimize_cost shows as positive cost)
                display_val = p_score if strategy == "maximize_value" else -p_score
                decision_tree.append(f"  {p}%: {c}C/{d}D → €{display_val:.4f}{marker}")
            else:
                decision_tree.append(f"  {p}%: No feasible config")
        decision_tree.append("")

        if coarse_best != (0, 0, 25):
            display_best = best_score if strategy == "maximize_value" else best_cost
            decision_tree.append(f"Coarse best: {coarse_best[0]}C/{coarse_best[1]}D @ {coarse_best[2]}% → €{display_best:.4f} ({metric_label})")
        else:
            decision_tree.append("Coarse best: Baseline (no arbitrage profitable)")
        if skipped_infeasible > 0:
            decision_tree.append(f"Skipped: {skipped_infeasible} infeasible (discharge > available energy)")
        decision_tree.append("")

        # Phase 2: Fine refinement around best coarse result
        fine_iterations = 0
        if best_config != (0, 0, 25):  # Only refine if we found something better than baseline
            _LOGGER.debug(f"Optimizer Phase 2: Fine refinement around {best_config}")

            coarse_charge, coarse_discharge, coarse_percentile = best_config

            # Define fine search ranges
            charge_min = max(0, coarse_charge - self.FINE_OFFSET)
            charge_max_fine = min(max_charge, coarse_charge + self.FINE_OFFSET)
            discharge_min = max(0, coarse_discharge - self.FINE_OFFSET)
            discharge_max_fine = min(max_discharge, coarse_discharge + self.FINE_OFFSET)
            percentile_min = max(5, coarse_percentile - 3)
            percentile_max = min(50, coarse_percentile + 3)

            charge_range = range(charge_min, charge_max_fine + 1)
            discharge_range = range(discharge_min, discharge_max_fine + 1)
            percentile_range = range(percentile_min, percentile_max + 1, 3)

            decision_tree.append("--- Phase 2: Fine Refinement ---")
            decision_tree.append(f"Search around: {coarse_charge}C/{coarse_discharge}D @ {coarse_percentile}%")
            decision_tree.append(f"  Charge: {charge_min}-{charge_max_fine}")
            decision_tree.append(f"  Discharge: {discharge_min}-{discharge_max_fine}")
            decision_tree.append(f"  Percentile: {list(percentile_range)}")
            decision_tree.append("")

            for percentile in percentile_range:
                for num_charge in charge_range:
                    for num_discharge in discharge_range:
                        # Skip if already tested in coarse phase
                        if (num_charge % self.CHARGE_STEP_COARSE == 0 and
                            num_discharge % self.DISCHARGE_STEP_COARSE == 0 and
                            percentile in self.PERCENTILE_COARSE):
                            continue

                        # CRITICAL: Skip infeasible configurations (energy balance constraint)
                        if not self._is_energy_feasible(num_charge, num_discharge, test_config, suffix):
                            continue

                        iter_config = test_config.copy()
                        iter_config[f"charging_windows{suffix}"] = num_charge
                        iter_config[f"expensive_windows{suffix}"] = num_discharge
                        iter_config[f"percentile_threshold{suffix}"] = percentile

                        result = self._calculation_engine.calculate_windows(
                            raw_prices, iter_config, is_tomorrow, hass
                        )
                        iterations += 1
                        fine_iterations += 1
                        score = self._get_optimization_score(result, strategy)
                        cost = result.get("planned_total_cost", float('inf'))

                        if score > best_score:  # Higher score is better
                            best_score = score
                            best_cost = cost
                            best_config = (num_charge, num_discharge, percentile)
                            best_result = result

            if best_config != coarse_best:
                display_best = best_score if strategy == "maximize_value" else best_cost
                decision_tree.append(f"Improved: {best_config[0]}C/{best_config[1]}D @ {best_config[2]}% → €{display_best:.4f} ({metric_label})")
            else:
                decision_tree.append(f"No improvement (coarse was optimal)")
            decision_tree.append(f"Fine iterations: {fine_iterations}")
            decision_tree.append("")
        else:
            decision_tree.append("--- Phase 2: Skipped ---")
            decision_tree.append("Baseline was optimal, no refinement needed")
            decision_tree.append("")

        # Check if improvement meets minimum threshold
        # For minimize_cost: savings = baseline_cost - best_cost
        # For maximize_value: improvement = best_score - baseline_score (value improvement)
        if strategy == "maximize_value":
            # For maximize_value, compare scores (value improvement)
            improvement = best_score - baseline_score
            below_min_savings = improvement < min_daily_savings and best_config != (0, 0, 25)
        else:
            # For minimize_cost, use cost savings
            improvement = baseline_cost - best_cost
            below_min_savings = improvement < min_daily_savings and best_config != (0, 0, 25)

        # Keep traditional savings for logging
        savings = baseline_cost - best_cost

        # Capture what was found BEFORE threshold may revert to baseline
        found_charge = best_config[0]
        found_discharge = best_config[1]
        found_percentile = best_config[2]
        found_savings = improvement  # Use improvement for found_savings

        # Decision section
        decision_tree.append("--- Decision ---")
        improvement_label = "value improvement" if strategy == "maximize_value" else "savings"
        decision_tree.append(f"Potential {improvement_label}: €{improvement:.4f}")
        decision_tree.append(f"Min threshold: €{min_daily_savings:.2f}")

        if below_min_savings:
            _LOGGER.info(
                f"Optimizer: {improvement_label} {improvement:.4f} EUR < min {min_daily_savings:.2f} EUR, "
                f"using 0 windows (baseline)"
            )
            decision_tree.append(f"Result: Below threshold → 0 windows")
            decision_tree.append(f"Reason: €{improvement:.2f}/day not worth battery wear")
            # Revert to baseline but keep solar/buffer benefits
            best_config = (0, 0, 25)
            best_cost = baseline_cost
            best_result = baseline_result
            savings = 0.0
        elif best_config == (0, 0, 25):
            decision_tree.append(f"Result: 0 windows (baseline optimal)")
            decision_tree.append(f"Reason: No profitable arbitrage found")
        else:
            decision_tree.append(f"Result: {best_config[0]}C/{best_config[1]}D @ {best_config[2]}%")
            if strategy == "maximize_value":
                decision_tree.append(f"Value improvement: €{improvement:.4f}/day")
            else:
                decision_tree.append(f"Savings: €{savings:.4f}/day")

        elapsed_ms = (time.time() - start_time) * 1000
        decision_tree.append("")
        decision_tree.append(f"Completed in {elapsed_ms:.0f}ms ({iterations} iterations)")

        # Log energy balance info for transparency
        available_kwh = self._calculate_available_discharge_kwh(best_config[0], test_config, suffix)
        required_kwh = self._calculate_required_discharge_kwh(best_config[1], test_config)

        # Calculate total_value from best_result
        final_total_value = self._get_optimization_score(best_result, "maximize_value")

        _LOGGER.info(
            f"Optimizer ({strategy}): charge={best_config[0]}, discharge={best_config[1]}, "
            f"percentile={best_config[2]}, cost={best_cost:.4f}, value={final_total_value:.4f}, "
            f"savings={savings:.4f}, "
            f"energy_available={available_kwh:.2f}kWh, energy_required={required_kwh:.2f}kWh, "
            f"iterations={iterations}, time={elapsed_ms:.0f}ms"
        )

        return OptimizationResult(
            optimal_charge_windows=best_config[0],
            optimal_discharge_windows=best_config[1],
            optimal_percentile=best_config[2],
            planned_total_cost=best_cost,
            baseline_cost=baseline_cost,
            savings_vs_baseline=savings,
            iterations_checked=iterations,
            optimization_time_ms=elapsed_ms,
            result=best_result,
            decision_tree=decision_tree,
            below_min_savings=below_min_savings,
            found_charge_windows=found_charge,
            found_discharge_windows=found_discharge,
            found_percentile=found_percentile,
            found_savings=found_savings,
            total_value=final_total_value,
            optimization_strategy=strategy,
        )
