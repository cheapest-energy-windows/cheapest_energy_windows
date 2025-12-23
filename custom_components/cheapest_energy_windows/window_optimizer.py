"""Window optimizer for automatic window count selection.

Uses a two-phase grid search to minimize planned_total_cost:
- Phase 1 (Coarse): ~585 iterations with step=1
- Phase 2 (Fine): ~27 iterations around the best coarse result

Total: ~600 iterations, designed for sub-5-second execution.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .calculation_engine import WindowCalculationEngine
from .const import LOGGER_NAME

_LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass
class DecisionNode:
    """Node in the decision tree for debugging/transparency."""
    charge_windows: int
    discharge_windows: int
    percentile: int
    planned_cost: float
    is_optimal: bool = False


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
    decision_tree: List[DecisionNode] = field(default_factory=list)
    below_min_savings: bool = False


class WindowOptimizer:
    """Optimizer using two-phase search: coarse grid then fine refinement.

    Respects physical constraints and operational rules while bypassing
    user preference thresholds to find the true optimal configuration.

    CRITICAL: Enforces energy balance constraint - you cannot discharge
    more energy than is available from (buffer + solar + charging).
    """

    # Phase 1: Grid search (step=1 for comprehensive coverage)
    PERCENTILE_COARSE = [10, 20, 30, 40, 50]
    CHARGE_STEP_COARSE = 1
    DISCHARGE_STEP_COARSE = 1
    MAX_CHARGE = 12
    MAX_DISCHARGE = 8

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

    def _calculate_available_discharge_kwh(
        self,
        num_charge: int,
        config: Dict[str, Any],
        suffix: str = ""
    ) -> float:
        """Calculate maximum available energy for discharge.

        Available energy = buffer + solar + (charge_windows × charge_power × RTE)

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

        # Total available for discharge
        available_kwh = buffer_kwh + solar_kwh + charge_kwh

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

        # Bypass profit thresholds
        bypassed[f"min_profit_charge{suffix}"] = 0
        bypassed[f"min_profit_discharge{suffix}"] = 0
        bypassed["min_price_difference"] = 0

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
        decision_tree: List[DecisionNode] = []

        suffix = "_tomorrow" if is_tomorrow and config.get("tomorrow_settings_enabled", False) else ""
        min_daily_savings = config.get("min_daily_savings", 0.50)

        # Bypass thresholds to find true optimal
        test_config = self._bypass_thresholds(config, suffix)

        # Calculate baseline (0 windows = no battery action, just base usage)
        baseline_config = test_config.copy()
        baseline_config[f"charging_windows{suffix}"] = 0
        baseline_config[f"expensive_windows{suffix}"] = 0

        baseline_result = self._calculation_engine.calculate_windows(
            raw_prices, baseline_config, is_tomorrow, hass
        )
        baseline_cost = baseline_result.get("planned_total_cost", float('inf'))
        iterations += 1

        # Start with baseline as best
        best_cost = baseline_cost
        best_config: Tuple[int, int, int] = (0, 0, 25)  # (charge, discharge, percentile)
        best_result: Dict[str, Any] = baseline_result

        decision_tree.append(DecisionNode(
            charge_windows=0,
            discharge_windows=0,
            percentile=25,
            planned_cost=baseline_cost,
            is_optimal=False
        ))

        # Phase 1: Coarse grid search
        _LOGGER.debug(f"Optimizer Phase 1: Coarse grid search (step={self.CHARGE_STEP_COARSE})")
        skipped_infeasible = 0

        for percentile in self.PERCENTILE_COARSE:
            for num_charge in range(0, self.MAX_CHARGE + 1, self.CHARGE_STEP_COARSE):
                for num_discharge in range(0, self.MAX_DISCHARGE + 1, self.DISCHARGE_STEP_COARSE):
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
                    cost = result.get("planned_total_cost", float('inf'))

                    decision_tree.append(DecisionNode(
                        charge_windows=num_charge,
                        discharge_windows=num_discharge,
                        percentile=percentile,
                        planned_cost=cost,
                        is_optimal=False
                    ))

                    if cost < best_cost:
                        best_cost = cost
                        best_config = (num_charge, num_discharge, percentile)
                        best_result = result

        _LOGGER.debug(f"Optimizer Phase 1: Skipped {skipped_infeasible} infeasible configs")

        # Phase 2: Fine refinement around best coarse result
        if best_config != (0, 0, 25):  # Only refine if we found something better than baseline
            _LOGGER.debug(f"Optimizer Phase 2: Fine refinement around {best_config}")

            coarse_charge, coarse_discharge, coarse_percentile = best_config

            # Define fine search ranges
            charge_range = range(
                max(0, coarse_charge - self.FINE_OFFSET),
                min(self.MAX_CHARGE + 1, coarse_charge + self.FINE_OFFSET + 1)
            )
            discharge_range = range(
                max(0, coarse_discharge - self.FINE_OFFSET),
                min(self.MAX_DISCHARGE + 1, coarse_discharge + self.FINE_OFFSET + 1)
            )
            # +/- 5 percentile points
            percentile_range = range(
                max(5, coarse_percentile - 5),
                min(51, coarse_percentile + 6),
                5
            )

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
                        cost = result.get("planned_total_cost", float('inf'))

                        decision_tree.append(DecisionNode(
                            charge_windows=num_charge,
                            discharge_windows=num_discharge,
                            percentile=percentile,
                            planned_cost=cost,
                            is_optimal=False
                        ))

                        if cost < best_cost:
                            best_cost = cost
                            best_config = (num_charge, num_discharge, percentile)
                            best_result = result

        # Mark the optimal node in decision tree
        for node in decision_tree:
            if (node.charge_windows == best_config[0] and
                node.discharge_windows == best_config[1] and
                node.percentile == best_config[2]):
                node.is_optimal = True
                break

        # Check if savings meet minimum threshold
        savings = baseline_cost - best_cost
        below_min_savings = savings < min_daily_savings and best_config != (0, 0, 25)

        if below_min_savings:
            _LOGGER.info(
                f"Optimizer: Savings {savings:.4f} EUR < min {min_daily_savings:.2f} EUR, "
                f"using 0 windows (baseline)"
            )
            # Revert to baseline but keep solar/buffer benefits
            best_config = (0, 0, 25)
            best_cost = baseline_cost
            best_result = baseline_result
            savings = 0.0

        elapsed_ms = (time.time() - start_time) * 1000

        # Log energy balance info for transparency
        available_kwh = self._calculate_available_discharge_kwh(best_config[0], test_config, suffix)
        required_kwh = self._calculate_required_discharge_kwh(best_config[1], test_config)

        _LOGGER.info(
            f"Optimizer: charge={best_config[0]}, discharge={best_config[1]}, "
            f"percentile={best_config[2]}, cost={best_cost:.4f}, "
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
            below_min_savings=below_min_savings
        )
