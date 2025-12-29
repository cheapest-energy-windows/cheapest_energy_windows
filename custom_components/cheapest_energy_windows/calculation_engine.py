"""Calculation engine for Cheapest Energy Windows."""
from __future__ import annotations

from datetime import datetime, timedelta, time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from homeassistant.util import dt as dt_util

from .const import (
    LOGGER_NAME,
    PRICING_15_MINUTES,
    PRICING_1_HOUR,
    MODE_CHARGE,
    MODE_DISCHARGE,
    MODE_NORMAL,
    MODE_OFF,
    STATE_CHARGE,
    STATE_DISCHARGE,
    STATE_NORMAL,
    STATE_OFF,
    DEFAULT_PRICE_COUNTRY,
    DEFAULT_MIN_SELL_PRICE,
    DEFAULT_USE_MIN_SELL_PRICE,
    DEFAULT_MIN_SELL_PRICE_BYPASS_SPREAD,
    DEFAULT_SELL_FORMULA_PARAM_A,
    DEFAULT_SELL_FORMULA_PARAM_B,
    DEFAULT_BUY_FORMULA_PARAM_A,
    DEFAULT_BUY_FORMULA_PARAM_B,
    DEFAULT_VAT_RATE,
    DEFAULT_TAX,
    DEFAULT_ADDITIONAL_COST,
)
from .formulas import get_formula

_LOGGER = logging.getLogger(LOGGER_NAME)


class WindowCalculationEngine:
    """High-performance window selection engine."""

    def __init__(self) -> None:
        """Initialize the calculation engine."""
        pass

    def calculate_windows(
        self,
        raw_prices: List[Dict[str, Any]],
        config: Dict[str, Any],
        is_tomorrow: bool = False,
        hass: Any = None,
        energy_statistics: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Calculate optimal charging/discharging windows.

        Args:
            raw_prices: List of price data from NordPool or similar
            config: Configuration from input entities
            is_tomorrow: Whether calculating for tomorrow
            hass: Home Assistant instance (for sensor value lookup)

        Returns:
            Dictionary with calculated windows and attributes
        """
        # Get configuration values
        pricing_mode = config.get("pricing_window_duration", PRICING_15_MINUTES)

        # Use tomorrow's config if applicable
        suffix = "_tomorrow" if is_tomorrow and config.get("tomorrow_settings_enabled", False) else ""

        num_charge_windows = int(config.get(f"charging_windows{suffix}", 4))
        num_discharge_windows = int(config.get(f"expensive_windows{suffix}", 4))
        percentile_threshold = config.get(f"percentile_threshold{suffix}", 25)
        min_profit_charge = config.get(f"min_profit_charge{suffix}", 10)
        min_profit_discharge = config.get(f"min_profit_discharge{suffix}", 10)
        min_price_diff_enabled = config.get("min_price_diff_enabled", True)  # Global setting
        min_price_diff = config.get("min_price_difference", 0.05) if min_price_diff_enabled else float('-inf')  # Global
        # Calculate RTE loss for profit checks
        battery_rte = config.get("battery_rte", 85)
        rte_loss = 100 - battery_rte

        # Process prices based on mode (uses buy price formula from config)
        processed_prices = self._process_prices(raw_prices, pricing_mode, config)

        if not processed_prices:
            _LOGGER.debug("No prices to process")
            return self._empty_result(is_tomorrow)

        # Store all prices for base usage calculations (before any filtering)
        # Base usage should always be calculated over full 24h, not just calculation window
        all_prices = processed_prices.copy()

        # Apply calculation window filter if enabled (use suffix for tomorrow settings)
        # This only affects charge/discharge window selection, NOT base usage calculations
        calc_window_enabled = config.get(f"calculation_window_enabled{suffix}", False)
        if calc_window_enabled:
            calc_window_start = config.get(f"calculation_window_start{suffix}", "00:00:00")
            calc_window_end = config.get(f"calculation_window_end{suffix}", "23:59:59")
            processed_prices = self._filter_prices_by_calculation_window(
                processed_prices,
                calc_window_start,
                calc_window_end
            )
            if not processed_prices:
                return self._empty_result(is_tomorrow)

        # Calculate arbitrage_avg for profit display
        arbitrage_avg = self._calculate_arbitrage_avg(processed_prices, config, is_tomorrow)

        # Pre-filter prices based on time override to prevent normal/off periods from being selected
        # This ensures that windows calculations respect time overrides from the start
        time_override_enabled = config.get(f"time_override_enabled{suffix}", False)
        prices_for_charge_calc = processed_prices
        prices_for_discharge_calc = processed_prices

        if time_override_enabled:
            override_mode = config.get(f"time_override_mode{suffix}", MODE_NORMAL)

            # Get time values and ensure they're in string format
            override_start = config.get(f"time_override_start{suffix}", "")
            override_end = config.get(f"time_override_end{suffix}", "")

            # Convert to string format if needed
            if hasattr(override_start, 'strftime'):
                override_start_str = override_start.strftime("%H:%M:%S")
            elif override_start:
                override_start_str = str(override_start)
            else:
                override_start_str = ""

            if hasattr(override_end, 'strftime'):
                override_end_str = override_end.strftime("%H:%M:%S")
            elif override_end:
                override_end_str = str(override_end)
            else:
                override_end_str = ""

            if override_start_str and override_end_str:
                # For normal/off modes, exclude override periods from window calculations
                if override_mode in [MODE_NORMAL, MODE_OFF]:
                    filtered_prices = []
                    for price_data in processed_prices:
                        if not self._is_in_time_range(price_data["timestamp"], override_start_str, override_end_str):
                            filtered_prices.append(price_data)
                    prices_for_charge_calc = filtered_prices
                    prices_for_discharge_calc = filtered_prices

                # For charge mode, only charge windows should be in override period
                elif override_mode == MODE_CHARGE:
                    # Charge windows: only consider prices within override period
                    charge_override_prices = []
                    for price_data in processed_prices:
                        if self._is_in_time_range(price_data["timestamp"], override_start_str, override_end_str):
                            charge_override_prices.append(price_data)
                    prices_for_charge_calc = charge_override_prices
                    # Discharge windows: exclude override period
                    discharge_filtered = []
                    for price_data in processed_prices:
                        if not self._is_in_time_range(price_data["timestamp"], override_start_str, override_end_str):
                            discharge_filtered.append(price_data)
                    prices_for_discharge_calc = discharge_filtered

                # For discharge mode, only discharge windows should be in override period
                elif override_mode == MODE_DISCHARGE:
                    # Charge windows: exclude override period
                    charge_filtered = []
                    for price_data in processed_prices:
                        if not self._is_in_time_range(price_data["timestamp"], override_start_str, override_end_str):
                            charge_filtered.append(price_data)
                    prices_for_charge_calc = charge_filtered
                    # Discharge windows: only consider prices within override period
                    discharge_override_prices = []
                    for price_data in processed_prices:
                        if self._is_in_time_range(price_data["timestamp"], override_start_str, override_end_str):
                            discharge_override_prices.append(price_data)
                    prices_for_discharge_calc = discharge_override_prices

        # Find windows using the pre-filtered prices
        charge_windows = self._find_charge_windows(
            prices_for_charge_calc,  # Use filtered prices
            num_charge_windows,
            percentile_threshold,
            min_profit_charge,
            rte_loss,
            min_price_diff
        )

        # Also get ALL candidates that pass profit threshold (for capacity-first selection)
        # Chrono simulation will use all candidates to determine feasibility based on battery capacity
        # After chrono, we select the cheapest N from feasible windows
        all_charge_candidates = self._find_charge_windows(
            prices_for_charge_calc,
            num_charge_windows,  # Not used when return_all=True
            percentile_threshold,
            min_profit_charge,
            rte_loss,
            min_price_diff,
            return_all=True
        )

        # Get sell price configuration
        use_min_sell = config.get("use_min_sell_price", DEFAULT_USE_MIN_SELL_PRICE)
        bypass_spread = config.get("min_sell_price_bypass_spread", DEFAULT_MIN_SELL_PRICE_BYPASS_SPREAD)

        # If bypass_spread is enabled, set thresholds to allow any profit (including negative)
        # Use -inf to truly bypass the profit check for negative arbitrage scenarios
        effective_min_profit_discharge = float('-inf') if bypass_spread else min_profit_discharge
        effective_min_price_diff_discharge = float('-inf') if bypass_spread else min_price_diff

        discharge_windows = self._find_discharge_windows(
            prices_for_discharge_calc,  # Use filtered prices
            charge_windows,
            num_discharge_windows,
            percentile_threshold,
            effective_min_profit_discharge,
            rte_loss,
            effective_min_price_diff_discharge,
            config  # Pass config for sell price calculation
        )

        # Apply minimum sell price filter to discharge windows (only if use_min_sell is enabled)
        discharge_windows = self._filter_discharge_by_min_sell_price(
            discharge_windows,
            processed_prices,
            config
        )

        # Calculate current state (pass arbitrage_avg and is_tomorrow for RTE protection)
        # Note: arbitrage_avg was already calculated earlier for the protection check
        current_state = self._determine_current_state(
            processed_prices,
            charge_windows,
            discharge_windows,
            config,
            arbitrage_avg,
            is_tomorrow
        )

        # Build result
        result = self._build_result(
            processed_prices,
            all_prices,
            charge_windows,
            discharge_windows,
            current_state,
            config,
            is_tomorrow,
            hass,
            all_charge_candidates,  # Pass all candidates for capacity-first selection
            energy_statistics  # HA Energy Dashboard stats
        )

        return result

    def _process_prices(
        self,
        raw_prices: List[Dict[str, Any]],
        pricing_mode: str,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process raw prices using buy price formula from config."""
        # Get buy price formula settings
        buy_country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
        buy_param_a = config.get("buy_formula_param_a", DEFAULT_BUY_FORMULA_PARAM_A)
        buy_param_b = config.get("buy_formula_param_b", DEFAULT_BUY_FORMULA_PARAM_B)
        vat = config.get("vat", DEFAULT_VAT_RATE)
        tax = config.get("tax", DEFAULT_TAX)
        additional_cost = config.get("additional_cost", DEFAULT_ADDITIONAL_COST)

        processed = []

        if pricing_mode == PRICING_1_HOUR:
            # Group by hour and average
            hourly_prices = {}
            for item in raw_prices:
                try:
                    # Validate item is a dict
                    if not isinstance(item, dict):
                        _LOGGER.error(f"Item is not a dict! Type: {type(item)}, Value: {item}")
                        continue

                    # Parse timestamp - handle both datetime objects and strings
                    start_value = item.get("start")
                    if not start_value:
                        _LOGGER.warning(f"Item has no 'start' key: {item}")
                        continue

                    if isinstance(start_value, datetime):
                        # Already a datetime object (new Nordpool format)
                        timestamp = start_value
                    elif isinstance(start_value, str):
                        # String format (old format)
                        timestamp_str = start_value.replace('"', '')
                        timestamp = datetime.fromisoformat(timestamp_str)
                    else:
                        _LOGGER.error(f"Unexpected start type: {type(start_value)}, Value: {start_value}")
                        continue

                    hour = timestamp.replace(minute=0, second=0, microsecond=0)

                    if hour not in hourly_prices:
                        hourly_prices[hour] = []

                    # Calculate total price using buy price formula
                    base_price = item.get("value", 0)
                    total_price = self._calculate_buy_price(
                        base_price, buy_country, buy_param_a, buy_param_b,
                        vat, tax, additional_cost
                    )
                    hourly_prices[hour].append(total_price)

                except (ValueError, TypeError, AttributeError) as e:
                    _LOGGER.error(f"Failed to process price item: {e}", exc_info=True)
                    _LOGGER.error(f"Problematic item: {item}")
                    continue

            # Average hourly prices
            # Also track raw prices for sell price calculation
            hourly_raw_prices = {}
            for item in raw_prices:
                try:
                    start_value = item.get("start")
                    if isinstance(start_value, datetime):
                        timestamp = start_value
                    elif isinstance(start_value, str):
                        timestamp = datetime.fromisoformat(start_value.replace('"', ''))
                    else:
                        continue
                    hour = timestamp.replace(minute=0, second=0, microsecond=0)
                    if hour not in hourly_raw_prices:
                        hourly_raw_prices[hour] = []
                    hourly_raw_prices[hour].append(item.get("value", 0))
                except Exception:
                    continue

            for hour, prices in hourly_prices.items():
                if prices:
                    raw_avg = float(np.mean(hourly_raw_prices.get(hour, prices)))
                    processed.append({
                        "timestamp": hour,
                        "price": float(np.mean(prices)),  # Convert numpy.float64 to Python float
                        "raw_price": raw_avg,  # Store raw price for sell price calculation
                        "duration": 60  # 60 minutes
                    })

        else:  # 15-minute mode
            for item in raw_prices:
                try:
                    # Validate item is a dict
                    if not isinstance(item, dict):
                        _LOGGER.error(f"Item is not a dict! Type: {type(item)}, Value: {item}")
                        continue

                    # Parse timestamp - handle both datetime objects and strings
                    start_value = item.get("start")
                    if not start_value:
                        _LOGGER.warning(f"Item has no 'start' key: {item}")
                        continue

                    if isinstance(start_value, datetime):
                        # Already a datetime object (new Nordpool format)
                        timestamp = start_value
                    elif isinstance(start_value, str):
                        # String format (old format)
                        timestamp_str = start_value.replace('"', '')
                        timestamp = datetime.fromisoformat(timestamp_str)
                    else:
                        _LOGGER.error(f"Unexpected start type: {type(start_value)}, Value: {start_value}")
                        continue

                    base_price = item.get("value", 0)
                    total_price = self._calculate_buy_price(
                        base_price, buy_country, buy_param_a, buy_param_b,
                        vat, tax, additional_cost
                    )

                    processed.append({
                        "timestamp": timestamp,
                        "price": total_price,
                        "raw_price": base_price,  # Store raw price for sell price calculation
                        "duration": 15  # 15 minutes
                    })

                except (ValueError, TypeError, AttributeError) as e:
                    _LOGGER.error(f"Failed to process price item: {e}", exc_info=True)
                    _LOGGER.error(f"Problematic item: {item}")
                    continue

        # Sort by timestamp
        processed.sort(key=lambda x: x["timestamp"])

        return processed

    def _filter_prices_by_calculation_window(
        self,
        prices: List[Dict[str, Any]],
        start_str: str,
        end_str: str
    ) -> List[Dict[str, Any]]:
        """Filter prices to only include those within the calculation window time range.

        This restricts the price analysis to a specific time window each day.
        For example, if you only want to charge/discharge between 06:00-22:00,
        set the calculation window to those times.
        """
        if not prices:
            return prices

        filtered = []

        try:
            # Parse time strings (HH:MM:SS format)
            start_parts = start_str.split(":")
            end_parts = end_str.split(":")

            start_hour = int(start_parts[0])
            start_minute = int(start_parts[1])
            end_hour = int(end_parts[0])
            end_minute = int(end_parts[1])

            for price_data in prices:
                timestamp = price_data["timestamp"]
                price_hour = timestamp.hour
                price_minute = timestamp.minute

                # Convert to minutes since midnight for easier comparison
                price_time = price_hour * 60 + price_minute
                start_time = start_hour * 60 + start_minute
                end_time = end_hour * 60 + end_minute

                # Handle overnight periods
                if end_time < start_time:
                    # Overnight: include if time >= start OR time < end
                    if price_time >= start_time or price_time < end_time:
                        filtered.append(price_data)
                else:
                    # Same day: include if start <= time < end
                    if start_time <= price_time < end_time:
                        filtered.append(price_data)

        except (ValueError, IndexError, AttributeError) as e:
            _LOGGER.error(f"Failed to parse calculation window times: {e}")
            return prices  # Return unfiltered on error

        return filtered

    def _find_charge_windows(
        self,
        prices: List[Dict[str, Any]],
        num_windows: int,
        percentile_threshold: float,
        min_profit: float,
        rte_loss: float,
        min_price_diff: float,
        return_all: bool = False
    ) -> List[Dict[str, Any]]:
        """Find cheapest windows for charging.

        Uses percentile_threshold symmetrically:
        - Candidates: prices in the bottom percentile_threshold% (cheapest)
        - Spread comparison: against average of top percentile_threshold% (most expensive)

        Uses profit-based threshold (profit = spread - RTE_loss):
        - Buy-buy spread: if we buy now vs. buy later, how much do we save?
        - RTE loss applies because energy stored now loses efficiency

        Args:
            return_all: If True, return ALL candidates that pass the profit threshold,
                       ignoring num_windows limit. Used for capacity-first selection.
        """
        if not prices or (num_windows <= 0 and not return_all):
            return []

        # Convert to numpy array for efficient operations
        price_array = np.array([p["price"] for p in prices])

        # Calculate percentile threshold for cheap prices (bottom X%)
        cheap_threshold = np.percentile(price_array, percentile_threshold)

        # Get candidates below threshold
        candidates = []
        for i, price_data in enumerate(prices):
            if price_data["price"] <= cheap_threshold:
                candidates.append({
                    "index": i,
                    "timestamp": price_data["timestamp"],
                    "price": price_data["price"],
                    "duration": price_data["duration"]
                })

        # Sort by price
        candidates.sort(key=lambda x: x["price"])

        # Progressive selection with profit check
        # Compare against top percentile_threshold% (most expensive)
        selected = []
        expensive_threshold = np.percentile(price_array, 100 - percentile_threshold)
        expensive_prices = price_array[price_array >= expensive_threshold]
        expensive_avg = np.mean(expensive_prices) if len(expensive_prices) > 0 else np.max(price_array)
        expensive_max = np.max(expensive_prices) if len(expensive_prices) > 0 else np.max(price_array)

        for candidate in candidates:
            # Only enforce num_windows limit if not returning all candidates
            if not return_all and len(selected) >= num_windows:
                break

            # Calculate spread using this candidate's individual price (not running average)
            # This ensures each window is judged on its own merit, preventing the paradox
            # where increasing max windows can result in fewer selected windows
            # Profit = spread - RTE loss (energy stored now loses efficiency)
            if candidate["price"] > 0:
                spread_pct = ((expensive_avg - candidate["price"]) / candidate["price"]) * 100
                profit_pct = spread_pct - rte_loss  # Apply RTE loss to get true profit
                price_diff = expensive_max - candidate["price"]  # Per-window: max expensive price minus this charge price

                if profit_pct >= min_profit and price_diff >= min_price_diff:
                    selected.append(candidate)

        return selected

    def _find_discharge_windows(
        self,
        prices: List[Dict[str, Any]],
        charge_windows: List[Dict[str, Any]],
        num_windows: int,
        percentile_threshold: float,
        min_profit: float,
        rte_loss: float,
        min_price_diff: float,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find expensive windows for discharging based on SELL prices.

        Discharge = selling to grid, so we select windows with highest SELL prices.
        This applies to all countries - future-proofs for sell cost additions.

        Uses percentile_threshold symmetrically:
        - Candidates: SELL prices in the top percentile_threshold% (most expensive)
        - Spread comparison: sell price avg vs buy price avg (charge windows)

        Uses profit-based threshold (profit = spread - RTE_loss):
        - Buy-sell spread: buy cheap, sell expensive
        - RTE loss applies because we lose efficiency in the round-trip
        """
        if not prices or num_windows <= 0:
            return []

        # Get sell formula config
        sell_country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
        sell_param_a = config.get("sell_formula_param_a", DEFAULT_SELL_FORMULA_PARAM_A)
        sell_param_b = config.get("sell_formula_param_b", DEFAULT_SELL_FORMULA_PARAM_B)

        # Exclude charging times
        charge_indices = {w["index"] for w in charge_windows}

        # Filter out charging windows and calculate sell prices
        available_prices = []
        for i, price_data in enumerate(prices):
            if i not in charge_indices:
                raw_price = price_data.get("raw_price", price_data["price"])
                sell_price = self._calculate_sell_price(
                    raw_price, price_data["price"], sell_country, sell_param_a, sell_param_b
                )
                available_prices.append({
                    "index": i,
                    "timestamp": price_data["timestamp"],
                    "price": price_data["price"],  # Buy price (for reference)
                    "raw_price": raw_price,
                    "sell_price": sell_price,  # Sell price (for selection)
                    "duration": price_data["duration"]
                })

        if not available_prices:
            return []

        # Use SELL prices for percentile threshold (discharge = selling)
        sell_price_array = np.array([p["sell_price"] for p in available_prices])

        # Calculate percentile threshold for expensive SELL prices (top X%)
        expensive_threshold = np.percentile(sell_price_array, 100 - percentile_threshold)

        # Calculate fixed expensive_avg from top percentile (symmetric with charge)
        expensive_sell_prices = sell_price_array[sell_price_array >= expensive_threshold]
        expensive_avg = np.mean(expensive_sell_prices) if len(expensive_sell_prices) > 0 else np.max(sell_price_array)

        # Get candidates above threshold (by SELL price)
        candidates = []
        for price_data in available_prices:
            if price_data["sell_price"] >= expensive_threshold:
                candidates.append(price_data)

        # Sort by SELL price (descending for discharge)
        candidates.sort(key=lambda x: x["sell_price"], reverse=True)

        # Progressive selection with spread check
        # Compare SELL price avg against BUY price avg (charge windows)
        selected = []
        if charge_windows:
            cheap_avg = np.mean([w["price"] for w in charge_windows])  # Buy prices
            cheap_min = min(w["price"] for w in charge_windows)
        else:
            # No charge windows - use bottom percentile_threshold% buy prices as reference
            buy_price_array = np.array([p["price"] for p in available_prices])
            cheap_threshold = np.percentile(buy_price_array, percentile_threshold)
            cheap_prices = buy_price_array[buy_price_array <= cheap_threshold]
            cheap_avg = np.mean(cheap_prices) if len(cheap_prices) > 0 else np.min(buy_price_array)
            cheap_min = np.min(cheap_prices) if len(cheap_prices) > 0 else np.min(buy_price_array)

        for candidate in candidates:
            if len(selected) >= num_windows:
                break

            # Calculate spread and profit using THIS candidate's individual sell price
            # Compare candidate's sell price against average charge window buy price
            # Profit = spread - RTE loss (accounts for battery efficiency)
            # This mirrors charge window logic which uses individual candidate prices
            if cheap_avg > 0:
                spread_pct = ((candidate["sell_price"] - cheap_avg) / cheap_avg) * 100
                profit_pct = spread_pct - rte_loss  # Apply RTE loss to get true profit

                if profit_pct >= min_profit:
                    selected.append(candidate)

        return selected

    def _calculate_arbitrage_avg(
        self,
        prices: List[Dict[str, Any]],
        config: Dict[str, Any],
        is_tomorrow: bool
    ) -> float:
        """Calculate arbitrage_avg (sell vs buy spread percentage) for RTE protection check.

        This calculates the potential arbitrage margin based on percentile thresholds,
        independent of whether windows are actually selected.
        """
        if not prices:
            return 0.0

        suffix = "_tomorrow" if is_tomorrow and config.get("tomorrow_settings_enabled", False) else ""
        percentile_threshold = config.get(f"percentile_threshold{suffix}", 25)

        # Get sell price formula settings
        sell_country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
        sell_param_a = config.get("sell_formula_param_a", DEFAULT_SELL_FORMULA_PARAM_A)
        sell_param_b = config.get("sell_formula_param_b", DEFAULT_SELL_FORMULA_PARAM_B)

        all_buy_prices = np.array([p["price"] for p in prices])

        # Get cheapest buy prices (bottom percentile)
        cheap_threshold = np.percentile(all_buy_prices, percentile_threshold)
        cheap_buy_prices = all_buy_prices[all_buy_prices <= cheap_threshold]
        avg_cheap_buy = float(np.mean(cheap_buy_prices)) if len(cheap_buy_prices) > 0 else 0.0

        if avg_cheap_buy <= 0:
            return 0.0

        # Calculate sell prices for all time slots
        all_sell_prices = []
        for p in prices:
            raw = p.get("raw_price", p["price"])
            sell = self._calculate_sell_price(raw, p["price"], sell_country, sell_param_a, sell_param_b)
            all_sell_prices.append(sell)
        all_sell_prices = np.array(all_sell_prices)

        # Get most expensive sell prices (top percentile)
        expensive_sell_threshold = np.percentile(all_sell_prices, 100 - percentile_threshold)
        expensive_sell_prices = all_sell_prices[all_sell_prices >= expensive_sell_threshold]
        avg_expensive_sell = float(np.mean(expensive_sell_prices)) if len(expensive_sell_prices) > 0 else 0.0

        # Calculate arbitrage: (sell - buy) / buy * 100
        return float(((avg_expensive_sell - avg_cheap_buy) / avg_cheap_buy) * 100)

    def _determine_current_state(
        self,
        prices: List[Dict[str, Any]],
        charge_windows: List[Dict[str, Any]],
        discharge_windows: List[Dict[str, Any]],
        config: Dict[str, Any],
        arbitrage_avg: float = 0.0,
        is_tomorrow: bool = False,
        rte_preserved_periods: List[Dict[str, Any]] = None
    ) -> str:
        """Determine current state based on time and configuration.

        Profit thresholds control window qualification - if profit is below
        threshold, no windows are selected and system defaults to normal mode.
        """
        # Check if automation is enabled
        if not config.get("automation_enabled", True):
            return STATE_OFF

        now = dt_util.now()
        current_time = now.replace(second=0, microsecond=0)

        # Check time override
        if config.get("time_override_enabled", False):
            start_str = config.get("time_override_start", "")
            end_str = config.get("time_override_end", "")
            mode = config.get("time_override_mode", MODE_NORMAL)

            if self._is_in_time_range(current_time, start_str, end_str):
                return self._mode_to_state(mode)

        # Check price override
        if config.get("price_override_enabled", False):
            threshold = config.get("price_override_threshold", 0.15)
            current_price = self._get_current_price(prices, current_time)
            if current_price and current_price <= threshold:
                return STATE_CHARGE

        # Check scheduled windows
        for window in discharge_windows:
            if self._is_window_active(window, current_time):
                return STATE_DISCHARGE

        for window in charge_windows:
            if self._is_window_active(window, current_time):
                return STATE_CHARGE

        # Check if current time is in an RTE-preserved period (battery held due to low price)
        # RTE-preserved periods should trigger OFF state so automations stop grid matching
        if rte_preserved_periods:
            for period in rte_preserved_periods:
                period_time = period["timestamp"]
                # Handle both datetime objects and ISO strings
                if isinstance(period_time, str):
                    period_time = datetime.fromisoformat(period_time.replace('Z', '+00:00'))
                period_end = period_time + timedelta(minutes=period["duration"])
                if period_time <= current_time < period_end:
                    return STATE_OFF

        return STATE_NORMAL

    def _is_window_active(self, window: Dict[str, Any], current_time: datetime) -> bool:
        """Check if a window is currently active."""
        window_time = window["timestamp"]
        window_duration = window["duration"]

        # Check if current time falls within the window
        window_start = window_time
        window_end = window_time + timedelta(minutes=window_duration)

        return window_start <= current_time < window_end

    def _is_in_time_range(self, current_time: datetime, start_str: str, end_str: str) -> bool:
        """Check if current time is within a time range."""
        try:
            # Parse time strings (HH:MM:SS format)
            start_parts = start_str.split(":")
            end_parts = end_str.split(":")

            start_time = current_time.replace(
                hour=int(start_parts[0]),
                minute=int(start_parts[1]),
                second=0
            )
            end_time = current_time.replace(
                hour=int(end_parts[0]),
                minute=int(end_parts[1]),
                second=0
            )

            # Handle overnight periods
            if end_time < start_time:
                return current_time >= start_time or current_time < end_time
            else:
                return start_time <= current_time < end_time

        except (ValueError, IndexError, AttributeError):
            return False

    def _get_current_price(
        self, prices: List[Dict[str, Any]], current_time: datetime
    ) -> Optional[float]:
        """Get the current price."""
        for price_data in prices:
            if self._is_window_active(price_data, current_time):
                return price_data["price"]
        return None

    def _calculate_buy_price(
        self,
        spot_price_mwh: float,
        country: str,
        param_a: float,
        param_b: float,
        vat: float,
        tax: float,
        additional_cost: float,
    ) -> float:
        """Calculate buy price based on country formula with configurable params.

        Uses the formula registry to look up the country-specific formula.
        Falls back to raw price if country not found.

        Args:
            spot_price_mwh: Raw spot price in EUR/kWh (from price sensor)
                Note: Variable named 'mwh' for historical reasons but sensor provides EUR/kWh
            country: Country/formula selection (e.g., "netherlands", "belgium_engie")
            param_a: Cost component A in EUR/kWh
            param_b: Multiplier B
            vat: VAT rate as percentage (0-100)
            tax: Energy tax in EUR/kWh
            additional_cost: Additional cost in EUR/kWh

        Returns:
            Calculated buy price (EUR/kWh)
        """
        formula = get_formula(country)
        if formula:
            # Build params dict from all available parameters
            params = {
                "vat": vat,
                "tax": tax,
                "additional_cost": additional_cost,
                "param_a": param_a,
                "param_b": param_b,
            }
            return formula.buy_formula(spot_price_mwh, params)

        # Fallback to raw price if formula not found
        _LOGGER.warning(f"Unknown buy price country '{country}', using raw price")
        return max(0, spot_price_mwh)

    def _calculate_sell_price(
        self,
        spot_price_mwh: float,
        buy_price: float,
        country: str,
        param_a: float,
        param_b: float,
    ) -> float:
        """Calculate sell price based on country formula with configurable params.

        Uses the formula registry to look up the country-specific formula.
        Falls back to buy price if country not found.

        Args:
            spot_price_mwh: Raw spot price in EUR/kWh (from price sensor)
                Note: Variable named 'mwh' for historical reasons but sensor provides EUR/kWh
            buy_price: Calculated buy price with VAT/tax/additional (EUR/kWh)
            country: Country/formula selection (e.g., "netherlands", "belgium_engie")
            param_a: Cost component A in EUR/kWh
            param_b: Multiplier B

        Returns:
            Calculated sell price (EUR/kWh)
        """
        formula = get_formula(country)
        if formula:
            # Build params dict from available parameters
            params = {
                "param_a": param_a,
                "param_b": param_b,
            }
            return formula.sell_formula(spot_price_mwh, buy_price, params)

        # Fallback to buy price if formula not found
        _LOGGER.warning(f"Unknown sell price country '{country}', using buy price")
        return buy_price

    def _get_current_sell_price(
        self, prices: List[Dict[str, Any]], current_time: datetime, config: Dict[str, Any]
    ) -> Optional[float]:
        """Get the current sell price."""
        for price_data in prices:
            if self._is_window_active(price_data, current_time):
                country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
                param_a = config.get("sell_formula_param_a", DEFAULT_SELL_FORMULA_PARAM_A)
                param_b = config.get("sell_formula_param_b", DEFAULT_SELL_FORMULA_PARAM_B)
                return self._calculate_sell_price(
                    price_data.get("raw_price", price_data["price"]),
                    price_data["price"],
                    country,
                    param_a,
                    param_b,
                )
        return None

    def _filter_discharge_by_min_sell_price(
        self,
        discharge_windows: List[Dict[str, Any]],
        prices: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Filter discharge windows by minimum sell price.

        Args:
            discharge_windows: List of selected discharge windows
            prices: Full list of processed prices (to get raw_price)
            config: Configuration dictionary

        Returns:
            Filtered list of discharge windows
        """
        use_min_sell = config.get("use_min_sell_price", DEFAULT_USE_MIN_SELL_PRICE)
        if not use_min_sell:
            return discharge_windows

        min_sell_price = config.get("min_sell_price", DEFAULT_MIN_SELL_PRICE)
        country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
        param_a = config.get("sell_formula_param_a", DEFAULT_SELL_FORMULA_PARAM_A)
        param_b = config.get("sell_formula_param_b", DEFAULT_SELL_FORMULA_PARAM_B)

        filtered = []
        # Build a lookup for raw prices by timestamp
        price_lookup = {p["timestamp"]: p for p in prices}

        for window in discharge_windows:
            # Get the raw price from the prices list
            price_data = price_lookup.get(window["timestamp"], {})
            raw_price = price_data.get("raw_price", window["price"])
            buy_price = window["price"]

            # Calculate sell price
            sell_price = self._calculate_sell_price(
                raw_price,
                buy_price,
                country,
                param_a,
                param_b,
            )

            # Check if sell price meets minimum
            if sell_price >= min_sell_price:
                # Store sell price in window for later use
                window["sell_price"] = sell_price
                filtered.append(window)
            else:
                _LOGGER.debug(
                    f"Filtering out discharge window at {window['timestamp']}: "
                    f"sell_price {sell_price:.4f} < min_sell_price {min_sell_price:.4f}"
                )

        return filtered

    def _mode_to_state(self, mode: str) -> str:
        """Convert override mode to state."""
        mode_map = {
            MODE_NORMAL: STATE_NORMAL,
            MODE_CHARGE: STATE_CHARGE,
            MODE_DISCHARGE: STATE_DISCHARGE,
            MODE_OFF: STATE_OFF,
        }
        return mode_map.get(mode, STATE_NORMAL)

    def _calculate_actual_windows(
        self,
        prices: List[Dict[str, Any]],
        charge_windows: List[Dict[str, Any]],
        discharge_windows: List[Dict[str, Any]],
        config: Dict[str, Any],
        is_tomorrow: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Calculate actual charge/discharge windows considering time and price overrides.

        This shows what the battery will ACTUALLY do when overrides are applied.
        For example:
        - Time override: if 8:00-10:00 is calculated as charge, but 9:00-10:00 has a
          discharge override, the actual charge window will only be 8:00-9:00.
        - Price override: if price drops below threshold, those periods become charge windows
          even if not in calculated windows.

        Args:
            prices: List of processed price data
            charge_windows: Calculated charge windows
            discharge_windows: Calculated discharge windows
            config: Configuration dictionary
            is_tomorrow: Whether calculating for tomorrow (affects config key suffix)

        Returns:
            Tuple of (actual_charge_windows, actual_discharge_windows)
        """
        # Use tomorrow's config if applicable
        suffix = "_tomorrow" if is_tomorrow and config.get("tomorrow_settings_enabled", False) else ""

        # Check if any override is enabled
        time_override_enabled = config.get(f"time_override_enabled{suffix}", False)
        price_override_enabled = config.get(f"price_override_enabled{suffix}", False)

        if not time_override_enabled and not price_override_enabled:
            # No overrides, return calculated windows as-is
            return list(charge_windows), list(discharge_windows)

        # Get override configuration (using suffix for tomorrow settings)
        # Get time values and ensure they're in string format
        override_start = config.get(f"time_override_start{suffix}", "")
        override_end = config.get(f"time_override_end{suffix}", "")
        override_mode = config.get(f"time_override_mode{suffix}", MODE_NORMAL)

        # Convert to string format if needed
        if hasattr(override_start, 'strftime'):
            override_start_str = override_start.strftime("%H:%M:%S")
        elif override_start:
            override_start_str = str(override_start)
        else:
            override_start_str = ""

        if hasattr(override_end, 'strftime'):
            override_end_str = override_end.strftime("%H:%M:%S")
        elif override_end:
            override_end_str = str(override_end)
        else:
            override_end_str = ""

        price_override_threshold = config.get(f"price_override_threshold{suffix}", 0.15)

        # Validate time override config if enabled
        if time_override_enabled and (not override_start_str or not override_end_str):
            # Invalid time override config, disable it
            time_override_enabled = False

        # Build a complete timeline of all price windows with their states
        # considering calculated windows, time overrides, and price overrides
        timeline = []

        for price_data in prices:
            timestamp = price_data["timestamp"]
            duration = price_data["duration"]
            price = price_data["price"]

            # Determine state for this time period (priority order: time override > price override > calculated)
            state = STATE_NORMAL  # Default

            # Check time override first (highest priority)
            if time_override_enabled and self._is_in_time_range(timestamp, override_start_str, override_end_str):
                state = self._mode_to_state(override_mode)
            # Check price override
            elif price_override_enabled and price <= price_override_threshold:
                state = STATE_CHARGE
            else:
                # Check calculated windows
                for window in discharge_windows:
                    if self._is_window_active(window, timestamp):
                        state = STATE_DISCHARGE
                        break

                if state == STATE_NORMAL:
                    for window in charge_windows:
                        if self._is_window_active(window, timestamp):
                            state = STATE_CHARGE
                            break

            timeline.append({
                "timestamp": timestamp,
                "price": price_data["price"],
                "duration": duration,
                "state": state
            })

        # Extract actual charge and discharge windows from timeline
        new_actual_charge = [w for w in timeline if w["state"] == STATE_CHARGE]
        new_actual_discharge = [w for w in timeline if w["state"] == STATE_DISCHARGE]

        return new_actual_charge, new_actual_discharge

    def _get_buffer_energy(self, config: Dict[str, Any], is_tomorrow: bool, hass: Any = None) -> float:
        """Get battery starting state for the calculation period.

        Priority for TODAY (when use_battery_buffer_sensor is ON):
        1. Midnight battery state (fetched from recorder history)
        2. Current battery sensor value (if midnight state unavailable)
        3. Fall back to manual buffer_kwh value

        Priority for TOMORROW:
        1. Projected buffer from today's calculation
        2. Manual buffer_kwh_tomorrow value

        Priority (when use_battery_buffer_sensor is OFF):
        1. Manual buffer_kwh value (per-day: today or tomorrow)

        Args:
            config: Configuration dictionary
            is_tomorrow: Whether calculating for tomorrow
            hass: Home Assistant instance (for sensor value lookup)

        Returns:
            Buffer energy in kWh
        """
        # For tomorrow: check if projected buffer from today's calculation is available
        if is_tomorrow:
            projected = config.get("_projected_buffer_tomorrow")
            if projected is not None:
                return float(projected)

        use_sensor = config.get("use_battery_buffer_sensor", False)
        sensor_entity = config.get("battery_available_energy_sensor", "")

        # Get per-day buffer value (respects tomorrow_settings_enabled)
        suffix = "_tomorrow" if is_tomorrow and config.get("tomorrow_settings_enabled", False) else ""
        buffer_kwh = config.get(f"battery_buffer_kwh{suffix}", 0.0)

        # For TODAY with sensor enabled, prefer midnight state for full-day simulation
        if use_sensor and not is_tomorrow:
            midnight_state = config.get("_midnight_battery_state")
            if midnight_state is not None:
                return float(midnight_state)

        # If using sensor and we have a valid sensor entity (fallback to current value)
        if use_sensor and sensor_entity and hass:
            try:
                sensor_state = hass.states.get(sensor_entity)
                if sensor_state and sensor_state.state not in ("unknown", "unavailable", None):
                    return float(sensor_state.state)
            except (ValueError, TypeError):
                pass  # Fall back to manual value

        return float(buffer_kwh)

    def _build_chronological_timeline(
        self,
        prices: List[Dict[str, Any]],
        charge_windows: List[Dict[str, Any]],
        discharge_windows: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build time-ordered list of all periods with their window types.

        Args:
            prices: List of all price periods
            charge_windows: Selected charge windows
            discharge_windows: Selected discharge windows
            config: Configuration dictionary

        Returns:
            List of timeline entries sorted by timestamp
        """
        # Get sell price config for sell price calculation
        sell_country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
        sell_param_a = config.get("sell_formula_param_a", DEFAULT_SELL_FORMULA_PARAM_A)
        sell_param_b = config.get("sell_formula_param_b", DEFAULT_SELL_FORMULA_PARAM_B)

        # Create lookup sets for O(1) window type detection
        charge_timestamps = {w["timestamp"] for w in charge_windows}
        discharge_timestamps = {w["timestamp"] for w in discharge_windows}

        timeline = []
        for price_data in sorted(prices, key=lambda x: x["timestamp"]):
            ts = price_data["timestamp"]

            if ts in charge_timestamps:
                window_type = "charge"
            elif ts in discharge_timestamps:
                window_type = "discharge"
            else:
                window_type = "normal"

            # Calculate sell price for this period
            raw_price = price_data.get("raw_price", price_data["price"])
            sell_price = self._calculate_sell_price(
                raw_price, price_data["price"], sell_country, sell_param_a, sell_param_b
            )

            timeline.append({
                "timestamp": ts,
                "duration": price_data["duration"],
                "price": price_data["price"],
                "sell_price": sell_price,
                "window_type": window_type
            })

        return timeline

    def _get_expected_solar(self, config: Dict[str, Any], is_tomorrow: bool, hass: Any = None, energy_statistics: Dict[str, Any] = None) -> float:
        """Get expected solar production for the calculation period.

        Args:
            config: Configuration dictionary
            is_tomorrow: Whether calculating for tomorrow
            hass: Home Assistant instance (for sensor value lookup)
            energy_statistics: HA Energy Dashboard statistics

        Returns:
            Expected solar production in kWh
        """
        energy_stats = energy_statistics or {}
        if config.get("use_ha_energy_dashboard", False):
            # Priority 1: Use solar forecast from HA Energy Dashboard (Forecast.Solar)
            if is_tomorrow:
                forecast = energy_stats.get("solar_forecast_tomorrow")
            else:
                forecast = energy_stats.get("solar_forecast_today")
            if forecast is not None:
                _LOGGER.debug(f"Solar forecast from HA Energy Dashboard ({'tomorrow' if is_tomorrow else 'today'}): {forecast} kWh")
                return float(forecast)

            # Priority 2: Use actual solar production for today (not available for tomorrow)
            if not is_tomorrow:
                actual_solar = energy_stats.get("total_solar_production_kwh", 0.0)
                if actual_solar and actual_solar > 0:
                    _LOGGER.debug(f"No solar forecast, using actual production from HA Energy: {actual_solar} kWh")
                    return float(actual_solar)

            # No HA Energy data available - return 0 (no manual fallback when HA Energy is enabled)
            return 0.0

        # Check if solar forecast is enabled (only for non-HA-Energy modes)
        if not config.get("use_solar_forecast", True):
            return 0.0

        # Try sensor if enabled and hass is available
        if config.get("use_solar_forecast_sensor", False) and hass:
            sensor_key = "solar_forecast_sensor_tomorrow" if is_tomorrow else "solar_forecast_sensor"
            sensor_entity = config.get(sensor_key, "not_configured")

            if sensor_entity and sensor_entity != "not_configured":
                try:
                    sensor_state = hass.states.get(sensor_entity)
                    if sensor_state and sensor_state.state not in ("unknown", "unavailable", None):
                        solar_value = float(sensor_state.state)
                        _LOGGER.debug(f"Solar forecast from sensor {sensor_entity}: {solar_value} kWh")
                        return solar_value
                except (ValueError, TypeError) as e:
                    _LOGGER.warning(f"Could not read solar forecast sensor {sensor_entity}: {e}")
                    # Fall through to manual values

        # Fall back to manual values
        if is_tomorrow:
            # Tomorrow uses tomorrow-specific setting if enabled
            suffix = "_tomorrow" if config.get("tomorrow_settings_enabled", False) else ""
            return float(config.get(f"expected_solar_kwh{suffix}", 0.0))

        return float(config.get("expected_solar_kwh", 0.0))

    def _get_solar_for_period(
        self,
        timestamp: datetime,
        duration_minutes: int,
        config: Dict[str, Any],
        expected_solar_kwh: float,
        energy_statistics: Dict[str, Any] = None,
        is_tomorrow: bool = False
    ) -> float:
        """Get solar production for a specific time period.

        Uses actual hourly data for completed hours when HA Energy is enabled.
        Distributes remaining forecast across future solar window hours.

        Args:
            timestamp: Start of the period
            duration_minutes: Period duration in minutes
            config: Configuration dict with solar window settings
            expected_solar_kwh: Total expected solar for the day
            energy_statistics: HA Energy Dashboard statistics
            is_tomorrow: Whether calculating for tomorrow

        Returns:
            Solar power in kW for this period (0 if outside window)
        """
        if expected_solar_kwh <= 0:
            return 0.0

        energy_stats = energy_statistics or {}
        use_ha_energy = config.get("use_ha_energy_dashboard", False) and energy_stats.get("stats_available", False)

        if use_ha_energy and not is_tomorrow:
            solar_hourly = energy_stats.get("solar_hourly", {})
            current_hour = dt_util.now().hour
            local_timestamp = dt_util.as_local(timestamp) if timestamp.tzinfo else timestamp
            period_hour = local_timestamp.hour

            if solar_hourly and period_hour <= current_hour:
                _LOGGER.debug(
                    f"Solar for period {period_hour}:00 (local): actual={solar_hourly.get(period_hour, 'N/A')}, "
                    f"available_hours={list(solar_hourly.keys())}, current_hour={current_hour}"
                )

            if period_hour <= current_hour and period_hour in solar_hourly:
                # ACTUAL solar for completed hour - return as kW (hourly data is in kWh)
                return solar_hourly[period_hour]
            elif period_hour > current_hour:
                # FORECAST for future hour - use hourly forecast if available
                forecast_key = "solar_forecast_hourly_tomorrow" if is_tomorrow else "solar_forecast_hourly_today"
                hourly_forecast = energy_stats.get(forecast_key, {})

                # Forecast keys may be strings, try both int and str
                if period_hour in hourly_forecast:
                    return hourly_forecast[period_hour]
                elif str(period_hour) in hourly_forecast:
                    return hourly_forecast[str(period_hour)]

                # Fallback: distribute remaining forecast across remaining window
                actual_solar_so_far = sum(solar_hourly.values())
                remaining_forecast = max(0, expected_solar_kwh - actual_solar_so_far)

                if remaining_forecast <= 0:
                    return 0.0

                # Use remaining_forecast instead of expected_solar_kwh for future hours
                expected_solar_kwh = remaining_forecast

        # For tomorrow or when no stats available, try hourly forecast directly
        if config.get("use_ha_energy_dashboard", False):
            forecast_key = "solar_forecast_hourly_tomorrow" if is_tomorrow else "solar_forecast_hourly_today"
            hourly_forecast = energy_stats.get(forecast_key, {})
            local_ts = dt_util.as_local(timestamp) if timestamp.tzinfo else timestamp
            period_hour = local_ts.hour

            # Forecast keys may be strings, try both int and str
            if period_hour in hourly_forecast:
                return hourly_forecast[period_hour]
            elif str(period_hour) in hourly_forecast:
                return hourly_forecast[str(period_hour)]

        # Parse solar window times
        try:
            window_start_str = config.get("solar_window_start", "09:00:00")
            window_end_str = config.get("solar_window_end", "19:00:00")

            # Parse time strings
            if isinstance(window_start_str, str):
                start_parts = window_start_str.split(":")
                window_start = time(int(start_parts[0]), int(start_parts[1]))
            else:
                window_start = window_start_str

            if isinstance(window_end_str, str):
                end_parts = window_end_str.split(":")
                window_end = time(int(end_parts[0]), int(end_parts[1]))
            else:
                window_end = window_end_str

        except (ValueError, IndexError, AttributeError):
            # Default window on parse error
            window_start = time(9, 0)
            window_end = time(19, 0)

        # Check if period is within solar window
        period_time = timestamp.time()

        # Normal case: window within same day (e.g., 09:00-19:00)
        if window_start <= window_end:
            in_window = window_start <= period_time < window_end
        else:
            # Overnight window (unlikely for solar, but handle it)
            in_window = period_time >= window_start or period_time < window_end

        if not in_window:
            return 0.0

        # Calculate solar window duration in hours
        start_minutes = window_start.hour * 60 + window_start.minute
        end_minutes = window_end.hour * 60 + window_end.minute

        # Handle zero-duration window (start == end)
        if window_start == window_end:
            return 0.0

        if end_minutes > start_minutes:
            window_hours = (end_minutes - start_minutes) / 60
        else:
            # Overnight (24h - start + end)
            window_hours = (24 * 60 - start_minutes + end_minutes) / 60

        if window_hours <= 0:
            return 0.0

        # Distribute solar evenly across window
        # Return average power in kW
        average_solar_kw = expected_solar_kwh / window_hours

        return average_solar_kw

    def _calculate_actual_battery_flows(
        self,
        energy_statistics: Dict[str, Any],
        prices: List[Dict[str, Any]],
        config: Dict[str, Any],
        current_hour: int
    ) -> Dict[str, float]:
        """Calculate actual battery flows from HA Energy hourly data.

        Determines grid vs solar charging and base vs grid discharging
        by analyzing concurrent energy flows for each hour.
        This captures ALL actual charging including manual charging outside windows.

        Args:
            energy_statistics: Dict with battery_charge_hourly, grid_import_hourly, etc.
            prices: List of price dicts with timestamp, price, raw_price
            config: Config dict for sell price calculation
            current_hour: Current hour (0-23) to limit to completed hours

        Returns:
            Dict with charged_from_grid_kwh, charged_from_solar_kwh, charged_from_grid_cost,
            discharged_to_base_kwh, discharged_to_grid_kwh, discharged_revenue
        """
        result = {
            "charged_from_grid_kwh": 0.0,
            "charged_from_grid_cost": 0.0,
            "charged_from_solar_kwh": 0.0,
            "discharged_to_base_kwh": 0.0,
            "discharged_to_grid_kwh": 0.0,
            "discharged_revenue": 0.0,
        }

        if not energy_statistics or not energy_statistics.get("stats_available"):
            return result

        battery_charge = energy_statistics.get("battery_charge_hourly", {})
        battery_discharge = energy_statistics.get("battery_discharge_hourly", {})
        grid_import = energy_statistics.get("grid_import_hourly", {})
        grid_export = energy_statistics.get("grid_export_hourly", {})
        solar = energy_statistics.get("solar_hourly", {})
        real_consumption = energy_statistics.get("real_consumption_hourly", {})

        # Build price lookup by hour
        price_by_hour = {}
        for p in prices:
            ts = p.get("timestamp")
            if ts and hasattr(ts, "hour"):
                price_by_hour[ts.hour] = p

        # Get sell price config
        sell_country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
        sell_param_a = config.get("sell_formula_param_a", DEFAULT_SELL_FORMULA_PARAM_A)
        sell_param_b = config.get("sell_formula_param_b", DEFAULT_SELL_FORMULA_PARAM_B)

        # Analyze each completed hour
        for hour in range(current_hour + 1):
            # Handle both int and str keys
            charge_kwh = battery_charge.get(hour, battery_charge.get(str(hour), 0.0))
            discharge_kwh = battery_discharge.get(hour, battery_discharge.get(str(hour), 0.0))
            import_kwh = grid_import.get(hour, grid_import.get(str(hour), 0.0))
            export_kwh = grid_export.get(hour, grid_export.get(str(hour), 0.0))
            solar_kwh = solar.get(hour, solar.get(str(hour), 0.0))

            # Get prices for this hour
            price_data = price_by_hour.get(hour, {})
            buy_price = price_data.get("price", 0.25)
            raw_price = price_data.get("raw_price", buy_price)
            sell_price = self._calculate_sell_price(raw_price, buy_price, sell_country, sell_param_a, sell_param_b)

            # Charging: determine grid vs solar source
            # Logic: Grid import first covers base load consumption, only remainder goes to battery
            # This correctly accounts for concurrent household consumption during charging hours
            if charge_kwh > 0:
                # Get base load consumption for this hour
                consumption_kwh = real_consumption.get(hour, real_consumption.get(str(hour), 0.0))

                # Grid power available for battery = grid import - base load consumption
                # (base load must be covered first before battery can charge from grid)
                grid_available_for_battery = max(0, import_kwh - consumption_kwh)

                # Grid contribution to battery is the lesser of available and actual charge
                grid_charge = min(grid_available_for_battery, charge_kwh)
                solar_charge = max(0, charge_kwh - grid_charge)

                result["charged_from_grid_kwh"] += grid_charge
                result["charged_from_grid_cost"] += grid_charge * buy_price
                result["charged_from_solar_kwh"] += solar_charge

            # Discharging: determine base offset vs grid export
            # Logic: If there's grid export during discharge, that amount went to grid
            # The rest offset base consumption
            if discharge_kwh > 0:
                # Grid export portion is the lesser of export and discharge
                grid_discharge = min(export_kwh, discharge_kwh)
                base_discharge = max(0, discharge_kwh - grid_discharge)

                result["discharged_to_grid_kwh"] += grid_discharge
                result["discharged_revenue"] += grid_discharge * sell_price
                result["discharged_to_base_kwh"] += base_discharge

        return result

    def _detect_manual_charging(
        self,
        energy_statistics: Dict[str, Any],
        actual_charge_windows: List[Dict[str, Any]],
        all_prices: List[Dict[str, Any]],
        current_hour: int
    ) -> Dict[str, Any]:
        """Detect battery charging that occurred outside CEW planned windows.

        Compares HA Energy battery_charge_hourly data against actual_charge_times
        to identify manual or external charging events.

        Args:
            energy_statistics: HA Energy Dashboard statistics
            actual_charge_windows: List of planned charge windows
            all_prices: Price data for cost calculation
            current_hour: Current hour (0-23)

        Returns:
            Dict with:
            - manual_charge_hours: List of hours where unplanned charging occurred
            - manual_charge_kwh: Total kWh from manual charging
            - manual_charge_cost: Estimated cost of manual charging
            - manual_charge_detected: Boolean flag
        """
        result = {
            "manual_charge_hours": [],
            "manual_charge_kwh": 0.0,
            "manual_charge_cost": 0.0,
            "manual_charge_detected": False,
        }

        if not energy_statistics or not energy_statistics.get("stats_available"):
            return result

        battery_charge_hourly = energy_statistics.get("battery_charge_hourly", {})
        if not battery_charge_hourly:
            return result

        # Build set of planned charge hours from actual_charge_windows
        planned_charge_hours = set()
        for w in actual_charge_windows:
            ts = w.get("timestamp")
            if ts and hasattr(ts, "hour"):
                planned_charge_hours.add(ts.hour)

        # Build hour-to-price lookup
        hour_prices = {}
        for price_data in all_prices:
            hour_int = price_data["timestamp"].hour
            if hour_int not in hour_prices:
                hour_prices[hour_int] = price_data["price"]

        # Compare actual charging against planned windows
        for hour, charge_kwh in battery_charge_hourly.items():
            hour_int = int(hour) if isinstance(hour, str) else hour
            if hour_int >= current_hour:
                continue  # Only check completed hours

            # Threshold to ignore noise (0.1 kWh minimum)
            if charge_kwh > 0.1 and hour_int not in planned_charge_hours:
                result["manual_charge_hours"].append(hour_int)
                result["manual_charge_kwh"] += charge_kwh
                # Calculate cost using hour's price
                if hour_int in hour_prices:
                    result["manual_charge_cost"] += charge_kwh * hour_prices[hour_int]
                result["manual_charge_detected"] = True

        result["manual_charge_kwh"] = round(result["manual_charge_kwh"], 3)
        result["manual_charge_cost"] = round(result["manual_charge_cost"], 4)

        if result["manual_charge_detected"]:
            _LOGGER.info(
                f"Manual charging detected: {result['manual_charge_kwh']} kWh "
                f"at hours {result['manual_charge_hours']} "
                f"(cost: €{result['manual_charge_cost']:.4f})"
            )

        return result

    def _simulate_chronological_costs(
        self,
        timeline: List[Dict[str, Any]],
        config: Dict[str, Any],
        buffer_energy: float,
        limit_discharge: bool = False,
        is_tomorrow: bool = False,
        hass: Any = None,
        energy_statistics: Dict[str, Any] = None,
        actual_battery_flows: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Simulate battery state and calculate costs chronologically.

        Args:
            timeline: Time-ordered list of periods with window types
            config: Configuration dictionary
            buffer_energy: Starting battery energy in kWh
            limit_discharge: If True, skip discharge windows that can't be supported
            hass: Home Assistant instance (for sensor value lookup)
            actual_battery_flows: Actual battery charging data from HA Energy (for RTE)

        Returns:
            Dictionary with calculated costs, battery trajectory, and feasibility info
        """
        # Get configuration
        battery_capacity = config.get("battery_capacity", 100.0)  # kWh
        charge_power = config.get("charge_power", 0) / 1000  # W to kW
        discharge_power = config.get("discharge_power", 0) / 1000
        manual_base_usage = config.get("base_usage", 0) / 1000
        battery_rte = config.get("battery_rte", 85) / 100

        # HA Energy Dashboard integration
        energy_stats = energy_statistics or {}
        use_ha_energy = config.get("use_ha_energy_dashboard", False) and energy_stats.get("stats_available", False)
        # Use weighted 72h average for base_usage when HA Energy enabled
        weighted_avg = energy_stats.get("weighted_avg_consumption", 0.0) if use_ha_energy else 0.0
        base_usage = weighted_avg if (use_ha_energy and weighted_avg > 0) else manual_base_usage
        # Use real consumption (formula result) instead of just grid import
        real_consumption_hourly = energy_stats.get("real_consumption_hourly", {}) if use_ha_energy else {}
        avg_real_consumption = energy_stats.get("avg_real_consumption", 0.0) if use_ha_energy else 0.0
        current_hour = energy_stats.get("current_hour", 23)  # For past/future determination
        # Legacy compatibility aliases
        consumption_hourly = real_consumption_hourly
        avg_hourly_consumption = avg_real_consumption

        # Strategies
        charge_strategy = config.get("base_usage_charge_strategy", "grid_covers_both")
        discharge_strategy = config.get("base_usage_discharge_strategy", "subtract_base")
        normal_strategy = config.get("base_usage_normal_strategy", "grid_covers")

        # RTE-aware discharge settings (global)
        rte_aware_discharge = config.get("rte_aware_discharge", True)
        rte_discharge_margin = config.get("rte_discharge_margin", 2) / 100  # Convert from % to decimal

        # Initialize state
        battery_state = buffer_energy
        planned_charge_cost = 0.0
        planned_discharge_revenue = 0.0
        planned_base_usage_cost = 0.0
        actual_charge_kwh = 0.0
        actual_discharge_kwh = 0.0
        uncovered_base_kwh = 0.0
        battery_trajectory = []
        feasibility_issues = []
        skipped_discharge_windows = []
        feasible_discharge_windows = []
        feasible_charge_windows = []
        skipped_charge_windows = []

        # Grid usage tracking
        grid_kwh_total = 0.0  # Cumulative grid kWh (charge + uncovered base)

        # Solar configuration and tracking
        expected_solar_kwh = self._get_expected_solar(config, is_tomorrow, hass, energy_statistics)
        solar_priority = config.get("solar_priority_strategy", "base_then_grid")
        solar_to_battery_kwh = 0.0
        solar_offset_base_kwh = 0.0
        solar_exported_kwh = 0.0
        solar_export_revenue = 0.0
        grid_savings_from_solar = 0.0
        solar_export_events = []  # Track per-period exports for completed calculation
        solar_base_offset_events = []  # Track per-period solar base offsets for completed calculation

        # Detailed battery tracking for energy flow report
        battery_charged_from_grid_kwh = 0.0
        battery_charged_from_grid_cost = 0.0
        battery_discharged_to_base_kwh = 0.0
        battery_discharged_to_grid_kwh = 0.0
        grid_charging_prices = []  # To calculate avg charge price
        grid_discharging_prices = []  # To calculate avg discharge price

        # RTE (Round-Trip Efficiency) loss tracking
        rte_loss_kwh = 0.0  # Total kWh lost to battery conversion inefficiency
        rte_loss_value = 0.0  # Opportunity cost - what those kWh would have earned if exported

        # RTE-aware discharge tracking
        rte_preserved_kwh = 0.0  # kWh preserved by RTE-aware logic
        rte_preserved_periods = []  # List of periods where battery was preserved
        current_breakeven_price = 0.0  # Most recent calculated breakeven price
        rte_breakeven_source = "simulation"  # "actual_ha" or "simulation"
        rte_solar_opportunity_price = 0.0  # Top percentile sell price for solar-only RTE

        # Energy consumption diagnostic tracking
        energy_actual_kwh = 0.0  # Sum of ACTUAL consumption used (past hours)
        energy_estimated_kwh = 0.0  # Sum of ESTIMATED consumption (future hours)
        energy_hours_with_actual = 0  # Count of hours using actual data
        energy_hours_with_estimate = 0  # Count of hours using estimates

        # Sell formula parameters for solar export revenue
        sell_country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
        sell_param_a = config.get("sell_formula_param_a", DEFAULT_SELL_FORMULA_PARAM_A)
        sell_param_b = config.get("sell_formula_param_b", DEFAULT_SELL_FORMULA_PARAM_B)

        # Extract discharge window sell prices for RTE-aware normal mode logic
        # Used to calculate opportunity cost when no grid charging occurred
        actual_discharge_sell_prices = [
            p["sell_price"] for p in timeline if p.get("window_type") == "discharge"
        ]

        # Log first few and last few windows to verify order
        if timeline and limit_discharge:
            first_windows = timeline[:5]
            last_windows = timeline[-3:] if len(timeline) > 5 else []
            _LOGGER.info(f"Timeline order (first 5): {[(w['timestamp'], w['window_type']) for w in first_windows]}")
            if last_windows:
                _LOGGER.info(f"Timeline order (last 3): {[(w['timestamp'], w['window_type']) for w in last_windows]}")

        for period in timeline:
            duration_hours = period["duration"] / 60
            price = period["price"]
            sell_price = period["sell_price"]
            window_type = period["window_type"]
            battery_before = battery_state

            # Calculate effective base usage for this period (HA Energy Dashboard integration)
            # When HA Energy ON: use ONLY HA Energy data (NO fallback to manual)
            # When HA Energy OFF: use manual base_usage only
            effective_base_usage_kw = base_usage  # Default to configured manual value
            used_actual_data = False
            if use_ha_energy:
                # HA Energy enabled - use ONLY HA Energy data (no manual fallback)
                period_ts = period.get("timestamp")
                if period_ts:
                    period_hour = period_ts.hour if hasattr(period_ts, 'hour') else 0
                    if is_tomorrow:
                        # Tomorrow is all future - use HA Energy average
                        effective_base_usage_kw = avg_hourly_consumption
                        energy_estimated_kwh += effective_base_usage_kw * duration_hours
                        energy_hours_with_estimate += 1
                    elif period_hour <= current_hour:
                        # Past or current hour - use actual if available
                        # Note: current_hour data may be partial but still more accurate than estimate
                        if period_hour in consumption_hourly:
                            effective_base_usage_kw = consumption_hourly[period_hour]  # kWh/hour = kW
                            energy_actual_kwh += effective_base_usage_kw * duration_hours
                            energy_hours_with_actual += 1
                            used_actual_data = True
                        else:
                            # Past/current hour but no data - use HA Energy average (no manual fallback)
                            effective_base_usage_kw = avg_hourly_consumption
                            energy_estimated_kwh += effective_base_usage_kw * duration_hours
                            energy_hours_with_estimate += 1
                    else:
                        # Future hour - use HA Energy average (no manual fallback)
                        effective_base_usage_kw = avg_hourly_consumption
                        energy_estimated_kwh += effective_base_usage_kw * duration_hours
                        energy_hours_with_estimate += 1
            # else: HA Energy disabled - use manual base_usage (already set as default)

            if window_type == "charge":
                # Calculate how much we CAN charge (limited by capacity)
                desired_charge = charge_power * duration_hours
                base_demand = effective_base_usage_kw * duration_hours
                available_capacity = max(0, battery_capacity - battery_state)  # Usable kWh space left

                # Get solar for this period
                solar_kw = self._get_solar_for_period(
                    period["timestamp"],
                    period["duration"],
                    config,
                    expected_solar_kwh,
                    energy_statistics,
                    is_tomorrow
                )
                solar_available = solar_kw * duration_hours

                # Solar covers base first
                solar_to_base = min(solar_available, base_demand)
                solar_remaining = solar_available - solar_to_base

                if solar_to_base > 0:
                    solar_offset_base_kwh += solar_to_base
                    grid_savings_from_solar += solar_to_base * price
                    # Track for completed calculation
                    solar_base_offset_events.append({
                        "timestamp": period["timestamp"],
                        "duration": period["duration"],
                        "kwh": solar_to_base,
                        "savings": solar_to_base * price
                    })

                # Solar can also contribute to charging
                solar_to_charge = min(solar_remaining, desired_charge, available_capacity)
                if solar_to_charge > 0:
                    # Track RTE loss - energy lost to battery inefficiency
                    rte_loss = solar_to_charge * (1 - battery_rte)
                    rte_loss_kwh += rte_loss
                    # Calculate sell price for opportunity cost
                    raw_price = period.get("raw_price", price)
                    current_sell_price = self._calculate_sell_price(raw_price, price, sell_country, sell_param_a, sell_param_b)
                    rte_loss_value += rte_loss * current_sell_price  # What we'd have earned exporting

                    battery_state += solar_to_charge * battery_rte
                    solar_to_battery_kwh += solar_to_charge
                    grid_savings_from_solar += solar_to_charge * price
                    solar_remaining -= solar_to_charge

                # Export any remaining solar
                if solar_remaining > 0:
                    solar_exported_kwh += solar_remaining
                    # Calculate sell price and add export revenue
                    raw_price = period.get("raw_price", price)
                    sell_price = self._calculate_sell_price(raw_price, price, sell_country, sell_param_a, sell_param_b)
                    revenue = solar_remaining * sell_price
                    solar_export_revenue += revenue
                    # Track for completed calculation
                    solar_export_events.append({
                        "timestamp": period["timestamp"],
                        "duration": period["duration"],
                        "revenue": revenue,
                        "kwh": solar_remaining
                    })

                # Grid provides remaining charge (after solar contribution)
                grid_charge_needed = max(0, desired_charge - solar_to_charge)
                # RTE does NOT affect grid charging calculation - only affects buffer size display
                max_grid_draw = max(0, available_capacity - solar_to_charge)  # Remaining capacity after solar
                actual_grid_charge = min(grid_charge_needed, max_grid_draw)  # Grid draw in kWh

                # Check if charge window is feasible (can charge at least 10% of desired from grid+solar)
                total_charge = solar_to_charge + actual_grid_charge
                if total_charge >= desired_charge * 0.1:
                    feasible_charge_windows.append(period)

                    # RTE DOES affect battery state - physical reality of charging losses
                    usable_grid_charge = actual_grid_charge * battery_rte  # Battery stores less than grid draw
                    # Track RTE loss from grid charging - both kWh and cost
                    grid_rte_loss = actual_grid_charge * (1 - battery_rte)
                    rte_loss_kwh += grid_rte_loss
                    rte_loss_value += grid_rte_loss * price  # Cost of energy lost to RTE

                    # Remaining base demand after solar
                    base_from_other = base_demand - solar_to_base

                    if charge_strategy == "grid_covers_both":
                        # Grid provides charge power AND remaining base usage
                        battery_state += usable_grid_charge
                        planned_charge_cost += price * (actual_grid_charge + base_from_other)
                        grid_kwh_total += actual_grid_charge + base_from_other
                    else:  # battery_covers_base
                        # Grid provides charge power, battery covers remaining base usage
                        net_battery_change = usable_grid_charge - base_from_other
                        battery_state += net_battery_change
                        planned_charge_cost += price * actual_grid_charge
                        grid_kwh_total += actual_grid_charge
                        # If battery can't cover base (goes negative), grid must cover the shortfall
                        if battery_state < 0:
                            shortfall = -battery_state
                            planned_base_usage_cost += price * shortfall
                            grid_kwh_total += shortfall
                            uncovered_base_kwh += shortfall
                            battery_state = 0

                    battery_state = min(max(0, battery_state), battery_capacity)  # Clamp to valid range
                    actual_charge_kwh += actual_grid_charge + solar_to_charge

                    # Track grid charging separately for energy flow report
                    if actual_grid_charge > 0:
                        battery_charged_from_grid_kwh += actual_grid_charge
                        battery_charged_from_grid_cost += actual_grid_charge * price
                        grid_charging_prices.append(price)
                else:
                    # Battery too full - skip this charge window
                    skipped_charge_windows.append({
                        "timestamp": period["timestamp"].isoformat() if hasattr(period["timestamp"], 'isoformat') else str(period["timestamp"]),
                        "price": price,
                        "reason": f"Battery nearly full ({battery_state:.2f}/{battery_capacity:.2f} kWh, only {available_capacity:.2f} kWh space)"
                    })

            elif window_type == "discharge":
                # Use discharge strategy for all discharge windows
                strategy = discharge_strategy

                # Calculate discharge parameters based on strategy
                base_demand = effective_base_usage_kw * duration_hours

                # Get solar for this period
                solar_kw = self._get_solar_for_period(
                    period["timestamp"],
                    period["duration"],
                    config,
                    expected_solar_kwh,
                    energy_statistics,
                    is_tomorrow
                )
                solar_available = solar_kw * duration_hours

                # Solar covers base during discharge (more battery available for export)
                solar_to_base = min(solar_available, base_demand)
                if solar_to_base > 0:
                    solar_offset_base_kwh += solar_to_base
                    # During discharge, solar covering base means less battery needed for house
                    # This is effectively a grid saving (battery would otherwise need to cover this)
                    grid_savings_from_solar += solar_to_base * price
                    # Track for completed calculation
                    solar_base_offset_events.append({
                        "timestamp": period["timestamp"],
                        "duration": period["duration"],
                        "kwh": solar_to_base,
                        "savings": solar_to_base * price
                    })

                # Export remaining solar (adds to revenue)
                solar_remaining = solar_available - solar_to_base
                if solar_remaining > 0:
                    solar_exported_kwh += solar_remaining
                    # Calculate sell price and add export revenue
                    raw_price = period.get("raw_price", price)
                    sell_price = self._calculate_sell_price(raw_price, price, sell_country, sell_param_a, sell_param_b)
                    revenue = solar_remaining * sell_price
                    solar_export_revenue += revenue
                    # Track for completed calculation
                    solar_export_events.append({
                        "timestamp": period["timestamp"],
                        "duration": period["duration"],
                        "revenue": revenue,
                        "kwh": solar_remaining
                    })

                # Effective base demand after solar covers some
                effective_base_demand = base_demand - solar_to_base

                if strategy == "already_included":
                    # User configured discharge_power as NET export to grid
                    # Battery must also cover house (minus what solar covers), so total drain = discharge + effective_base
                    desired_net_export = discharge_power * duration_hours
                    total_drain_needed = desired_net_export + effective_base_demand
                else:  # subtract_base
                    # User configured discharge_power as GROSS battery output
                    # Battery outputs at discharge_power, house takes effective_base first (after solar), grid gets remainder
                    desired_battery_output = discharge_power * duration_hours
                    total_drain_needed = desired_battery_output  # Battery max output
                    # Net export increases because solar covers part of base (less goes to house)
                    desired_net_export = max(0, desired_battery_output - effective_base_demand)

                if limit_discharge:
                    # Conservative mode: buffer is the STARTING energy, not a floor
                    # All current battery energy is available for discharge (down to 0)
                    available_for_discharge = max(0, battery_state)

                    if strategy == "already_included":
                        # Need full discharge + effective base (solar already covered part)
                        if available_for_discharge >= total_drain_needed:
                            actual_net_export = desired_net_export
                            actual_base_from_battery = effective_base_demand
                        else:
                            # Limited - prioritize base usage, rest to export
                            actual_base_from_battery = min(effective_base_demand, available_for_discharge)
                            actual_net_export = min(desired_net_export, available_for_discharge - actual_base_from_battery)
                    else:  # subtract_base
                        # Battery outputs up to discharge_power, split between house and grid
                        if available_for_discharge >= total_drain_needed:
                            actual_battery_output = total_drain_needed
                            actual_base_from_battery = effective_base_demand
                            actual_net_export = actual_battery_output - actual_base_from_battery
                        else:
                            # Limited - house gets priority
                            actual_battery_output = available_for_discharge
                            actual_base_from_battery = min(effective_base_demand, actual_battery_output)
                            actual_net_export = max(0, actual_battery_output - actual_base_from_battery)
                else:
                    # Optimistic mode: can discharge all the way to 0
                    if strategy == "already_included":
                        if battery_state >= total_drain_needed:
                            actual_net_export = desired_net_export
                            actual_base_from_battery = effective_base_demand
                        else:
                            # Limited - prioritize base usage
                            actual_base_from_battery = min(effective_base_demand, battery_state)
                            actual_net_export = min(desired_net_export, battery_state - actual_base_from_battery)
                    else:  # subtract_base
                        if battery_state >= total_drain_needed:
                            actual_battery_output = total_drain_needed
                            actual_base_from_battery = effective_base_demand
                            actual_net_export = actual_battery_output - actual_base_from_battery
                        else:
                            # Limited - house gets priority
                            actual_battery_output = battery_state
                            actual_base_from_battery = min(effective_base_demand, actual_battery_output)
                            actual_net_export = max(0, actual_battery_output - actual_base_from_battery)

                total_actual_drain = actual_net_export + actual_base_from_battery

                if limit_discharge:
                    # Conservative mode: only include if we can do FULL discharge
                    if battery_state >= total_drain_needed:
                        feasible_discharge_windows.append(period)
                        battery_state -= total_actual_drain
                        actual_discharge_kwh += actual_net_export

                        # Revenue based on net export to grid
                        planned_discharge_revenue += sell_price * actual_net_export

                        # Track discharge destinations for energy flow report
                        battery_discharged_to_base_kwh += actual_base_from_battery
                        battery_discharged_to_grid_kwh += actual_net_export
                        if actual_net_export > 0:
                            grid_discharging_prices.append(sell_price)
                    else:
                        skipped_discharge_windows.append({
                            "timestamp": period["timestamp"].isoformat() if hasattr(period["timestamp"], 'isoformat') else str(period["timestamp"]),
                            "price": price,
                            "sell_price": sell_price,
                            "reason": f"Insufficient battery ({battery_state:.2f} kWh < {total_drain_needed:.2f} kWh needed)"
                        })
                        # Even if discharge window is skipped, base usage still drains battery
                        battery_state -= actual_base_from_battery
                        # Track base coverage even when export is skipped
                        battery_discharged_to_base_kwh += actual_base_from_battery
                        if actual_base_from_battery < effective_base_demand:
                            # Grid fallback: battery can't cover all base usage (after solar)
                            uncovered = effective_base_demand - actual_base_from_battery
                            planned_base_usage_cost += price * uncovered
                            grid_kwh_total += uncovered
                            uncovered_base_kwh += uncovered
                else:
                    # Optimistic mode: include all, track what we can actually discharge
                    if actual_net_export < desired_net_export * 0.5:
                        feasibility_issues.append(
                            f"{period['timestamp']}: Discharge limited by battery state "
                            f"({actual_net_export:.2f}/{desired_net_export:.2f} kWh)"
                        )

                    battery_state -= total_actual_drain
                    actual_discharge_kwh += actual_net_export

                    # Revenue based on net export to grid
                    planned_discharge_revenue += sell_price * actual_net_export

                    # Track discharge destinations for energy flow report
                    battery_discharged_to_base_kwh += actual_base_from_battery
                    battery_discharged_to_grid_kwh += actual_net_export
                    if actual_net_export > 0:
                        grid_discharging_prices.append(sell_price)

                    # Include window in optimistic mode (limit_discharge=False)
                    # Without this, feasible_discharge_windows stays empty!
                    feasible_discharge_windows.append(period)

                battery_state = max(0, battery_state)  # Ensure non-negative

            else:  # normal
                base_demand = effective_base_usage_kw * duration_hours

                # Get solar for this period
                solar_kw = self._get_solar_for_period(
                    period["timestamp"],
                    period["duration"],
                    config,
                    expected_solar_kwh,
                    energy_statistics,
                    is_tomorrow
                )
                solar_available = solar_kw * duration_hours  # kWh available this period

                # Solar priority: ALWAYS covers base usage first
                solar_to_base = min(solar_available, base_demand)
                solar_remaining = solar_available - solar_to_base

                # Track solar contribution to base
                if solar_to_base > 0:
                    solar_offset_base_kwh += solar_to_base
                    grid_savings_from_solar += solar_to_base * price
                    # Track for completed calculation
                    solar_base_offset_events.append({
                        "timestamp": period["timestamp"],
                        "duration": period["duration"],
                        "kwh": solar_to_base,
                        "savings": solar_to_base * price
                    })

                # Handle excess solar based on strategy
                if solar_remaining > 0:
                    if solar_priority == "base_then_battery":
                        # Charge battery with excess solar (limited by charge rate)
                        available_capacity = max(0, battery_capacity - battery_state)
                        max_charge_this_period = charge_power * duration_hours
                        solar_to_batt = min(solar_remaining, available_capacity, max_charge_this_period)
                        if solar_to_batt > 0:
                            # Track RTE loss - energy lost to battery inefficiency
                            rte_loss = solar_to_batt * (1 - battery_rte)
                            rte_loss_kwh += rte_loss
                            # Calculate opportunity cost (what we'd have earned exporting)
                            raw_price = period.get("raw_price", price)
                            current_sell_price = self._calculate_sell_price(raw_price, price, sell_country, sell_param_a, sell_param_b)
                            rte_loss_value += rte_loss * current_sell_price

                            battery_state += solar_to_batt * battery_rte
                            solar_to_battery_kwh += solar_to_batt
                            solar_remaining -= solar_to_batt

                    # Remaining solar exported to grid
                    if solar_remaining > 0:
                        solar_exported_kwh += solar_remaining
                        # Calculate sell price and add export revenue
                        raw_price = period.get("raw_price", price)
                        sell_price = self._calculate_sell_price(raw_price, price, sell_country, sell_param_a, sell_param_b)
                        revenue = solar_remaining * sell_price
                        solar_export_revenue += revenue
                        # Track for completed calculation
                        solar_export_events.append({
                            "timestamp": period["timestamp"],
                            "duration": period["duration"],
                            "revenue": revenue,
                            "kwh": solar_remaining
                        })

                # Remaining base demand after solar
                base_from_other = base_demand - solar_to_base

                # Handle remaining base demand with existing strategy
                if base_from_other > 0:
                    if normal_strategy == "grid_covers":
                        # Grid provides remaining base usage
                        planned_base_usage_cost += price * base_from_other
                        grid_kwh_total += base_from_other
                    elif normal_strategy in ("battery_covers", "battery_covers_limited"):
                        # RTE-aware discharge decision
                        use_battery = True

                        # Apply min usable threshold - below this, battery is considered depleted
                        min_usable_kwh = config.get("battery_min_usable_kwh", 0.0)
                        effective_battery = max(0, battery_state - min_usable_kwh)
                        if rte_aware_discharge and effective_battery > 0:
                            # Calculate breakeven price based on what charged the SIMULATION battery
                            # IMPORTANT: battery_state is the SIMULATION state, so we must use
                            # SIMULATION charging data to determine RTE behavior, not HA Energy data.
                            # HA Energy data reflects reality, but the simulation may differ
                            # (e.g., no charge windows elected means simulation has no grid charging,
                            # even if manual grid charging occurred in reality).

                            # Use SIMULATION charging data - this matches what filled battery_state
                            sim_grid_charged = battery_charged_from_grid_kwh
                            sim_grid_cost = battery_charged_from_grid_cost
                            rte_breakeven_source = "simulation"

                            if sim_grid_charged > 0:
                                # Simulation had grid charging - use simulation cost for breakeven
                                avg_charge_price = sim_grid_cost / sim_grid_charged
                                current_breakeven_price = (avg_charge_price / battery_rte) * (1 + rte_discharge_margin)

                                # Only use battery if current price exceeds breakeven + margin
                                use_battery = price > current_breakeven_price
                            else:
                                # Simulation had NO grid charging - battery_state came from SOLAR
                                # Check if user wants RTE protection for solar-charged battery
                                rte_protect_solar = config.get("rte_protect_solar_charge", True)

                                if rte_protect_solar:
                                    # Use TOP percentile sell prices as opportunity baseline
                                    # This represents what we COULD sell the energy for
                                    all_sell_prices = [p["sell_price"] for p in timeline if p.get("sell_price", 0) > 0]

                                    if all_sell_prices:
                                        # Use top 20% of sell prices as opportunity baseline
                                        sorted_prices = sorted(all_sell_prices, reverse=True)
                                        top_count = max(1, len(sorted_prices) // 5)  # Top 20%
                                        top_sell_prices = sorted_prices[:top_count]
                                        avg_opportunity = sum(top_sell_prices) / len(top_sell_prices)

                                        # Breakeven = opportunity price minus margin
                                        current_breakeven_price = avg_opportunity * (1 - rte_discharge_margin)
                                        use_battery = price > current_breakeven_price
                                        rte_solar_opportunity_price = avg_opportunity
                                    else:
                                        # No price data - freely use battery
                                        use_battery = True
                                else:
                                    # User disabled solar RTE protection - freely use solar-charged battery
                                    use_battery = True

                        if use_battery:
                            # Battery provides remaining base usage (drain battery)
                            battery_drain = min(base_from_other, battery_state)
                            battery_state -= battery_drain
                            battery_discharged_to_base_kwh += battery_drain
                            uncovered = base_from_other - battery_drain
                        else:
                            # RTE-aware: preserve battery, use grid instead
                            battery_drain = 0
                            uncovered = base_from_other
                            rte_preserved_kwh += min(base_from_other, effective_battery)  # What we would have drained
                            rte_preserved_periods.append({
                                "timestamp": period["timestamp"].isoformat() if hasattr(period["timestamp"], 'isoformat') else str(period["timestamp"]),
                                "duration": period["duration"],
                                "price": round(price, 4),
                                "breakeven": round(current_breakeven_price, 4),
                                "preserved_kwh": round(min(base_from_other, effective_battery), 3)
                            })

                        if uncovered > 0:
                            # Grid fallback: battery empty/preserved, grid covers the rest
                            planned_base_usage_cost += price * uncovered
                            grid_kwh_total += uncovered
                            uncovered_base_kwh += uncovered

            battery_trajectory.append({
                "timestamp": period["timestamp"].isoformat() if hasattr(period["timestamp"], 'isoformat') else str(period["timestamp"]),
                "window_type": window_type,
                "battery_before": round(battery_before, 3),
                "battery_after": round(battery_state, 3),
                "price": price
            })

        planned_total_cost = round(
            planned_charge_cost + planned_base_usage_cost - planned_discharge_revenue - solar_export_revenue, 3
        )

        _LOGGER.info(
            f"Simulation complete: charged={actual_charge_kwh:.2f} kWh, "
            f"discharged={actual_discharge_kwh:.2f} kWh, final_battery={battery_state:.2f} kWh, "
            f"feasible_charge={len(feasible_charge_windows)}, skipped_charge={len(skipped_charge_windows)}, "
            f"feasible_discharge={len(feasible_discharge_windows)}, "
            f"skipped_discharge={len(skipped_discharge_windows)}"
        )

        return {
            "planned_total_cost": planned_total_cost,
            "planned_charge_cost": round(planned_charge_cost, 3),
            "planned_discharge_revenue": round(planned_discharge_revenue, 3),
            "planned_base_usage_cost": round(planned_base_usage_cost, 3),
            "battery_trajectory": battery_trajectory,
            "feasibility_issues": feasibility_issues,
            "actual_charge_kwh": round(actual_charge_kwh, 3),
            "actual_discharge_kwh": round(actual_discharge_kwh, 3),
            "uncovered_base_kwh": round(uncovered_base_kwh, 3),
            "final_battery_state": round(battery_state, 3),
            "grid_kwh_total": round(grid_kwh_total, 3),
            "skipped_discharge_windows": skipped_discharge_windows,
            "feasible_discharge_windows": feasible_discharge_windows,
            "feasible_charge_windows": feasible_charge_windows,
            "skipped_charge_windows": skipped_charge_windows,
            # Solar integration metrics
            "solar_to_battery_kwh": round(solar_to_battery_kwh, 3),
            "solar_offset_base_kwh": round(solar_offset_base_kwh, 3),
            "solar_exported_kwh": round(solar_exported_kwh, 3),
            "solar_export_revenue": round(solar_export_revenue, 3),
            "solar_total_contribution_kwh": round(solar_to_battery_kwh + solar_offset_base_kwh, 3),
            "grid_savings_from_solar": round(grid_savings_from_solar, 3),
            "expected_solar_kwh": round(expected_solar_kwh, 3),
            "solar_export_events": solar_export_events,
            "solar_base_offset_events": solar_base_offset_events,
            # Detailed battery tracking for energy flow report
            "battery_charged_from_grid_kwh": round(battery_charged_from_grid_kwh, 3),
            "battery_charged_from_grid_cost": round(battery_charged_from_grid_cost, 4),
            "battery_charged_from_solar_kwh": round(solar_to_battery_kwh, 3),  # Alias for clarity
            "battery_charged_avg_price": round(
                sum(grid_charging_prices) / len(grid_charging_prices), 5
            ) if grid_charging_prices else 0.0,
            "battery_discharged_to_base_kwh": round(battery_discharged_to_base_kwh, 3),
            "battery_discharged_to_grid_kwh": round(battery_discharged_to_grid_kwh, 3),
            "battery_discharged_avg_price": round(
                sum(grid_discharging_prices) / len(grid_discharging_prices), 5
            ) if grid_discharging_prices else 0.0,
            # RTE (Round-Trip Efficiency) loss tracking
            "rte_loss_kwh": round(rte_loss_kwh, 3),
            "rte_loss_value": round(rte_loss_value, 4),  # Opportunity cost of lost energy
            # RTE-aware discharge tracking
            "rte_preserved_kwh": round(rte_preserved_kwh, 3),
            "rte_preserved_periods": rte_preserved_periods,
            "rte_breakeven_price": round(current_breakeven_price, 4),
            "rte_breakeven_source": rte_breakeven_source,
            "rte_solar_opportunity_price": round(rte_solar_opportunity_price, 4),
            # Energy consumption diagnostic tracking
            "energy_actual_kwh": round(energy_actual_kwh, 3),
            "energy_estimated_kwh": round(energy_estimated_kwh, 3),
            "energy_hours_with_actual": energy_hours_with_actual,
            "energy_hours_with_estimate": energy_hours_with_estimate,
        }

    def _build_result(
        self,
        prices: List[Dict[str, Any]],
        all_prices: List[Dict[str, Any]],
        charge_windows: List[Dict[str, Any]],
        discharge_windows: List[Dict[str, Any]],
        current_state: str,
        config: Dict[str, Any],
        is_tomorrow: bool,
        hass: Any = None,
        all_charge_candidates: List[Dict[str, Any]] = None,
        energy_statistics: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Build the result dictionary with all attributes.

        Args:
            all_charge_candidates: All windows passing profit threshold (for capacity-first selection).
                                   If provided, chrono uses these for feasibility, then selects cheapest N.
        """
        now = dt_util.now()
        current_time = now.replace(second=0, microsecond=0)
        current_price = self._get_current_price(prices, current_time)
        current_sell_price = self._get_current_sell_price(prices, current_time, config)

        # Get sell formula config
        sell_country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
        sell_param_a = config.get("sell_formula_param_a", DEFAULT_SELL_FORMULA_PARAM_A)
        sell_param_b = config.get("sell_formula_param_b", DEFAULT_SELL_FORMULA_PARAM_B)

        # Calculate avg buy price from charge windows (for operational spread)
        cheap_prices = [w["price"] for w in charge_windows]
        avg_cheap = float(np.mean(cheap_prices)) if cheap_prices else 0.0

        # Calculate avg sell price from discharge windows (for operational spread)
        avg_expensive = 0.0
        if discharge_windows:
            sell_prices = [w.get("sell_price", w["price"]) for w in discharge_windows]
            avg_expensive = float(np.mean(sell_prices))

        # Calculate spread_avg and arbitrage_avg for display (INDEPENDENT of window selection)
        # Uses percentile threshold to find cheapest/most expensive prices
        # These metrics show potential even when no windows are selected
        suffix = "_tomorrow" if is_tomorrow and config.get("tomorrow_settings_enabled", False) else ""
        percentile_threshold = config.get(f"percentile_threshold{suffix}", 25)

        # Get all buy prices
        all_buy_prices = np.array([p["price"] for p in prices])

        # Get cheapest buy prices (bottom percentile)
        cheap_threshold = np.percentile(all_buy_prices, percentile_threshold)
        cheap_buy_prices = all_buy_prices[all_buy_prices <= cheap_threshold]
        avg_cheap_buy = float(np.mean(cheap_buy_prices)) if len(cheap_buy_prices) > 0 else 0.0

        # Get most expensive buy prices (top percentile) - for spread_avg (buy vs buy)
        expensive_buy_threshold = np.percentile(all_buy_prices, 100 - percentile_threshold)
        expensive_buy_prices = all_buy_prices[all_buy_prices >= expensive_buy_threshold]
        avg_expensive_buy = float(np.mean(expensive_buy_prices)) if len(expensive_buy_prices) > 0 else 0.0

        # Calculate spread_avg (buy vs buy) - price volatility
        # Shows how much prices vary throughout the day
        spread_avg = 0.0
        if avg_cheap_buy > 0:
            spread_avg = float(((avg_expensive_buy - avg_cheap_buy) / avg_cheap_buy) * 100)

        # Calculate sell prices for all time slots
        all_sell_prices = []
        for p in prices:
            raw = p.get("raw_price", p["price"])
            sell = self._calculate_sell_price(raw, p["price"], sell_country, sell_param_a, sell_param_b)
            all_sell_prices.append(sell)
        all_sell_prices = np.array(all_sell_prices)

        # Get most expensive sell prices (top percentile) - for arbitrage_avg (sell vs buy)
        expensive_sell_threshold = np.percentile(all_sell_prices, 100 - percentile_threshold)
        expensive_sell_prices = all_sell_prices[all_sell_prices >= expensive_sell_threshold]
        avg_expensive_sell = float(np.mean(expensive_sell_prices)) if len(expensive_sell_prices) > 0 else 0.0

        # Calculate arbitrage_avg (sell vs buy) - arbitrage margin
        # Shows profitability of battery arbitrage (can be negative when sell < buy)
        arbitrage_avg = 0.0
        if avg_cheap_buy > 0:
            arbitrage_avg = float(((avg_expensive_sell - avg_cheap_buy) / avg_cheap_buy) * 100)

        # Calculate actual windows considering time and price overrides
        actual_charge, actual_discharge = self._calculate_actual_windows(
            prices,
            charge_windows,
            discharge_windows,
            config,
            is_tomorrow
        )

        # Count completed windows (use actual windows to include price/time overrides)
        completed_charge = sum(
            1 for w in actual_charge
            if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time
        )
        completed_discharge = sum(
            1 for w in actual_discharge
            if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time
        )

        # Calculate costs with base usage strategies
        charge_power = config.get("charge_power", 2400) / 1000  # Convert to kW
        discharge_power = config.get("discharge_power", 2400) / 1000
        manual_base_usage = config.get("base_usage", 0) / 1000
        battery_rte = config.get("battery_rte", 85) / 100  # Convert to decimal

        energy_stats = energy_statistics or {}
        use_ha_energy = config.get("use_ha_energy_dashboard", False) and energy_stats.get("stats_available", False)
        # Use weighted 72h average for base_usage when HA Energy enabled
        weighted_avg = energy_stats.get("weighted_avg_consumption", 0.0) if use_ha_energy else 0.0
        base_usage = weighted_avg if (use_ha_energy and weighted_avg > 0) else manual_base_usage
        real_consumption_hourly = energy_stats.get("real_consumption_hourly", {}) if use_ha_energy else {}
        avg_real_consumption = energy_stats.get("avg_real_consumption", 0.0) if use_ha_energy else 0.0
        grid_import_hourly = energy_stats.get("grid_import_hourly", {}) if use_ha_energy else {}
        grid_export_hourly = energy_stats.get("grid_export_hourly", {}) if use_ha_energy else {}
        solar_hourly = energy_stats.get("solar_hourly", {}) if use_ha_energy else {}
        battery_charge_hourly = energy_stats.get("battery_charge_hourly", {}) if use_ha_energy else {}
        battery_discharge_hourly = energy_stats.get("battery_discharge_hourly", {}) if use_ha_energy else {}
        current_hour = dt_util.now().hour

        actual_battery_flows = self._calculate_actual_battery_flows(
            energy_statistics, all_prices, config, current_hour
        ) if use_ha_energy and not is_tomorrow else {
            "charged_from_grid_kwh": 0.0,
            "charged_from_grid_cost": 0.0,
            "charged_from_solar_kwh": 0.0,
            "discharged_to_base_kwh": 0.0,
            "discharged_to_grid_kwh": 0.0,
            "discharged_revenue": 0.0,
        }

        # Manual charging detection will be called after actual_charge is determined
        # (placeholder - actual call happens later in the function)
        manual_charging_info = {
            "manual_charge_hours": [],
            "manual_charge_kwh": 0.0,
            "manual_charge_cost": 0.0,
            "manual_charge_detected": False,
        }

        def get_effective_base_usage(timestamp):
            """Get base usage for a specific period, using HA Energy data if available."""
            if use_ha_energy:
                period_hour = timestamp.hour if hasattr(timestamp, 'hour') else 0
                if period_hour in real_consumption_hourly:
                    return real_consumption_hourly[period_hour]  # kWh/hour = kW
                elif avg_real_consumption > 0:
                    return avg_real_consumption
            return base_usage

        # Get strategies
        charge_strategy = config.get("base_usage_charge_strategy", "grid_covers_both")
        normal_strategy = config.get("base_usage_normal_strategy", "grid_covers")
        discharge_strategy = config.get("base_usage_discharge_strategy", "subtract_base")

        # Get buffer energy early - needed for usable_kwh calculations
        # Note: min_usable threshold is applied later at the RTE preservation decision point
        buffer_energy = self._get_buffer_energy(config, is_tomorrow, hass)

        # Get sell price config for revenue calculations
        sell_country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
        sell_param_a = config.get("sell_formula_param_a", DEFAULT_SELL_FORMULA_PARAM_A)
        sell_param_b = config.get("sell_formula_param_b", DEFAULT_SELL_FORMULA_PARAM_B)

        # Build price lookup from all_prices (not filtered prices) to ensure all timestamps are covered
        price_lookup = {p["timestamp"]: p for p in all_prices}

        # Initialize tracking variables
        completed_charge_cost = 0
        completed_discharge_revenue = 0
        completed_solar_export_revenue = 0  # Revenue from solar exports in completed periods
        completed_base_usage_cost = 0  # Grid cost for base usage
        completed_base_usage_battery = 0  # Battery kWh used for base usage
        # kWh tracking for completed periods
        completed_charge_kwh = 0  # Grid draw for charging
        completed_discharge_kwh = 0  # Export to grid
        completed_base_grid_kwh = 0  # Grid draw for uncovered base usage
        completed_rte_loss_kwh = 0  # RTE loss from completed charging
        completed_rte_loss_value = 0  # Cost of energy lost to RTE during charging

        # CHARGE windows: Apply charge strategy
        if use_ha_energy and battery_charge_hourly:
            # Sum actual battery charging for hours that have completed
            for hour, charge_kwh in battery_charge_hourly.items():
                hour_int = int(hour) if isinstance(hour, str) else hour
                if hour_int < current_hour:  # Only fully completed hours
                    completed_charge_kwh += charge_kwh
                    # Find price for this hour to calculate cost
                    for price_data in all_prices:
                        if price_data["timestamp"].hour == hour_int:
                            completed_charge_cost += charge_kwh * price_data["price"]
                            # Calculate RTE loss for this hour's charging
                            rte_loss_kwh = charge_kwh * (1 - battery_rte)
                            completed_rte_loss_kwh += rte_loss_kwh
                            completed_rte_loss_value += rte_loss_kwh * price_data["price"]
                            break
        else:
            # Fallback to scheduled window calculation
            for w in actual_charge:
                if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time:
                    duration_hours = w["duration"] / 60
                    # RTE loss applies to charge power going to battery (not base usage which goes directly to house)
                    charge_kwh_to_battery = duration_hours * charge_power
                    rte_loss_kwh = charge_kwh_to_battery * (1 - battery_rte)
                    completed_rte_loss_kwh += rte_loss_kwh
                    # Look up verified buy price from price_lookup (w["price"] may have spot price in some cases)
                    price_data = price_lookup.get(w["timestamp"], {})
                    buy_price = price_data.get("price", w["price"])  # Fallback to w["price"] if not found
                    completed_rte_loss_value += rte_loss_kwh * buy_price  # Cost of lost energy

                    effective_base = get_effective_base_usage(w["timestamp"])
                    if charge_strategy == "grid_covers_both":
                        # Grid provides charge power + base usage
                        completed_charge_cost += w["price"] * duration_hours * (charge_power + effective_base)
                        completed_charge_kwh += duration_hours * (charge_power + effective_base)
                    else:  # battery_covers_base
                        # Grid provides charge power only, battery covers base
                        completed_charge_cost += w["price"] * duration_hours * charge_power
                        completed_charge_kwh += duration_hours * charge_power
                        completed_base_usage_battery += duration_hours * effective_base

        # DISCHARGE windows: Apply discharge strategy
        if use_ha_energy and battery_discharge_hourly:
            # Sum actual battery discharging for hours that have completed
            for hour, discharge_kwh in battery_discharge_hourly.items():
                hour_int = int(hour) if isinstance(hour, str) else hour
                if hour_int < current_hour:  # Only fully completed hours
                    completed_discharge_kwh += discharge_kwh
                    # Find price for this hour to calculate revenue
                    for price_data in all_prices:
                        if price_data["timestamp"].hour == hour_int:
                            raw_price = price_data.get("raw_price", price_data["price"])
                            sell_price = self._calculate_sell_price(
                                raw_price, price_data["price"], sell_country, sell_param_a, sell_param_b
                            )
                            completed_discharge_revenue += discharge_kwh * sell_price
                            break
        else:
            # Fallback to scheduled window calculation
            for w in actual_discharge:
                if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time:
                    duration_hours = w["duration"] / 60

                    # Calculate sell price for this window
                    price_data = price_lookup.get(w["timestamp"], {})
                    raw_price = price_data.get("raw_price", w["price"])
                    sell_price = self._calculate_sell_price(
                        raw_price, w["price"], sell_country, sell_param_a, sell_param_b
                    )

                    effective_base = get_effective_base_usage(w["timestamp"])
                    if discharge_strategy == "already_included":
                        # Full discharge power generates revenue at SELL price
                        completed_discharge_revenue += sell_price * duration_hours * discharge_power
                        completed_discharge_kwh += duration_hours * discharge_power
                    else:  # subtract_base (NoM)
                        # Battery covers base first, exports the rest at SELL price
                        net_export = max(0, discharge_power - effective_base)
                        completed_discharge_revenue += sell_price * duration_hours * net_export
                        completed_discharge_kwh += duration_hours * net_export
                        completed_base_usage_battery += duration_hours * effective_base

        # NORMAL periods: Apply normal strategy
        # Build sets of timestamps for active windows
        charge_timestamps = {w["timestamp"] for w in actual_charge}
        discharge_timestamps = {w["timestamp"] for w in actual_discharge}

        # Track remaining battery for base usage
        # Use buffer_energy which already accounts for sensor reading (from _get_buffer_energy)
        # This tracks battery draining during normal periods
        remaining_battery_for_base = buffer_energy

        # Use all_prices (not filtered by calculation window) for base usage calculations
        # Base usage should cover full 24h regardless of calculation window
        for price_data in all_prices:
            timestamp = price_data["timestamp"]
            if timestamp + timedelta(minutes=price_data["duration"]) <= current_time:
                # Check if this period is normal (not in any active window)
                is_active = timestamp in charge_timestamps or timestamp in discharge_timestamps

                if not is_active:
                    duration_hours = price_data["duration"] / 60
                    effective_base = get_effective_base_usage(timestamp)
                    if normal_strategy == "grid_covers":
                        # Grid provides base usage, add to cost
                        completed_base_usage_cost += price_data["price"] * duration_hours * effective_base
                        completed_base_grid_kwh += duration_hours * effective_base
                    else:  # battery_covers or battery_covers_limited
                        # Battery provides base usage, tracking battery draining
                        base_kwh_needed = duration_hours * effective_base

                        if remaining_battery_for_base >= base_kwh_needed:
                            # Battery can cover full period
                            completed_base_usage_battery += base_kwh_needed
                            remaining_battery_for_base -= base_kwh_needed
                        elif remaining_battery_for_base > 0:
                            # Battery covers partial, grid covers rest
                            completed_base_usage_battery += remaining_battery_for_base
                            grid_kwh = base_kwh_needed - remaining_battery_for_base
                            grid_hours = grid_kwh / effective_base if effective_base > 0 else 0
                            completed_base_usage_cost += price_data["price"] * grid_hours * effective_base
                            completed_base_grid_kwh += grid_kwh
                            remaining_battery_for_base = 0
                        else:
                            # Battery empty, grid covers all
                            completed_base_usage_cost += price_data["price"] * duration_hours * effective_base
                            completed_base_grid_kwh += duration_hours * effective_base

        # Calculate planned total cost for ALL windows (for tomorrow's estimate)
        # Unlike total_cost which only counts completed windows, this estimates the full day
        planned_charge_cost = 0
        planned_discharge_revenue = 0
        planned_base_usage_cost = 0

        # All charge windows (not just completed)
        for w in actual_charge:
            duration_hours = w["duration"] / 60
            if charge_strategy == "grid_covers_both":
                planned_charge_cost += w["price"] * duration_hours * (charge_power + base_usage)
            else:  # battery_covers_base
                planned_charge_cost += w["price"] * duration_hours * charge_power

        # All discharge windows (not just completed)
        for w in actual_discharge:
            duration_hours = w["duration"] / 60

            # Calculate sell price for this window
            price_data = price_lookup.get(w["timestamp"], {})
            raw_price = price_data.get("raw_price", w["price"])
            sell_price = self._calculate_sell_price(
                raw_price, w["price"], sell_country, sell_param_a, sell_param_b
            )

            if discharge_strategy == "already_included":
                planned_discharge_revenue += sell_price * duration_hours * discharge_power
            else:  # subtract_base
                net_export = max(0, discharge_power - base_usage)
                planned_discharge_revenue += sell_price * duration_hours * net_export

        # All normal periods (use all_prices for full 24h coverage)
        for price_data in all_prices:
            timestamp = price_data["timestamp"]
            is_active = timestamp in charge_timestamps or timestamp in discharge_timestamps

            if not is_active:
                duration_hours = price_data["duration"] / 60
                if normal_strategy == "grid_covers":
                    planned_base_usage_cost += price_data["price"] * duration_hours * base_usage

        planned_total_cost = round(planned_charge_cost + planned_base_usage_cost - planned_discharge_revenue, 3)

        # Calculate effective base usage (limited by charged energy when normal strategy is "battery_covers_limited")
        # This is used by the dashboard for Estimated Savings calculation
        limit_savings_enabled = (normal_strategy == "battery_covers_limited")
        base_usage_kwh = base_usage * 24  # Full day base usage in kWh

        # Calculate net planned charge (accounts for battery_covers_base strategy)
        # When battery_covers_base: battery outputs base_usage while charging, so net charge = charge_power - base_usage
        # When grid_covers_both: full charge_power goes to battery
        net_planned_charge_kwh = 0
        for w in actual_charge:
            duration_hours = w["duration"] / 60
            if charge_strategy == "battery_covers_base":
                net_planned_charge_kwh += duration_hours * max(0, charge_power - base_usage)
            else:  # grid_covers_both
                net_planned_charge_kwh += duration_hours * charge_power
        net_planned_charge_kwh = round(net_planned_charge_kwh, 3)

        # Calculate net planned discharge (accounts for subtract_base strategy)
        # When subtract_base: only net export goes to grid (discharge - base_usage)
        # When already_included: full discharge_power goes to grid
        # NOTE: Must calculate BEFORE usable_kwh so we can account for discharged energy
        net_planned_discharge_kwh = 0
        for w in actual_discharge:
            duration_hours = w["duration"] / 60

            if discharge_strategy == "already_included":
                # Full discharge power goes to grid
                net_planned_discharge_kwh += duration_hours * discharge_power
            else:  # subtract_base
                # Only net export goes to grid (discharge - base_usage)
                net_planned_discharge_kwh += duration_hours * max(0, discharge_power - base_usage)
        net_planned_discharge_kwh = round(net_planned_discharge_kwh, 3)

        # Calculate effective base usage (limit to usable energy if toggle is on)
        battery_rte_pct = config.get("battery_rte", 85)
        battery_rte_decimal = battery_rte_pct / 100
        # usable_kwh = energy available for base usage = buffer + charged - discharged
        # Note: RTE is NOT applied here - it only affects battery state simulation, not cost calculations
        usable_kwh = max(0, buffer_energy + net_planned_charge_kwh - net_planned_discharge_kwh)
        if limit_savings_enabled:
            effective_base_usage_kwh = min(base_usage_kwh, usable_kwh)
        else:
            effective_base_usage_kwh = base_usage_kwh
        effective_base_usage_kwh = round(effective_base_usage_kwh, 3)

        # Calculate uncovered base usage cost (when limit enabled and battery can't cover all)
        # Use all_prices for day average (full 24h, not filtered by calculation window)
        day_avg_price = float(np.mean([p["price"] for p in all_prices])) if all_prices else 0
        if limit_savings_enabled and usable_kwh < base_usage_kwh:
            uncovered_kwh = base_usage_kwh - usable_kwh
            uncovered_cost = uncovered_kwh * day_avg_price
        else:
            uncovered_kwh = 0
            uncovered_cost = 0

        # Update planned_total_cost to include uncovered base usage cost
        planned_total_cost = round(planned_total_cost + uncovered_cost, 3)

        # Calculate completed (time-proportional) uncovered cost for total_cost
        # uncovered_cost is full-day projection; total_cost needs only elapsed portion
        if all_prices and limit_savings_enabled and uncovered_cost > 0:
            day_start = all_prices[0]["timestamp"]
            day_end = all_prices[-1]["timestamp"] + timedelta(minutes=all_prices[-1]["duration"])
            total_day_seconds = (day_end - day_start).total_seconds()
            elapsed_seconds = max(0, min((current_time - day_start).total_seconds(), total_day_seconds))
            time_fraction = elapsed_seconds / total_day_seconds if total_day_seconds > 0 else 0
            completed_uncovered_cost = round(uncovered_cost * time_fraction, 3)
        else:
            completed_uncovered_cost = 0

        # Calculate sell prices for discharge windows
        def get_sell_price_for_window(w):
            pd = price_lookup.get(w["timestamp"], {})
            raw = pd.get("raw_price", w["price"])
            return self._calculate_sell_price(raw, w["price"], sell_country, sell_param_a, sell_param_b)

        # Calculate profit percentages (profit = spread - RTE_loss)
        battery_rte_pct = config.get("battery_rte", 85)  # Use different var name to avoid shadowing decimal battery_rte
        rte_loss = 100 - battery_rte_pct
        charge_profit_pct = spread_avg - rte_loss  # Buy-buy spread for charging
        discharge_profit_pct = arbitrage_avg - rte_loss  # Buy-sell spread for discharge

        # Get profit thresholds from config
        min_profit_charge = config.get(f"min_profit_charge{suffix}", 10)
        min_profit_discharge = config.get(f"min_profit_discharge{suffix}", 10)

        # === DASHBOARD HELPER ATTRIBUTES ===
        # These reduce repetitive calculations in the dashboard

        # Grouped windows (consecutive windows merged into blocks)
        grouped_charge_windows = self._group_consecutive_windows(
            actual_charge, avg_expensive_buy, is_discharge=False
        )
        grouped_discharge_windows = self._group_consecutive_windows(
            actual_discharge, avg_cheap_buy, is_discharge=True
        )

        # Sorted prices for dashboard use
        sorted_buy_prices = sorted([p["price"] for p in prices])
        sorted_sell_prices = sorted(all_sell_prices.tolist())

        # Percentile averages (already calculated, just expose)
        percentile_cheap_avg = avg_cheap_buy
        percentile_expensive_avg = avg_expensive_buy
        percentile_expensive_sell_avg = avg_expensive_sell

        # Cheap/expensive half averages (for fallback display when no windows)
        half_idx = len(sorted_buy_prices) // 2
        cheap_half_avg = float(np.mean(sorted_buy_prices[:half_idx])) if half_idx > 0 else 0
        expensive_half_avg = float(np.mean(sorted_buy_prices[half_idx:])) if half_idx < len(sorted_buy_prices) else 0

        # Estimated savings calculations
        # These match what the dashboard was calculating in Jinja2
        battery_rte_decimal = battery_rte / 100
        window_duration_hours = (prices[0]["duration"] / 60) if prices else 0.25

        # Calculate charged/discharged kWh using effective power (respects base usage strategies)
        effective_charge_power = (charge_power - base_usage) if charge_strategy == "battery_covers_base" else charge_power
        effective_discharge_power = (discharge_power - base_usage) if discharge_strategy == "subtract_base" else discharge_power

        charged_kwh = len(actual_charge) * window_duration_hours * effective_charge_power
        # Note: RTE is NOT applied here - it only affects battery state simulation, not cost calculations
        # Include starting buffer in usable energy for base usage coverage
        usable_kwh = charged_kwh + buffer_energy
        discharged_kwh = len(actual_discharge) * window_duration_hours * effective_discharge_power
        remaining_kwh = usable_kwh - discharged_kwh

        # Uncovered base usage (when battery can't cover all base usage)
        uncovered_base = max(0, base_usage_kwh - max(0, remaining_kwh))

        # Net grid kWh (imports - exports)
        net_grid_kwh = (charged_kwh + uncovered_base) - discharged_kwh

        # Baseline cost (what you'd pay at average price)
        baseline_cost = net_grid_kwh * day_avg_price

        # Estimated savings
        estimated_savings = baseline_cost - planned_total_cost

        # Gross values for Battery line matching (needed for true_savings calculation)
        gross_charged_kwh = len(actual_charge) * window_duration_hours * charge_power
        # Note: RTE is NOT applied here - it only affects battery state simulation, not cost/display calculations
        gross_usable_kwh = gross_charged_kwh  # No RTE penalty
        gross_discharged_kwh = len(actual_discharge) * window_duration_hours * discharge_power
        actual_remaining_kwh = gross_usable_kwh - gross_discharged_kwh

        # True savings = savings on what we KEPT (base usage + remaining buffer)
        # Savings rate per kWh (avg price - actual price paid)
        savings_per_kwh = day_avg_price - (planned_total_cost / net_grid_kwh) if net_grid_kwh > 0 else 0
        true_savings = savings_per_kwh * (base_usage_kwh + max(0, actual_remaining_kwh))

        # Net post-discharge metrics (battery arbitrage value)

        # Net post-discharge €/kWh
        total_charge_cost = sum(w["price"] for w in actual_charge) * window_duration_hours * effective_charge_power if actual_charge else 0

        # Calculate discharge revenue - need to get sell prices for each window
        discharge_sell_prices_sum = 0
        for w in actual_discharge:
            if "sell_price" in w:
                discharge_sell_prices_sum += w["sell_price"]
            else:
                raw = price_lookup.get(w["timestamp"], {}).get("raw_price", w["price"])
                discharge_sell_prices_sum += self._calculate_sell_price(raw, w["price"], sell_country, sell_param_a, sell_param_b)
        discharge_revenue = discharge_sell_prices_sum * window_duration_hours * effective_discharge_power

        net_post_discharge_eur_kwh = ((total_charge_cost - discharge_revenue) / remaining_kwh) if remaining_kwh > 0 else 0
        battery_margin_eur_kwh = day_avg_price - net_post_discharge_eur_kwh
        battery_arbitrage_value = actual_remaining_kwh * battery_margin_eur_kwh if actual_remaining_kwh > 0 else 0
        # Initialize end-of-day buffer value (will be recalculated by chrono if available)
        battery_state_end_of_day_value = battery_arbitrage_value

        # === CHRONOLOGICAL BUFFER TRACKING ===
        # Get buffer energy and limiting mode from config
        buffer_energy = self._get_buffer_energy(config, is_tomorrow, hass)

        # Apply minimum usable threshold - below this level, battery is considered depleted
        min_usable_kwh = config.get("battery_min_usable_kwh", 0.0)
        if min_usable_kwh > 0:
            buffer_energy = max(0, buffer_energy - min_usable_kwh)

        limit_discharge = config.get("limit_discharge_to_buffer", False)
        battery_capacity = config.get("battery_capacity", 100.0)

        # For capacity-first selection: use ALL charge candidates for chrono simulation
        # This ensures chrono can determine feasibility for all candidates, not just limited N
        # After chrono, we select the cheapest N from feasible windows
        suffix = "_tomorrow" if is_tomorrow and config.get("tomorrow_settings_enabled", False) else ""
        num_charge_windows = int(config.get(f"charging_windows{suffix}", 4))

        if all_charge_candidates and len(all_charge_candidates) > len(actual_charge):
            # Apply overrides to ALL candidates (same logic as actual_charge)
            all_actual_charge, _ = self._calculate_actual_windows(
                prices,
                all_charge_candidates,  # Use all candidates
                discharge_windows,
                config,
                is_tomorrow
            )
            charge_for_chrono = all_actual_charge
        else:
            charge_for_chrono = actual_charge

        # Build chronological timeline and simulate battery state
        chrono_timeline = self._build_chronological_timeline(
            all_prices,  # Use all_prices for full day coverage
            charge_for_chrono,  # Use all candidates for capacity-first selection
            actual_discharge,
            config
        )

        # Determine simulation mode based on battery sensor and HA Energy toggles
        use_sensor = config.get("use_battery_buffer_sensor", False)
        sensor_entity = config.get("battery_available_energy_sensor", "")
        using_sensor_for_today = not is_tomorrow and use_sensor and sensor_entity and hass

        use_ha_energy = config.get("use_ha_energy_dashboard", False)

        # When HA Energy is enabled, force battery-aware mode for accurate tracking
        if use_ha_energy:
            limit_discharge = True

        simulation_timeline = chrono_timeline

        chrono_result = self._simulate_chronological_costs(
            simulation_timeline, config, buffer_energy, limit_discharge, is_tomorrow, hass,
            energy_statistics=energy_statistics,
            actual_battery_flows=actual_battery_flows if use_ha_energy else None
        )

        # Calculate buffer delta (end - start)
        final_battery_state = chrono_result["final_battery_state"]
        buffer_delta = final_battery_state - buffer_energy

        # Calculate completed solar export revenue and kWh from solar_export_events
        solar_export_events = chrono_result.get("solar_export_events", [])
        completed_solar_export_kwh = 0
        for event in solar_export_events:
            event_end = event["timestamp"] + timedelta(minutes=event["duration"])
            if event_end <= current_time:
                completed_solar_export_revenue += event["revenue"]
                completed_solar_export_kwh += event.get("kwh", 0)
        completed_solar_export_revenue = round(completed_solar_export_revenue, 3)
        completed_solar_export_kwh = round(completed_solar_export_kwh, 3)

        # Calculate completed solar grid savings (solar offsetting base usage)
        solar_base_offset_events = chrono_result.get("solar_base_offset_events", [])
        completed_solar_grid_savings = 0
        completed_solar_base_kwh = 0
        for event in solar_base_offset_events:
            event_end = event["timestamp"] + timedelta(minutes=event["duration"])
            if event_end <= current_time:
                completed_solar_grid_savings += event["savings"]
                completed_solar_base_kwh += event["kwh"]
        completed_solar_grid_savings = round(completed_solar_grid_savings, 3)
        completed_solar_base_kwh = round(completed_solar_base_kwh, 3)

        # Find current battery state
        # Priority: real sensor value (for today) > trajectory simulation > buffer_energy
        current_battery_state = buffer_energy  # Default to starting value
        use_sensor = config.get("use_battery_buffer_sensor", False)
        sensor_entity = config.get("battery_available_energy_sensor", "")
        sensor_value_obtained = False

        # For today: use real sensor value if available (most accurate)
        if not is_tomorrow and use_sensor and sensor_entity and hass:
            try:
                sensor_state = hass.states.get(sensor_entity)
                if sensor_state and sensor_state.state not in ("unknown", "unavailable", None):
                    # Show raw sensor value - min_usable threshold is applied in RTE calculations only
                    current_battery_state = float(sensor_state.state)
                    sensor_value_obtained = True
            except (ValueError, TypeError):
                pass  # Fall back to trajectory below

        # If no sensor value obtained, use simulated trajectory
        if not sensor_value_obtained:
            battery_trajectory = chrono_result.get("battery_trajectory", [])
            if battery_trajectory and not is_tomorrow:
                now = dt_util.now()
                for entry in battery_trajectory:
                    entry_ts = entry["timestamp"]
                    if isinstance(entry_ts, str):
                        entry_ts = dt_util.parse_datetime(entry_ts)
                    if entry_ts and entry_ts <= now:
                        current_battery_state = entry.get("battery_after", current_battery_state)
                    else:
                        break

        # Always filter charge windows based on battery capacity (physical constraint)
        # If battery is full, charge windows are skipped regardless of mode
        original_charge_count = len(actual_charge)
        feasible_charge = chrono_result.get("feasible_charge_windows", [])
        skipped_charge = chrono_result.get("skipped_charge_windows", [])

        # CHRONOLOGICAL FILTER: Exclude charge windows after last discharge window
        # Reason: Charging after all discharges increases cost without same-day benefit
        # The charged energy would only be useful for tomorrow, not minimizing today's cost
        feasible_discharge = chrono_result.get("feasible_discharge_windows", [])
        if feasible_discharge:
            last_discharge_time = max(d["timestamp"] for d in feasible_discharge)
            late_charge_count = len([c for c in feasible_charge if c["timestamp"] > last_discharge_time])
            if late_charge_count > 0:
                feasible_charge = [c for c in feasible_charge if c["timestamp"] <= last_discharge_time]
                _LOGGER.info(
                    f"Chronological filter: Excluded {late_charge_count} charge window(s) after last discharge "
                    f"({last_discharge_time.strftime('%H:%M')}) - no same-day benefit"
                )

        # CAPACITY-FIRST SELECTION: From all feasible windows, select the cheapest N
        # This ensures that increasing max windows never DECREASES elected windows
        # Feasibility is determined by battery capacity (chrono), selection by price
        if all_charge_candidates and len(feasible_charge) > num_charge_windows:
            # Sort feasible windows by price (cheapest first)
            feasible_charge_sorted = sorted(feasible_charge, key=lambda x: x["price"])
            # Keep only the cheapest num_charge_windows
            feasible_charge = feasible_charge_sorted[:num_charge_windows]
            _LOGGER.info(
                f"Capacity-first selection: {len(chrono_result.get('feasible_charge_windows', []))} feasible → "
                f"{len(feasible_charge)} cheapest selected (max={num_charge_windows})"
            )

        # ALWAYS use feasible charge windows from chronological simulation
        # This ensures we never show charge windows that couldn't execute (battery full, etc.)
        actual_charge = feasible_charge

        # Detect manual charging (outside CEW planned windows)
        if use_ha_energy and not is_tomorrow:
            manual_charging_info = self._detect_manual_charging(
                energy_statistics, actual_charge, all_prices, current_hour
            )

        # REBUILD grouped windows with filtered charge windows (for dashboard display)
        grouped_charge_windows = self._group_consecutive_windows(
            actual_charge, avg_expensive_sell, is_discharge=False
        )
        # RECALCULATE completed_charge from filtered windows (was calculated before filtering)
        completed_charge = sum(
            1 for w in actual_charge
            if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time
        )
        # Recalculate completed_charge_cost from filtered windows
        completed_charge_cost = 0
        if use_ha_energy and battery_charge_hourly:
            # Calculate cost from actual battery charging hourly data
            for hour, charge_kwh in battery_charge_hourly.items():
                hour_int = int(hour) if isinstance(hour, str) else hour
                if hour_int < current_hour:
                    # Find price for this hour
                    for price_data in all_prices:
                        if price_data["timestamp"].hour == hour_int:
                            completed_charge_cost += charge_kwh * price_data["price"]
                            break
        else:
            for w in actual_charge:
                if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time:
                    duration_hours = w["duration"] / 60
                    effective_base = get_effective_base_usage(w["timestamp"])
                    if charge_strategy == "grid_covers_both":
                        completed_charge_cost += w["price"] * duration_hours * (charge_power + effective_base)
                    else:  # battery_covers_base
                        completed_charge_cost += w["price"] * duration_hours * charge_power
        # Recalculate completed_charge_kwh from filtered windows
        completed_charge_kwh = 0
        if use_ha_energy and battery_charge_hourly:
            # Sum actual battery charging for completed hours
            for hour, charge_kwh in battery_charge_hourly.items():
                hour_int = int(hour) if isinstance(hour, str) else hour
                if hour_int < current_hour:
                    completed_charge_kwh += charge_kwh
        else:
            for w in actual_charge:
                if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time:
                    duration_hours = w["duration"] / 60
                    effective_base = get_effective_base_usage(w["timestamp"])
                    if charge_strategy == "grid_covers_both":
                        completed_charge_kwh += duration_hours * (charge_power + effective_base)
                    else:
                        completed_charge_kwh += duration_hours * charge_power
        # RECALCULATE completed_rte_loss from filtered windows
        completed_rte_loss_kwh = 0
        completed_rte_loss_value = 0
        for w in actual_charge:
            if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time:
                duration_hours = w["duration"] / 60
                charge_kwh_to_battery = duration_hours * charge_power
                rte_loss_kwh = charge_kwh_to_battery * (1 - battery_rte)
                completed_rte_loss_kwh += rte_loss_kwh
                # Look up verified buy price from price_lookup (w["price"] may have spot price in some cases)
                price_data = price_lookup.get(w["timestamp"], {})
                buy_price = price_data.get("price", w["price"])  # Fallback to w["price"] if not found
                completed_rte_loss_value += rte_loss_kwh * buy_price
        _LOGGER.info(
            f"Charge windows filtered: original={original_charge_count}, "
            f"feasible={len(actual_charge)}, skipped={len(skipped_charge)} (battery full), "
            f"completed={completed_charge}"
        )

        # ALWAYS use feasible discharge windows from chronological simulation
        # This ensures we never show discharge windows that couldn't execute (empty battery, etc.)
        original_discharge_count = len(actual_discharge)
        actual_discharge = chrono_result["feasible_discharge_windows"]

        # REBUILD grouped windows with filtered discharge windows (for dashboard display)
        grouped_discharge_windows = self._group_consecutive_windows(
            actual_discharge, avg_cheap_buy, is_discharge=True
        )

        # RECALCULATE completed_discharge from filtered windows (was calculated before filtering)
        completed_discharge = sum(
            1 for w in actual_discharge
            if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time
        )
        # Recalculate completed_discharge_revenue from filtered windows
        completed_discharge_revenue = 0
        for w in actual_discharge:
            if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time:
                duration_hours = w["duration"] / 60
                # Calculate sell price for this window
                price_data = price_lookup.get(w["timestamp"], {})
                raw_price = price_data.get("raw_price", w["price"])
                sell_price = self._calculate_sell_price(
                    raw_price, w["price"], sell_country, sell_param_a, sell_param_b
                )
                effective_base = get_effective_base_usage(w["timestamp"])
                if discharge_strategy == "already_included":
                    completed_discharge_revenue += sell_price * duration_hours * discharge_power
                else:  # subtract_base
                    net_export = max(0, discharge_power - effective_base)
                    completed_discharge_revenue += sell_price * duration_hours * net_export
        # Recalculate completed_discharge_kwh from filtered windows
        completed_discharge_kwh = 0
        for w in actual_discharge:
            if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time:
                duration_hours = w["duration"] / 60
                effective_base = get_effective_base_usage(w["timestamp"])
                if discharge_strategy == "already_included":
                    completed_discharge_kwh += duration_hours * discharge_power
                else:
                    net_export = max(0, discharge_power - effective_base)
                    completed_discharge_kwh += duration_hours * net_export

        _LOGGER.info(
            f"Discharge windows filtered: original={original_discharge_count} windows, "
            f"feasible={len(actual_discharge)} windows, "
            f"skipped={len(chrono_result['skipped_discharge_windows'])} windows, "
            f"completed={completed_discharge}, "
            f"final_battery={chrono_result['final_battery_state']:.2f} kWh"
        )

        # RE-RUN chrono with ELECTED windows after capacity-first selection
        # The first chrono run used ALL candidates to determine feasibility
        # Now we run again with only the elected windows to get correct trajectory
        if all_charge_candidates and len(all_charge_candidates) > len(charge_windows):
            # Rebuild timeline with ELECTED windows only (not all candidates)
            elected_timeline = self._build_chronological_timeline(
                all_prices,
                actual_charge,  # ELECTED windows, not all candidates
                actual_discharge,
                config
            )
            # Re-run chrono to get correct trajectory for elected windows
            elected_chrono_result = self._simulate_chronological_costs(
                elected_timeline, config, buffer_energy, limit_discharge, is_tomorrow, hass,
                energy_statistics=energy_statistics,
                actual_battery_flows=actual_battery_flows if use_ha_energy else None
            )
            # Update trajectory and battery state values from elected run
            chrono_result["battery_trajectory"] = elected_chrono_result["battery_trajectory"]
            chrono_result["final_battery_state"] = elected_chrono_result["final_battery_state"]
            chrono_result["actual_charge_kwh"] = elected_chrono_result["actual_charge_kwh"]
            chrono_result["actual_discharge_kwh"] = elected_chrono_result["actual_discharge_kwh"]
            chrono_result["grid_kwh_total"] = elected_chrono_result["grid_kwh_total"]
            chrono_result["uncovered_base_kwh"] = elected_chrono_result["uncovered_base_kwh"]
            chrono_result["planned_total_cost"] = elected_chrono_result["planned_total_cost"]
            chrono_result["planned_charge_cost"] = elected_chrono_result["planned_charge_cost"]
            chrono_result["planned_discharge_revenue"] = elected_chrono_result["planned_discharge_revenue"]
            chrono_result["planned_base_usage_cost"] = elected_chrono_result["planned_base_usage_cost"]
            # Solar integration metrics
            chrono_result["solar_to_battery_kwh"] = elected_chrono_result["solar_to_battery_kwh"]
            chrono_result["solar_offset_base_kwh"] = elected_chrono_result["solar_offset_base_kwh"]
            chrono_result["solar_exported_kwh"] = elected_chrono_result["solar_exported_kwh"]
            chrono_result["solar_export_revenue"] = elected_chrono_result["solar_export_revenue"]
            chrono_result["solar_total_contribution_kwh"] = elected_chrono_result["solar_total_contribution_kwh"]
            chrono_result["grid_savings_from_solar"] = elected_chrono_result["grid_savings_from_solar"]
            chrono_result["expected_solar_kwh"] = elected_chrono_result["expected_solar_kwh"]
            # Battery tracking from elected chrono
            chrono_result["battery_charged_from_grid_kwh"] = elected_chrono_result.get("battery_charged_from_grid_kwh", 0.0)
            chrono_result["battery_charged_from_grid_cost"] = elected_chrono_result.get("battery_charged_from_grid_cost", 0.0)
            chrono_result["battery_charged_from_solar_kwh"] = elected_chrono_result.get("battery_charged_from_solar_kwh", 0.0)
            chrono_result["battery_charged_avg_price"] = elected_chrono_result.get("battery_charged_avg_price", 0.0)
            chrono_result["battery_discharged_to_base_kwh"] = elected_chrono_result.get("battery_discharged_to_base_kwh", 0.0)
            chrono_result["battery_discharged_to_grid_kwh"] = elected_chrono_result.get("battery_discharged_to_grid_kwh", 0.0)
            chrono_result["battery_discharged_avg_price"] = elected_chrono_result.get("battery_discharged_avg_price", 0.0)
            # RTE loss tracking from elected chrono
            chrono_result["rte_loss_kwh"] = elected_chrono_result.get("rte_loss_kwh", 0.0)
            chrono_result["rte_loss_value"] = elected_chrono_result.get("rte_loss_value", 0.0)
            # RTE-aware discharge tracking from elected chrono
            chrono_result["rte_preserved_kwh"] = elected_chrono_result.get("rte_preserved_kwh", 0.0)
            chrono_result["rte_preserved_periods"] = elected_chrono_result.get("rte_preserved_periods", [])
            chrono_result["rte_breakeven_price"] = elected_chrono_result.get("rte_breakeven_price", 0.0)

            # CRITICAL: Update actual_discharge from re-run's feasible windows
            # The first run used ALL charge candidates, so some discharge windows may have been
            # feasible with that extra charge. With only ELECTED charge windows, those discharge
            # windows may no longer be feasible (battery not charged enough).
            elected_feasible_discharge = elected_chrono_result.get("feasible_discharge_windows", [])
            if len(elected_feasible_discharge) < len(actual_discharge):
                _LOGGER.info(
                    f"Re-run filtered discharge: {len(actual_discharge)} → {len(elected_feasible_discharge)} windows"
                )
                actual_discharge = elected_feasible_discharge
                # Rebuild grouped windows for dashboard
                grouped_discharge_windows = self._group_consecutive_windows(
                    actual_discharge, avg_cheap_buy, is_discharge=True
                )

            # Re-lookup current battery state from the new trajectory
            current_battery_state = buffer_energy
            sensor_value_obtained = False
            if not is_tomorrow and use_sensor and sensor_entity and hass:
                try:
                    sensor_state = hass.states.get(sensor_entity)
                    if sensor_state and sensor_state.state not in ("unknown", "unavailable", None):
                        # Show raw sensor value - min_usable threshold is applied in RTE calculations only
                        current_battery_state = float(sensor_state.state)
                        sensor_value_obtained = True
                except (ValueError, TypeError):
                    pass
            if not sensor_value_obtained:
                battery_trajectory = elected_chrono_result.get("battery_trajectory", [])
                if battery_trajectory and not is_tomorrow:
                    now = dt_util.now()
                    for entry in battery_trajectory:
                        entry_ts = entry["timestamp"]
                        if isinstance(entry_ts, str):
                            entry_ts = dt_util.parse_datetime(entry_ts)
                        if entry_ts and entry_ts <= now:
                            current_battery_state = entry.get("battery_after", current_battery_state)
                        else:
                            break

            final_battery_state = elected_chrono_result["final_battery_state"]
            buffer_delta = final_battery_state - buffer_energy
            _LOGGER.info(
                f"Re-ran chrono with {len(actual_charge)} elected windows: "
                f"battery_state_current={current_battery_state:.2f} kWh, "
                f"final={final_battery_state:.2f} kWh"
            )

        # Run FUTURE-ONLY projection starting from CURRENT battery state
        future_projection_applied = False
        future_total_cost = 0.0
        # Skip future projection for optimizer baseline (0/0 windows)
        # Baseline should simulate entire day with 0 windows, not mix actual past + simulated future
        skip_future_projection = config.get("_skip_future_projection", False)
        # Run future projection for today's sensor regardless of battery state
        # This ensures future costs use actual future prices, not averaged/incorrect values
        if not skip_future_projection and not is_tomorrow and use_ha_energy:
            # Build NEW timeline for FUTURE periods only
            # Must filter windows to only future ones, not just filter timeline entries
            now = dt_util.now()

            # Filter to only FUTURE charge/discharge windows (not yet completed)
            future_charge = [w for w in actual_charge if w["timestamp"] >= now]
            future_discharge = [w for w in actual_discharge if w["timestamp"] >= now]

            # Filter prices to only future periods
            future_prices = [p for p in all_prices if p["timestamp"] >= now]

            # Build fresh timeline with ONLY future windows and future prices
            future_timeline = self._build_chronological_timeline(
                future_prices,
                future_charge,
                future_discharge,
                config
            )

            _LOGGER.info(
                f"Future projection check: use_ha_energy={use_ha_energy}, "
                f"current_battery={current_battery_state:.2f}, buffer_energy={buffer_energy:.2f}, "
                f"future_periods={len(future_timeline)}, now={now}"
            )

            if not future_timeline:
                # No future periods left - EOD IS the current state
                # But we still need to estimate cost for remaining time in current hour
                hours_remaining = 24 - current_hour  # At hour 23: 1 hour remaining
                weighted_avg = energy_statistics.get("weighted_avg_consumption", 0.0) if energy_statistics else 0.0
                if weighted_avg > 0 and hours_remaining > 0:
                    # Get current price from all_prices (look up price for current time)
                    current_price = 0.0
                    for p in all_prices:
                        p_ts = p.get("timestamp")
                        if p_ts and p_ts <= now < p_ts + timedelta(hours=1):
                            current_price = p.get("price", 0.0)
                            break
                    if current_price > 0:
                        future_base_cost = weighted_avg * hours_remaining * current_price
                        future_total_cost = future_base_cost
                        _LOGGER.info(
                            f"No future periods - estimated remaining base cost: "
                            f"{hours_remaining}h × {weighted_avg:.3f}kWh × €{current_price:.3f} = €{future_base_cost:.3f}"
                        )
                    else:
                        future_total_cost = 0.0
                else:
                    future_total_cost = 0.0
                _LOGGER.info(
                    f"No future periods - EOD battery = current battery = {current_battery_state:.2f} kWh"
                )
                final_battery_state = current_battery_state
                chrono_result["final_battery_state"] = final_battery_state
                future_projection_applied = True
                buffer_delta = 0.0
            else:
                _LOGGER.info(
                    f"Future projection: Starting from current_battery_state={current_battery_state:.2f} kWh "
                    f"(midnight buffer was {buffer_energy:.2f} kWh), {len(future_timeline)} future periods"
                )

                # Run future-only chrono with current battery state as starting point
                future_chrono_result = self._simulate_chronological_costs(
                    future_timeline, config, current_battery_state, limit_discharge, is_tomorrow, hass,
                    energy_statistics=energy_statistics,
                    actual_battery_flows=actual_battery_flows if use_ha_energy else None
                )

                future_projection_applied = True
                future_total_cost = future_chrono_result.get("planned_total_cost", 0.0)

                # Update EOD battery state (projection from current, not midnight)
                final_battery_state = future_chrono_result["final_battery_state"]
                chrono_result["final_battery_state"] = final_battery_state

                # NOTE: Do NOT overwrite solar metrics here!
                # The chrono_result already contains correct completed+planned solar totals
                # from the full-day simulation. The future projection only updates
                # final_battery_state for EOD estimate.

                buffer_delta = final_battery_state - current_battery_state  # Delta from NOW, not midnight

                _LOGGER.info(
                    f"Future projection complete: EOD battery={final_battery_state:.2f} kWh, "
                    f"buffer_delta_from_now={buffer_delta:.2f} kWh, "
                    f"future_total_cost={future_total_cost:.3f}"
                )

        # ALWAYS recalculate metrics using chrono_result as single source of truth
        # The chrono simulation correctly tracks battery capacity and RTE
        # Initial calculations (before chrono) use bulk formulas that ignore capacity limits
        windows_were_filtered = skipped_charge or (limit_discharge and chrono_result.get('skipped_discharge_windows'))

        # Initialize actual_total_cost (calculated inside chrono_result block, used in result dict)
        actual_total_cost = None

        if chrono_result:  # Always use chrono_result when available
            # Use chrono_result as single source of truth for ALL kWh values
            # This ensures values respect battery capacity and RTE correctly
            gross_charged_kwh = chrono_result.get("actual_charge_kwh", 0)  # Actual grid draw (capped by battery)
            gross_usable_kwh = gross_charged_kwh  # For display purposes
            gross_discharged_kwh = chrono_result.get("actual_discharge_kwh", 0)  # Actual grid export
            actual_remaining_kwh = gross_usable_kwh - gross_discharged_kwh

            # Recalculate charge cost and discharge revenue for net metrics
            total_charge_cost = sum(w["price"] for w in actual_charge) * window_duration_hours * effective_charge_power if actual_charge else 0
            discharge_sell_prices_sum = 0
            for w in actual_discharge:
                if "sell_price" in w:
                    discharge_sell_prices_sum += w["sell_price"]
                else:
                    raw = price_lookup.get(w["timestamp"], {}).get("raw_price", w["price"])
                    discharge_sell_prices_sum += self._calculate_sell_price(raw, w["price"], sell_country, sell_param_a, sell_param_b)
            discharge_revenue = discharge_sell_prices_sum * window_duration_hours * effective_discharge_power

            # Recalculate remaining and margin metrics
            remaining_kwh = gross_usable_kwh - gross_discharged_kwh
            net_post_discharge_eur_kwh = ((total_charge_cost - discharge_revenue) / remaining_kwh) if remaining_kwh > 0 else 0
            battery_margin_eur_kwh = day_avg_price - net_post_discharge_eur_kwh
            battery_arbitrage_value = actual_remaining_kwh * battery_margin_eur_kwh if actual_remaining_kwh > 0 else 0

            # Recalculate net planned values from filtered windows (used by dashboard "Total Net")
            net_planned_charge_kwh = 0
            for w in actual_charge:
                duration_hours = w["duration"] / 60
                if charge_strategy == "battery_covers_base":
                    net_planned_charge_kwh += duration_hours * max(0, charge_power - base_usage)
                else:  # grid_covers_both
                    net_planned_charge_kwh += duration_hours * charge_power
            net_planned_charge_kwh = round(net_planned_charge_kwh, 3)

            net_planned_discharge_kwh = 0
            for w in actual_discharge:
                duration_hours = w["duration"] / 60
                if discharge_strategy == "already_included":
                    net_planned_discharge_kwh += duration_hours * discharge_power
                else:  # subtract_base
                    net_planned_discharge_kwh += duration_hours * max(0, discharge_power - base_usage)
            net_planned_discharge_kwh = round(net_planned_discharge_kwh, 3)

            # Recalculate dependent values
            # Note: RTE is NOT applied here - it only affects battery state simulation, not cost calculations
            usable_kwh = max(0, buffer_energy + net_planned_charge_kwh - net_planned_discharge_kwh)
            if limit_savings_enabled:
                effective_base_usage_kwh = min(base_usage_kwh, usable_kwh)

            # Use accurate kWh values from chronological simulation
            # (chrono_result tracks actual grid draw/export accounting for feasibility)
            charged_kwh = chrono_result.get("actual_charge_kwh", 0)  # Grid draw (no RTE)
            discharged_kwh = chrono_result.get("actual_discharge_kwh", 0)  # Grid export
            uncovered_base = chrono_result.get("uncovered_base_kwh", 0)  # Grid covered base when battery empty

            # net_grid_kwh = grid import - grid export (battery + solar)
            # grid_kwh_total already includes charge + uncovered base
            solar_exported = chrono_result.get("solar_exported_kwh", 0)
            net_grid_kwh = chrono_result.get("grid_kwh_total", 0) - discharged_kwh - solar_exported

            # For remaining calculations
            usable_kwh_calc = charged_kwh
            remaining_kwh = usable_kwh_calc - discharged_kwh

            # Recalculate planned costs from filtered windows
            # (planned_charge_cost, planned_discharge_revenue were calculated from SELECTED windows
            # but now actual_charge/actual_discharge are FILTERED to feasible windows)
            planned_charge_cost = 0
            for w in actual_charge:
                duration_hours = w["duration"] / 60
                if charge_strategy == "grid_covers_both":
                    planned_charge_cost += w["price"] * duration_hours * (charge_power + base_usage)
                else:  # battery_covers_base
                    planned_charge_cost += w["price"] * duration_hours * charge_power

            planned_discharge_revenue = 0
            for w in actual_discharge:
                duration_hours = w["duration"] / 60
                price_data = price_lookup.get(w["timestamp"], {})
                raw_price = price_data.get("raw_price", w["price"])
                sell_price = self._calculate_sell_price(
                    raw_price, w["price"], sell_country, sell_param_a, sell_param_b
                )
                if discharge_strategy == "already_included":
                    planned_discharge_revenue += sell_price * duration_hours * discharge_power
                else:  # subtract_base
                    net_export = max(0, discharge_power - base_usage)
                    planned_discharge_revenue += sell_price * duration_hours * net_export

            # Include solar benefits from chrono simulation
            # solar_export_revenue: Revenue from selling excess solar to grid
            # grid_savings_from_solar: Avoided grid costs from solar covering base usage
            solar_export_revenue = chrono_result.get("solar_export_revenue", 0.0)
            grid_savings_from_solar = chrono_result.get("grid_savings_from_solar", 0.0)

            # Calculate actual_total_cost from HA Energy data when available
            # This must be done BEFORE the future_projection_applied check which uses it
            actual_total_cost = None
            if use_ha_energy and grid_import_hourly:
                # Build hour-to-price lookup from price_lookup (keyed by timestamp)
                hour_prices = {}  # hour_int -> (buy_price, sell_price)
                for ts, price_data in price_lookup.items():
                    hour_int = ts.hour
                    buy_price = price_data.get("price", 0)
                    raw_price = price_data.get("raw_price", buy_price)
                    sell_price = self._calculate_sell_price(raw_price, buy_price, sell_country, sell_param_a, sell_param_b)
                    # Average if multiple periods per hour (15-min pricing)
                    if hour_int not in hour_prices:
                        hour_prices[hour_int] = {"buy": [], "sell": []}
                    hour_prices[hour_int]["buy"].append(buy_price)
                    hour_prices[hour_int]["sell"].append(sell_price)

                # Calculate actual cost from grid import/export × prices
                actual_total_cost = 0.0
                for hour_str, import_kwh in grid_import_hourly.items():
                    hour_int = int(hour_str)
                    if hour_int <= current_hour and hour_int in hour_prices:
                        avg_buy_price = sum(hour_prices[hour_int]["buy"]) / len(hour_prices[hour_int]["buy"])
                        actual_total_cost += import_kwh * avg_buy_price

                # Subtract export revenue
                if grid_export_hourly:
                    for hour_str, export_kwh in grid_export_hourly.items():
                        hour_int = int(hour_str)
                        if hour_int <= current_hour and hour_int in hour_prices:
                            avg_sell_price = sum(hour_prices[hour_int]["sell"]) / len(hour_prices[hour_int]["sell"])
                            actual_total_cost -= export_kwh * avg_sell_price

            if future_projection_applied:
                # Use actual cost from HA Energy (most accurate) when available
                if actual_total_cost is not None:
                    completed_total = actual_total_cost
                else:
                    # Fall back to calculated completed costs
                    completed_total = (
                        completed_charge_cost + completed_base_usage_cost
                        - completed_discharge_revenue - completed_solar_export_revenue
                        - completed_solar_grid_savings
                    )
                planned_total_cost = round(completed_total + future_total_cost, 3)
                _LOGGER.info(
                    f"Planned total cost from future projection: "
                    f"completed={completed_total:.3f} (actual={actual_total_cost is not None}) + "
                    f"future={future_total_cost:.3f} = {planned_total_cost:.3f}"
                )
            else:
                # Original calculation: estimate from windows (full day simulation)
                planned_total_cost = round(
                    planned_charge_cost + planned_base_usage_cost - planned_discharge_revenue
                    - solar_export_revenue - completed_solar_export_revenue - grid_savings_from_solar, 3
                )

            # Recalculate uncovered cost after chrono filtering (battery_covers_limited fallback to grid)
            if limit_savings_enabled and usable_kwh < base_usage_kwh and not future_projection_applied:
                uncovered_kwh = base_usage_kwh - usable_kwh
                uncovered_cost = uncovered_kwh * day_avg_price
                planned_total_cost = round(planned_total_cost + uncovered_cost, 3)

            # Recalculate savings (baseline_cost accounts for solar offsetting base usage)
            solar_offset_base_kwh = chrono_result.get("solar_offset_base_kwh", 0.0)
            solar_base_savings = solar_offset_base_kwh * day_avg_price
            baseline_cost = base_usage_kwh * day_avg_price - solar_base_savings
            estimated_savings = baseline_cost - planned_total_cost
            # True savings = savings on what we KEPT (base usage + end-of-day buffer)
            # Use final_battery_state from chrono (more accurate than actual_remaining_kwh)
            savings_per_kwh = day_avg_price - (planned_total_cost / net_grid_kwh) if net_grid_kwh > 0 else 0
            true_savings = savings_per_kwh * (base_usage_kwh + max(0, final_battery_state))

            # End-of-day buffer value (for dashboard display)
            # Method 2: EOD value = snapshot of TODAY's performance
            # Effective price = charge cost spread across usable energy (after RTE losses)
            if final_battery_state > 0:
                # Get charge cost and energy data from chrono
                chrono_charge_cost = chrono_result.get("planned_charge_cost", planned_charge_cost)
                # Use actual_charge_kwh from chrono (net_planned_charge_kwh is calculated separately)
                total_charged_kwh = chrono_result.get("actual_charge_kwh", 0)
                rte_loss_kwh = chrono_result.get("rte_loss_kwh", 0)

                # Usable kWh = what we charged minus TODAY's RTE losses
                usable_kwh = total_charged_kwh - rte_loss_kwh

                # Effective price = charge cost spread across usable energy
                if usable_kwh > 0:
                    effective_price_per_kwh = chrono_charge_cost / usable_kwh
                else:
                    effective_price_per_kwh = day_avg_price  # fallback if no charging

                # Margin = what we can sell at (day_avg) - what we paid (effective price)
                # Negative margin = we overpaid relative to average price
                battery_margin_eur_kwh = day_avg_price - effective_price_per_kwh

                # Value of remaining battery (can be negative if we overpaid)
                battery_state_end_of_day_value = final_battery_state * battery_margin_eur_kwh
            else:
                battery_state_end_of_day_value = 0

        # ALWAYS re-determine current state after chrono simulation to include:
        # 1. Filtered charge/discharge windows
        # 2. RTE-preserved periods (battery held due to price < breakeven)
        current_state = self._determine_current_state(
            prices,
            actual_charge,  # Use filtered charge windows
            actual_discharge,  # Use filtered discharge windows
            config,
            0.0,  # arbitrage_avg not needed for state check
            is_tomorrow,
            rte_preserved_periods=chrono_result.get("rte_preserved_periods", [])
        )

        # Calculate completed_net_grid_kwh using actual HA data when available
        if use_ha_energy and grid_import_hourly and grid_export_hourly:
            # Actual net grid from meter: import - export for completed hours
            completed_net_grid_kwh = (
                sum(v for h, v in grid_import_hourly.items() if h <= current_hour) -
                sum(v for h, v in grid_export_hourly.items() if h <= current_hour)
            )
            # When HA Energy is enabled, use actual completed + future estimate
            # instead of pure simulation (which underestimates evening hours)
            # hours_remaining includes remainder of current hour until midnight
            hours_remaining = 24 - current_hour  # At hour 23: 1 hour left until EOD
            # Use weighted 72h avg for future estimate (more stable than daily avg)
            weighted_avg = energy_statistics.get("weighted_avg_consumption", 0.0)
            if weighted_avg > 0:
                future_base_kwh = weighted_avg * hours_remaining
            else:
                # Fallback to real consumption avg if weighted not available
                future_base_kwh = energy_statistics.get("avg_real_consumption", base_usage) * hours_remaining
            # Override net_grid_kwh with actual + future estimate
            net_grid_kwh = completed_net_grid_kwh + future_base_kwh
        else:
            # Simulation mode: reconstruct from individual components
            completed_net_grid_kwh = (
                completed_charge_kwh + completed_base_grid_kwh -
                completed_discharge_kwh - completed_solar_export_kwh - completed_solar_base_kwh
            )

        # Build result
        result = {
            "state": current_state,
            "cheapest_times": [w["timestamp"].isoformat() for w in charge_windows],
            "cheapest_prices": [float(w["price"]) for w in charge_windows],
            "expensive_times": [w["timestamp"].isoformat() for w in discharge_windows],
            "expensive_prices": [float(w["price"]) for w in discharge_windows],
            "expensive_sell_prices": [float(get_sell_price_for_window(w)) for w in discharge_windows],
            "actual_charge_times": [w["timestamp"].isoformat() for w in actual_charge],
            "actual_charge_prices": [float(w["price"]) for w in actual_charge],
            "actual_discharge_times": [w["timestamp"].isoformat() for w in actual_discharge],
            "actual_discharge_prices": [float(get_sell_price_for_window(w)) for w in actual_discharge],  # Use sell prices (discharge = selling to grid)
            "actual_discharge_sell_prices": [float(get_sell_price_for_window(w)) for w in actual_discharge],
            "completed_charge_windows": completed_charge,
            "completed_discharge_windows": completed_discharge,
            "completed_charge_cost": round(completed_charge_cost, 3),
            "completed_discharge_revenue": round(completed_discharge_revenue, 3),
            "completed_solar_export_revenue": round(completed_solar_export_revenue, 3),
            "completed_base_usage_cost": round(completed_base_usage_cost, 3),
            "completed_base_usage_battery": round(completed_base_usage_battery, 3),
            # Completed kWh tracking (current net grid usage)
            "completed_charge_kwh": round(completed_charge_kwh, 3),
            "completed_discharge_kwh": round(completed_discharge_kwh, 3),
            "completed_base_grid_kwh": round(completed_base_grid_kwh, 3),
            "completed_solar_base_kwh": round(completed_solar_base_kwh, 3),
            "completed_solar_export_kwh": round(completed_solar_export_kwh, 3),
            "completed_net_grid_kwh": round(completed_net_grid_kwh, 3),
            "uncovered_base_usage_kwh": round(uncovered_kwh, 3),
            "uncovered_base_usage_cost": round(uncovered_cost, 3),
            # Debug: chrono simulation's planned_total_cost (before recalculation)
            "chrono_planned_total_cost": chrono_result.get("planned_total_cost", 0),
            # Debug: future projection tracking
            "_debug_future_projection_applied": future_projection_applied,
            "_debug_future_total_cost": round(future_total_cost, 3),
            "_debug_completed_total": round(completed_charge_cost + completed_base_usage_cost - completed_discharge_revenue - completed_solar_export_revenue - completed_solar_grid_savings, 3),
            "completed_solar_grid_savings": round(completed_solar_grid_savings, 3),
            "total_cost": round(
                actual_total_cost if actual_total_cost is not None
                else (completed_charge_cost + completed_base_usage_cost - completed_discharge_revenue - completed_solar_export_revenue - completed_solar_grid_savings),
                3
            ),
            "planned_total_cost": planned_total_cost,
            "planned_charge_cost": round(planned_charge_cost, 3),
            "planned_discharge_revenue": round(planned_discharge_revenue, 3),
            # Total value for maximize_value optimization strategy
            # total_value = savings + EOD battery value
            # savings = baseline_cost - planned_total_cost
            "total_value": round(
                (baseline_cost - planned_total_cost) + battery_state_end_of_day_value,
                4
            ),
            "net_planned_charge_kwh": net_planned_charge_kwh,
            "net_planned_discharge_kwh": net_planned_discharge_kwh,
            "effective_base_usage_kwh": effective_base_usage_kwh,
            "base_usage_kwh": round(base_usage_kwh, 3),
            # base_usage_day_cost: Use weighted 72h average when HA Energy enabled, manual otherwise
            "base_usage_day_cost": round(
                ((energy_statistics or {}).get("weighted_avg_consumption", 0.0) * 24 * day_avg_price)
                if (use_ha_energy and (energy_statistics or {}).get("weighted_avg_consumption", 0.0) > 0)
                else (base_usage_kwh * day_avg_price),
                2
            ),
            "num_windows": len(charge_windows),
            # Profit-based attributes
            "charge_profit_pct": round(charge_profit_pct, 1),  # Buy-buy profit for charging
            "discharge_profit_pct": round(discharge_profit_pct, 1),  # Buy-sell profit for discharge
            "charge_profit_met": bool(charge_profit_pct >= min_profit_charge),
            "discharge_profit_met": bool(discharge_profit_pct >= min_profit_discharge),
            "spread_avg": round(spread_avg, 1),  # Buy price spread (%)
            "arbitrage_avg": round(arbitrage_avg, 1),  # Sell price spread (%)
            "avg_cheap_price": round(avg_cheap, 5),
            "avg_expensive_price": round(avg_expensive, 5),
            "current_price": round(current_price, 5) if current_price else 0,
            "current_sell_price": round(current_sell_price, 5) if current_sell_price else 0,
            "price_country": sell_country,
            "buy_formula_param_a": config.get("buy_formula_param_a", DEFAULT_BUY_FORMULA_PARAM_A),
            "buy_formula_param_b": config.get("buy_formula_param_b", DEFAULT_BUY_FORMULA_PARAM_B),
            "sell_formula_param_a": sell_param_a,
            "sell_formula_param_b": sell_param_b,
            "price_override_active": config.get("price_override_enabled", False) and
                                    current_price and
                                    current_price <= config.get("price_override_threshold", 0.15),
            "time_override_active": config.get("time_override_enabled", False),
            "automation_enabled": config.get("automation_enabled", True),
            "calculation_window_enabled": config.get("calculation_window_enabled", False),
            # === DASHBOARD HELPER ATTRIBUTES ===
            # Grouped windows (reduces ~140 lines of Jinja2)
            "grouped_charge_windows": grouped_charge_windows,
            "grouped_discharge_windows": grouped_discharge_windows,
            # Percentile averages (reduces ~60 lines)
            "percentile_cheap_avg": round(percentile_cheap_avg, 5),
            "percentile_expensive_avg": round(percentile_expensive_avg, 5),
            "percentile_expensive_sell_avg": round(percentile_expensive_sell_avg, 5),
            # Half averages for fallback display
            "cheap_half_avg": round(cheap_half_avg, 5),
            "expensive_half_avg": round(expensive_half_avg, 5),
            # Sorted prices
            "sorted_buy_prices": [round(p, 5) for p in sorted_buy_prices],
            "sorted_sell_prices": [round(p, 5) for p in sorted_sell_prices],
            # Day average price
            "day_avg_price": round(day_avg_price, 5),
            # Estimated savings (reduces ~80 lines)
            "net_grid_kwh": round(net_grid_kwh, 3),
            "baseline_cost": round(baseline_cost, 3),
            "estimated_savings": round(estimated_savings, 3),
            "true_savings": round(true_savings, 3),
            # Battery metrics (reduces ~40 lines)
            "gross_charged_kwh": round(gross_charged_kwh, 3),
            "gross_usable_kwh": round(gross_usable_kwh, 3),
            "gross_discharged_kwh": round(gross_discharged_kwh, 3),
            "actual_remaining_kwh": round(actual_remaining_kwh, 3),
            "net_post_discharge_eur_kwh": round(net_post_discharge_eur_kwh, 5),
            "battery_margin_eur_kwh": round(battery_margin_eur_kwh, 5),
            "battery_arbitrage_value": round(battery_arbitrage_value, 3),
            # Estimated Savings presentation (clearer format)
            "actual_price_kwh": round(planned_total_cost / net_grid_kwh, 5) if net_grid_kwh > 0 else 0,
            "cost_difference": round(baseline_cost - planned_total_cost, 2),
            # Power in kW (reduces repetitive W->kW conversion)
            "charge_power_kw": round(charge_power, 3),
            "discharge_power_kw": round(discharge_power, 3),
            "base_usage_kw": round(base_usage, 3),
            # Window duration
            "window_duration_hours": window_duration_hours,
            # === CHRONOLOGICAL BUFFER TRACKING ===
            # Buffer energy tracking (start → end of day)
            "buffer_energy_kwh": round(buffer_energy, 3),
            "final_battery_state_kwh": round(final_battery_state, 3),
            "buffer_delta_kwh": round(buffer_delta, 3),
            "battery_capacity_kwh": round(battery_capacity, 3),
            # Chronological simulation results
            "chrono_charge_kwh": chrono_result["actual_charge_kwh"],
            "chrono_discharge_kwh": chrono_result["actual_discharge_kwh"],
            "chrono_uncovered_base_kwh": chrono_result["uncovered_base_kwh"],
            # Grid usage tracking (uses actual + future estimate when HA Energy enabled)
            "grid_kwh_estimated_today": round(net_grid_kwh, 3),
            "battery_state_current": round(current_battery_state, 3),
            "battery_state_end_of_day": chrono_result.get("final_battery_state", 0.0),
            "battery_state_end_of_day_value": round(battery_state_end_of_day_value, 3),
            # Discharge limiting (conservative mode)
            "limit_discharge_to_buffer": limit_discharge,
            "skipped_discharge_windows": chrono_result["skipped_discharge_windows"],
            "discharge_windows_limited": len(chrono_result["skipped_discharge_windows"]) > 0,
            # Charge limiting (battery full)
            "skipped_charge_windows": chrono_result.get("skipped_charge_windows", []),
            "charge_windows_limited": len(chrono_result.get("skipped_charge_windows", [])) > 0,
            # Feasibility tracking
            "feasibility_issues": chrono_result["feasibility_issues"],
            "has_feasibility_issues": len(chrono_result["feasibility_issues"]) > 0,
            # Solar integration metrics
            "solar_to_battery_kwh": chrono_result.get("solar_to_battery_kwh", 0.0),
            "solar_offset_base_kwh": chrono_result.get("solar_offset_base_kwh", 0.0),
            "solar_exported_kwh": chrono_result.get("solar_exported_kwh", 0.0),
            "solar_export_revenue": chrono_result.get("solar_export_revenue", 0.0),
            "solar_total_contribution_kwh": chrono_result.get("solar_total_contribution_kwh", 0.0),
            "grid_savings_from_solar": chrono_result.get("grid_savings_from_solar", 0.0),
            "expected_solar_kwh": chrono_result.get("expected_solar_kwh", 0.0),
            # Battery tracking from chrono simulation
            "battery_charged_from_grid_kwh": chrono_result.get("battery_charged_from_grid_kwh", 0.0),
            "battery_charged_from_grid_cost": chrono_result.get("battery_charged_from_grid_cost", 0.0),
            "battery_charged_from_solar_kwh": chrono_result.get("battery_charged_from_solar_kwh", 0.0),
            "battery_charged_avg_price": chrono_result.get("battery_charged_avg_price", 0.0),
            "battery_discharged_to_base_kwh": chrono_result.get("battery_discharged_to_base_kwh", 0.0),
            "battery_discharged_to_grid_kwh": chrono_result.get("battery_discharged_to_grid_kwh", 0.0),
            "battery_discharged_avg_price": chrono_result.get("battery_discharged_avg_price", 0.0),
            # Actual battery flows from HA Energy
            "actual_battery_charged_from_grid_kwh": round(actual_battery_flows["charged_from_grid_kwh"], 3),
            "actual_battery_charged_from_solar_kwh": round(actual_battery_flows["charged_from_solar_kwh"], 3),
            "actual_battery_charge_cost": round(actual_battery_flows["charged_from_grid_cost"], 3),
            "actual_battery_discharged_to_base_kwh": round(actual_battery_flows["discharged_to_base_kwh"], 3),
            "actual_battery_discharged_to_grid_kwh": round(actual_battery_flows["discharged_to_grid_kwh"], 3),
            "actual_battery_discharge_revenue": round(actual_battery_flows["discharged_revenue"], 3),
            # RTE (Round-Trip Efficiency) loss tracking
            # kWh: Only add completed when using sensor (chrono doesn't include completed windows)
            # When NOT using sensor, chrono already includes ALL RTE loss kWh (avoid double counting)
            # Value: Always add completed because chrono only tracks solar RTE (opportunity cost),
            # while completed tracks grid RTE (actual cost) - they measure different things
            "rte_loss_kwh": (completed_rte_loss_kwh if using_sensor_for_today else 0) + chrono_result.get("rte_loss_kwh", 0.0),
            "rte_loss_value": completed_rte_loss_value + chrono_result.get("rte_loss_value", 0.0),
            # RTE-aware discharge tracking
            "rte_preserved_kwh": chrono_result.get("rte_preserved_kwh", 0.0),
            "rte_preserved_periods": chrono_result.get("rte_preserved_periods", []),
            "rte_breakeven_price": chrono_result.get("rte_breakeven_price", 0.0),
            "rte_breakeven_source": chrono_result.get("rte_breakeven_source", "simulation"),
            "rte_solar_opportunity_price": chrono_result.get("rte_solar_opportunity_price", 0.0),
            "rte_actual_grid_charge_kwh": actual_battery_flows.get("charged_from_grid_kwh", 0.0) if use_ha_energy else 0.0,
            "rte_actual_grid_charge_cost": actual_battery_flows.get("charged_from_grid_cost", 0.0) if use_ha_energy else 0.0,
            # HA Energy Dashboard integration status
            "energy_stats_available": (energy_statistics or {}).get("stats_available", False),
            "energy_consumption_hours": len((energy_statistics or {}).get("consumption_hourly", {})),
            "energy_consumption_sensor": (energy_statistics or {}).get("consumption_sensor", "none"),
            "energy_consumption_source": (energy_statistics or {}).get("consumption_source", "manual"),
            "energy_avg_hourly_consumption": (energy_statistics or {}).get("avg_hourly_consumption", 0.0),
            "energy_solar_sensor": (energy_statistics or {}).get("solar_sensor", "none"),
            "energy_solar_source": (energy_statistics or {}).get("solar_source", "today"),
            "energy_avg_hourly_solar": (energy_statistics or {}).get("avg_hourly_solar", 0.0),
            "energy_total_solar_production_kwh": (energy_statistics or {}).get("total_solar_production_kwh", 0.0),
            "energy_solar_forecast_today": (energy_statistics or {}).get("solar_forecast_today"),
            "energy_solar_forecast_tomorrow": (energy_statistics or {}).get("solar_forecast_tomorrow"),
            "energy_solar_forecast_hourly_today": (energy_statistics or {}).get("solar_forecast_hourly_today", {}),
            "energy_solar_forecast_hourly_tomorrow": (energy_statistics or {}).get("solar_forecast_hourly_tomorrow", {}),
            "energy_battery_sensor": (energy_statistics or {}).get("battery_sensor", "none"),
            "energy_battery_charge_source": (energy_statistics or {}).get("battery_charge_source", "today"),
            "energy_avg_hourly_battery_charge": (energy_statistics or {}).get("avg_hourly_battery_charge", 0.0),
            "energy_total_battery_charge_kwh": (energy_statistics or {}).get("total_battery_charge_kwh", 0.0),
            "energy_battery_discharge_source": (energy_statistics or {}).get("battery_discharge_source", "today"),
            "energy_avg_hourly_battery_discharge": (energy_statistics or {}).get("avg_hourly_battery_discharge", 0.0),
            "energy_total_battery_discharge_kwh": (energy_statistics or {}).get("total_battery_discharge_kwh", 0.0),
            # Real consumption (formula result)
            "energy_real_consumption_hourly": (energy_statistics or {}).get("real_consumption_hourly", {}),
            "energy_avg_real_consumption": (energy_statistics or {}).get("avg_real_consumption", 0.0),
            # Grid import/export raw data
            "energy_grid_import_hourly": (energy_statistics or {}).get("grid_import_hourly", {}),
            "energy_grid_export_hourly": (energy_statistics or {}).get("grid_export_hourly", {}),
            "energy_avg_grid_import": (energy_statistics or {}).get("avg_grid_import", 0.0),
            "energy_avg_grid_export": (energy_statistics or {}).get("avg_grid_export", 0.0),
            # Discovered sensors
            "energy_sensors": (energy_statistics or {}).get("sensors", {}),
            "energy_hours_with_data": (energy_statistics or {}).get("hours_with_data", 0),
            # Weighted 72h consumption average (for stable baseline)
            "energy_today_avg_consumption": (energy_statistics or {}).get("today_avg_consumption", 0.0),
            "energy_today_hours": (energy_statistics or {}).get("today_hours_with_data", 0),
            "energy_yesterday_avg_consumption": (energy_statistics or {}).get("yesterday_avg_consumption", 0.0),
            "energy_day_before_avg_consumption": (energy_statistics or {}).get("day_before_avg_consumption", 0.0),
            "energy_weighted_avg_consumption": (energy_statistics or {}).get("weighted_avg_consumption", 0.0),
            "energy_weighted_source": (energy_statistics or {}).get("weighted_consumption_source", "manual"),
            # Energy consumption diagnostic tracking from simulation
            "energy_actual_kwh": chrono_result.get("energy_actual_kwh", 0.0),
            "energy_estimated_kwh": chrono_result.get("energy_estimated_kwh", 0.0),
            "energy_hours_with_actual": chrono_result.get("energy_hours_with_actual", 0),
            "energy_hours_with_estimate": chrono_result.get("energy_hours_with_estimate", 0),
            # Manual charging detection (charging outside CEW planned windows)
            "manual_charge_detected": manual_charging_info.get("manual_charge_detected", False),
            "manual_charge_hours": manual_charging_info.get("manual_charge_hours", []),
            "manual_charge_kwh": manual_charging_info.get("manual_charge_kwh", 0.0),
            "manual_charge_cost": manual_charging_info.get("manual_charge_cost", 0.0),
        }

        return result

    def _group_consecutive_windows(
        self,
        windows: List[Dict[str, Any]],
        ref_price: float = 0.0,
        is_discharge: bool = False
    ) -> List[Dict[str, Any]]:
        """Group consecutive time windows into contiguous blocks.

        Args:
            windows: List of window dicts with timestamp, price, duration, and optionally sell_price
            ref_price: Reference price for spread calculation (expensive avg for charge, cheap avg for discharge)
            is_discharge: If True, use sell_price for calculations

        Returns:
            List of grouped windows with start, end, prices, avg_price, spread_pct, kwh, cost/revenue
        """
        if not windows:
            return []

        # Sort by timestamp
        sorted_windows = sorted(windows, key=lambda x: x["timestamp"])

        groups = []
        current_group = None

        for window in sorted_windows:
            window_start = window["timestamp"]
            window_end = window_start + timedelta(minutes=window["duration"])
            price = window.get("sell_price", window["price"]) if is_discharge else window["price"]

            if current_group is None:
                # Start new group
                current_group = {
                    "start": window_start,
                    "end": window_end,
                    "prices": [price],
                    "duration": window["duration"]
                }
            elif window_start == current_group["end"]:
                # Consecutive window, extend group
                current_group["end"] = window_end
                current_group["prices"].append(price)
            else:
                # Gap found, finalize current group and start new one
                groups.append(self._finalize_group(current_group, ref_price, is_discharge))
                current_group = {
                    "start": window_start,
                    "end": window_end,
                    "prices": [price],
                    "duration": window["duration"]
                }

        # Don't forget the last group
        if current_group:
            groups.append(self._finalize_group(current_group, ref_price, is_discharge))

        return groups

    def _finalize_group(
        self,
        group: Dict[str, Any],
        ref_price: float,
        is_discharge: bool
    ) -> Dict[str, Any]:
        """Finalize a window group with calculated metrics."""
        prices = group["prices"]
        avg_price = float(np.mean(prices))
        window_duration = group["duration"] / 60  # Convert to hours
        num_windows = len(prices)
        kwh = window_duration * num_windows  # Will be multiplied by power in dashboard

        # Calculate spread percentage
        if is_discharge:
            # For discharge: (sell_avg - ref_buy) / ref_buy * 100
            spread_pct = ((avg_price - ref_price) / ref_price * 100) if ref_price > 0 else 0
        else:
            # For charge: (ref_expensive - buy_avg) / buy_avg * 100
            spread_pct = ((ref_price - avg_price) / avg_price * 100) if avg_price > 0 else 0

        return {
            "start": group["start"].isoformat(),
            "end": group["end"].isoformat(),
            "start_time": group["start"].strftime("%H:%M"),
            "end_time": group["end"].strftime("%H:%M"),
            "prices": [round(p, 5) for p in prices],
            "avg_price": round(avg_price, 5),
            "spread_pct": round(spread_pct, 1),
            "num_windows": num_windows,
            "duration_hours": round(window_duration * num_windows, 2),
        }

    def _empty_result(self, is_tomorrow: bool) -> Dict[str, Any]:
        """Return an empty result structure."""
        return {
            "state": STATE_OFF,
            "cheapest_times": [],
            "cheapest_prices": [],
            "expensive_times": [],
            "expensive_prices": [],
            "expensive_sell_prices": [],
            "actual_charge_times": [],
            "actual_charge_prices": [],
            "actual_discharge_times": [],
            "actual_discharge_prices": [],
            "actual_discharge_sell_prices": [],
            "completed_charge_windows": 0,
            "completed_discharge_windows": 0,
            "completed_charge_cost": 0,
            "completed_discharge_revenue": 0,
            "completed_solar_export_revenue": 0,
            "completed_base_usage_cost": 0,
            "completed_base_usage_battery": 0,
            "completed_charge_kwh": 0,
            "completed_discharge_kwh": 0,
            "completed_base_grid_kwh": 0,
            "completed_solar_base_kwh": 0,
            "completed_solar_export_kwh": 0,
            "completed_net_grid_kwh": 0,
            "uncovered_base_usage_kwh": 0,
            "uncovered_base_usage_cost": 0,
            "completed_solar_grid_savings": 0,
            "total_cost": 0,
            "planned_total_cost": 0,
            "planned_charge_cost": 0,
            "planned_discharge_revenue": 0,
            "total_value": 0,
            "net_planned_charge_kwh": 0,
            "net_planned_discharge_kwh": 0,
            "effective_base_usage_kwh": 0,
            "base_usage_kwh": 0,
            "base_usage_day_cost": 0,
            "num_windows": 0,
            # Profit-based attributes
            "charge_profit_pct": 0,
            "discharge_profit_pct": 0,
            "charge_profit_met": False,
            "discharge_profit_met": False,
            "spread_avg": 0,
            "arbitrage_avg": 0,
            "avg_cheap_price": 0,
            "avg_expensive_price": 0,
            "current_price": 0,
            "current_sell_price": 0,
            "price_country": DEFAULT_PRICE_COUNTRY,
            "buy_formula_param_a": DEFAULT_BUY_FORMULA_PARAM_A,
            "buy_formula_param_b": DEFAULT_BUY_FORMULA_PARAM_B,
            "sell_formula_param_a": DEFAULT_SELL_FORMULA_PARAM_A,
            "sell_formula_param_b": DEFAULT_SELL_FORMULA_PARAM_B,
            "price_override_active": False,
            "time_override_active": False,
            "automation_enabled": False,
            "calculation_window_enabled": False,
            # === DASHBOARD HELPER ATTRIBUTES ===
            "grouped_charge_windows": [],
            "grouped_discharge_windows": [],
            "percentile_cheap_avg": 0,
            "percentile_expensive_avg": 0,
            "percentile_expensive_sell_avg": 0,
            "cheap_half_avg": 0,
            "expensive_half_avg": 0,
            "sorted_buy_prices": [],
            "sorted_sell_prices": [],
            "day_avg_price": 0,
            "net_grid_kwh": 0,
            "baseline_cost": 0,
            "estimated_savings": 0,
            "true_savings": 0,
            "gross_charged_kwh": 0,
            "gross_usable_kwh": 0,
            "gross_discharged_kwh": 0,
            "actual_remaining_kwh": 0,
            "net_post_discharge_eur_kwh": 0,
            "battery_margin_eur_kwh": 0,
            "battery_arbitrage_value": 0,
            "actual_price_kwh": 0,
            "cost_difference": 0,
            "charge_power_kw": 0,
            "discharge_power_kw": 0,
            "base_usage_kw": 0,
            "window_duration_hours": 0.25,
            # === CHRONOLOGICAL BUFFER TRACKING ===
            "buffer_energy_kwh": 0,
            "final_battery_state_kwh": 0,
            "buffer_delta_kwh": 0,
            "battery_capacity_kwh": 100.0,
            "chrono_charge_kwh": 0,
            "chrono_discharge_kwh": 0,
            "chrono_uncovered_base_kwh": 0,
            "grid_kwh_estimated_today": 0,
            "battery_state_current": 0,
            "battery_state_end_of_day": 0,
            "battery_state_end_of_day_value": 0,
            "limit_discharge_to_buffer": False,
            "skipped_discharge_windows": [],
            "discharge_windows_limited": False,
            "feasibility_issues": [],
            "has_feasibility_issues": False,
            # Solar integration metrics
            "solar_to_battery_kwh": 0.0,
            "solar_offset_base_kwh": 0.0,
            "solar_exported_kwh": 0.0,
            "solar_export_revenue": 0.0,
            "solar_total_contribution_kwh": 0.0,
            "grid_savings_from_solar": 0.0,
            "expected_solar_kwh": 0.0,
            # Battery tracking
            "battery_charged_from_grid_kwh": 0.0,
            "battery_charged_from_grid_cost": 0.0,
            "battery_charged_from_solar_kwh": 0.0,
            "battery_charged_avg_price": 0.0,
            "battery_discharged_to_base_kwh": 0.0,
            "battery_discharged_to_grid_kwh": 0.0,
            "battery_discharged_avg_price": 0.0,
            # Actual battery flows (fallback - no HA Energy data)
            "actual_battery_charged_from_grid_kwh": 0.0,
            "actual_battery_charged_from_solar_kwh": 0.0,
            "actual_battery_charge_cost": 0.0,
            "actual_battery_discharged_to_base_kwh": 0.0,
            "actual_battery_discharged_to_grid_kwh": 0.0,
            "actual_battery_discharge_revenue": 0.0,
            # RTE loss tracking
            "rte_loss_kwh": 0.0,
            "rte_loss_value": 0.0,
            # HA Energy Dashboard integration status
            "energy_stats_available": False,
            "energy_consumption_hours": 0,
            "energy_consumption_sensor": "none",
            "energy_consumption_source": "manual",
            "energy_avg_hourly_consumption": 0.0,
            "energy_solar_sensor": "none",
            "energy_solar_source": "today",
            "energy_avg_hourly_solar": 0.0,
            "energy_total_solar_production_kwh": 0.0,
            "energy_solar_forecast_today": None,
            "energy_solar_forecast_tomorrow": None,
            "energy_solar_forecast_hourly_today": {},
            "energy_solar_forecast_hourly_tomorrow": {},
            "energy_battery_sensor": "none",
            "energy_battery_charge_source": "today",
            "energy_avg_hourly_battery_charge": 0.0,
            "energy_total_battery_charge_kwh": 0.0,
            "energy_battery_discharge_source": "today",
            "energy_avg_hourly_battery_discharge": 0.0,
            "energy_total_battery_discharge_kwh": 0.0,
            # Real consumption (formula result)
            "energy_real_consumption_hourly": {},
            "energy_avg_real_consumption": 0.0,
            # Grid import/export raw data
            "energy_grid_import_hourly": {},
            "energy_grid_export_hourly": {},
            "energy_avg_grid_import": 0.0,
            "energy_avg_grid_export": 0.0,
            # Discovered sensors
            "energy_sensors": {},
            "energy_hours_with_data": 0,
            # Energy consumption diagnostic tracking
            "energy_actual_kwh": 0.0,
            "energy_estimated_kwh": 0.0,
            "energy_hours_with_actual": 0,
            "energy_hours_with_estimate": 0,
        }