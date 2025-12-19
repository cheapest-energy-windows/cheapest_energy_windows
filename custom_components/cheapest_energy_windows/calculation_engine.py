"""Calculation engine for Cheapest Energy Windows."""
from __future__ import annotations

from datetime import datetime, timedelta
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
    MODE_DISCHARGE_AGGRESSIVE,
    MODE_IDLE,
    MODE_OFF,
    STATE_CHARGE,
    STATE_DISCHARGE,
    STATE_DISCHARGE_AGGRESSIVE,
    STATE_IDLE,
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
        hass: Any = None
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
        # Debug logging for calculation window
        _LOGGER.debug(f"=== CALCULATION ENGINE CALLED for {'tomorrow' if is_tomorrow else 'today'} ===")
        _LOGGER.debug(f"Config keys received: {list(config.keys())}")
        _LOGGER.debug(f"calculation_window_enabled in config: {config.get('calculation_window_enabled', 'NOT PRESENT')}")
        _LOGGER.debug(f"calculation_window_start: {config.get('calculation_window_start', 'NOT PRESENT')}")
        _LOGGER.debug(f"calculation_window_end: {config.get('calculation_window_end', 'NOT PRESENT')}")

        # Get configuration values
        pricing_mode = config.get("pricing_window_duration", PRICING_15_MINUTES)

        # Use tomorrow's config if applicable
        suffix = "_tomorrow" if is_tomorrow and config.get("tomorrow_settings_enabled", False) else ""

        num_charge_windows = int(config.get(f"charging_windows{suffix}", 4))
        num_discharge_windows = int(config.get(f"expensive_windows{suffix}", 4))
        percentile_threshold = config.get(f"percentile_threshold{suffix}", 25)
        # Profit thresholds (v1.2.0+): profit = spread - RTE_loss
        min_profit_charge = config.get(f"min_profit_charge{suffix}", 10)
        min_profit_discharge = config.get(f"min_profit_discharge{suffix}", 10)
        min_profit_discharge_aggressive = config.get(f"min_profit_discharge_aggressive{suffix}", 10)
        min_price_diff_enabled = config.get(f"min_price_diff_enabled{suffix}", True)
        min_price_diff = config.get(f"min_price_difference{suffix}", 0.05) if min_price_diff_enabled else float('-inf')
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
            _LOGGER.debug(f"Calculation window ENABLED: {calc_window_start} - {calc_window_end}, filtering {len(processed_prices)} prices for window selection")
            processed_prices = self._filter_prices_by_calculation_window(
                processed_prices,
                calc_window_start,
                calc_window_end
            )
            _LOGGER.debug(f"After calculation window filter: {len(processed_prices)} prices for window selection, {len(all_prices)} prices for base usage")
            if not processed_prices:
                _LOGGER.debug("No prices after calculation window filter")
                return self._empty_result(is_tomorrow)
        else:
            _LOGGER.debug("Calculation window disabled")

        # Calculate arbitrage_avg for profit display
        arbitrage_avg = self._calculate_arbitrage_avg(processed_prices, config, is_tomorrow)

        # Note: Arbitrage Protection removed in v1.2.0
        # Profit thresholds now naturally control window qualification
        # If profit is below threshold, windows won't be selected → system is idle

        # Pre-filter prices based on time override to prevent idle/off periods from being selected
        # This ensures that windows calculations respect time overrides from the start
        time_override_enabled = config.get(f"time_override_enabled{suffix}", False)
        prices_for_charge_calc = processed_prices
        prices_for_discharge_calc = processed_prices

        if time_override_enabled:
            override_mode = config.get(f"time_override_mode{suffix}", MODE_IDLE)

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
                _LOGGER.debug(f"Time override enabled: {override_start_str} - {override_end_str}, mode: {override_mode}")

                # For idle/off modes, exclude override periods from window calculations
                if override_mode in [MODE_IDLE, MODE_OFF]:
                    filtered_prices = []
                    for price_data in processed_prices:
                        if not self._is_in_time_range(price_data["timestamp"], override_start_str, override_end_str):
                            filtered_prices.append(price_data)
                    prices_for_charge_calc = filtered_prices
                    prices_for_discharge_calc = filtered_prices
                    _LOGGER.debug(f"Filtered {len(processed_prices)} prices to {len(filtered_prices)} after excluding {override_mode} periods")

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
                    _LOGGER.debug(f"Charge mode: {len(charge_override_prices)} prices for charging, {len(discharge_filtered)} for discharge")

                # For discharge modes, only discharge windows should be in override period
                elif override_mode in [MODE_DISCHARGE, MODE_DISCHARGE_AGGRESSIVE]:
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
                    _LOGGER.debug(f"Discharge mode: {len(charge_filtered)} prices for charging, {len(discharge_override_prices)} for discharge")

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
        _LOGGER.debug(
            f"Capacity-first selection: {len(charge_windows)} limited windows, "
            f"{len(all_charge_candidates)} total candidates passing profit threshold"
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

        # Apply same bypass logic to aggressive profit threshold
        effective_min_profit_aggressive = float('-inf') if bypass_spread else min_profit_discharge_aggressive

        aggressive_windows = self._find_aggressive_discharge_windows(
            prices_for_discharge_calc,  # Use filtered prices for consistency
            charge_windows,
            discharge_windows,
            num_discharge_windows,
            percentile_threshold,
            effective_min_profit_aggressive,
            rte_loss,
            effective_min_price_diff_discharge
        )

        # Apply minimum sell price filter to aggressive windows too
        aggressive_windows = self._filter_discharge_by_min_sell_price(
            aggressive_windows,
            processed_prices,
            config
        )

        # Debug output when calculation window is enabled
        if calc_window_enabled:
            charge_times = [w["timestamp"].strftime("%H:%M") for w in charge_windows]
            discharge_times = [w["timestamp"].strftime("%H:%M") for w in discharge_windows]
            _LOGGER.debug(f"After calculation window filter - Charge windows: {charge_times}, Discharge windows: {discharge_times}")

        # Calculate current state (pass arbitrage_avg and is_tomorrow for RTE protection)
        # Note: arbitrage_avg was already calculated earlier for the protection check
        current_state = self._determine_current_state(
            processed_prices,
            charge_windows,
            discharge_windows,
            aggressive_windows,
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
            aggressive_windows,
            current_state,
            config,
            is_tomorrow,
            hass,
            all_charge_candidates  # Pass all candidates for capacity-first selection
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

        _LOGGER.debug("="*60)
        _LOGGER.debug("PROCESS PRICES START")
        _LOGGER.debug(f"Raw prices type: {type(raw_prices)}")
        _LOGGER.debug(f"Raw prices length: {len(raw_prices) if hasattr(raw_prices, '__len__') else 'N/A'}")
        _LOGGER.debug(f"Pricing mode: {pricing_mode}")
        _LOGGER.debug(f"Buy price country: {buy_country}")
        _LOGGER.debug(f"Buy formula param A: {buy_param_a}")
        _LOGGER.debug(f"Buy formula param B: {buy_param_b}")
        _LOGGER.debug(f"VAT: {vat}%")
        _LOGGER.debug(f"Tax: {tax} EUR/kWh")
        _LOGGER.debug(f"Additional cost: {additional_cost} EUR/kWh")

        if raw_prices and len(raw_prices) > 0:
            _LOGGER.debug(f"First item type: {type(raw_prices[0])}")
            _LOGGER.debug(f"First item: {raw_prices[0]}")
            if len(raw_prices) > 1:
                _LOGGER.debug(f"Second item: {raw_prices[1]}")

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

        _LOGGER.debug(f"Processed {len(processed)} price entries")
        if processed:
            _LOGGER.debug(f"First processed price: {processed[0]}")
            _LOGGER.debug(f"Last processed price: {processed[-1]}")
        _LOGGER.debug("PROCESS PRICES END")
        _LOGGER.debug("="*60)

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

            _LOGGER.debug(f"Calculation window filter: {len(prices)} -> {len(filtered)} prices (window: {start_str} to {end_str})")

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

        v1.2.0: Uses profit-based threshold (profit = spread - RTE_loss)
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
            # RTE does NOT affect window selection - only affects buffer size
            if candidate["price"] > 0:
                spread_pct = ((expensive_avg - candidate["price"]) / candidate["price"]) * 100
                profit_pct = spread_pct  # RTE removed - does not affect window selection
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

        v1.2.0: Uses profit-based threshold (profit = spread - RTE_loss)
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

            # Calculate spread and profit: (avg_sell - avg_buy) / avg_buy * 100
            # RTE does NOT affect window selection - only affects buffer size
            # Note: min_price_diff only applies to charge windows (buy-buy comparison)
            # For discharge, profit threshold is sufficient
            if cheap_avg > 0:
                spread_pct = ((expensive_avg - cheap_avg) / cheap_avg) * 100
                profit_pct = spread_pct  # RTE removed - does not affect window selection

                if profit_pct >= min_profit:
                    selected.append(candidate)

        return selected

    def _find_aggressive_discharge_windows(
        self,
        prices: List[Dict[str, Any]],
        charge_windows: List[Dict[str, Any]],
        discharge_windows: List[Dict[str, Any]],
        num_windows: int,
        percentile_threshold: float,
        min_profit: float,
        rte_loss: float,
        min_price_diff: float
    ) -> List[Dict[str, Any]]:
        """Find windows for aggressive discharge (peak SELL prices).

        Filters discharge windows by aggressive profit requirement.
        Uses SELL prices for spread comparison (discharge = selling to grid).
        Per-window price_diff check: window["sell_price"] - cheap_min >= threshold

        v1.2.0: Uses profit-based threshold (profit = spread - RTE_loss)
        - Higher threshold for aggressive discharge (e.g., drain battery to lower SOC)
        """
        if not prices or num_windows <= 0:
            return []

        # Use discharge windows as base, filter by aggressive profit threshold
        candidates = []

        if charge_windows:
            cheap_avg = np.mean([w["price"] for w in charge_windows])  # Buy prices
            cheap_min = min(w["price"] for w in charge_windows)
        else:
            # No charge windows - use bottom percentile_threshold% buy prices as reference
            price_array = np.array([p["price"] for p in prices])
            cheap_threshold = np.percentile(price_array, percentile_threshold)
            cheap_prices = price_array[price_array <= cheap_threshold]
            cheap_avg = np.mean(cheap_prices) if len(cheap_prices) > 0 else np.min(price_array)
            cheap_min = np.min(cheap_prices) if len(cheap_prices) > 0 else np.min(price_array)

        for window in discharge_windows:
            if cheap_avg > 0:
                # Use SELL price for spread/profit calculation (discharge = selling)
                # RTE does NOT affect window selection - only affects buffer size
                sell_price = window.get("sell_price", window["price"])
                spread_pct = ((sell_price - cheap_avg) / cheap_avg) * 100
                profit_pct = spread_pct  # RTE removed - does not affect window selection

                if profit_pct >= min_profit:
                    candidates.append(window)

        return candidates

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
        aggressive_windows: List[Dict[str, Any]],
        config: Dict[str, Any],
        arbitrage_avg: float = 0.0,
        is_tomorrow: bool = False
    ) -> str:
        """Determine current state based on time and configuration.

        v1.2.0: Arbitrage Protection removed. Profit thresholds now naturally
        control window qualification - if profit is below threshold, no windows
        are selected and system defaults to idle.
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
            mode = config.get("time_override_mode", MODE_IDLE)

            if self._is_in_time_range(current_time, start_str, end_str):
                return self._mode_to_state(mode)

        # Check price override
        if config.get("price_override_enabled", False):
            threshold = config.get("price_override_threshold", 0.15)
            current_price = self._get_current_price(prices, current_time)
            if current_price and current_price <= threshold:
                return STATE_CHARGE

        # Check scheduled windows
        for window in aggressive_windows:
            if self._is_window_active(window, current_time):
                return STATE_DISCHARGE_AGGRESSIVE

        for window in discharge_windows:
            if self._is_window_active(window, current_time):
                return STATE_DISCHARGE

        for window in charge_windows:
            if self._is_window_active(window, current_time):
                return STATE_CHARGE

        return STATE_IDLE

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
            MODE_IDLE: STATE_IDLE,
            MODE_CHARGE: STATE_CHARGE,
            MODE_DISCHARGE: STATE_DISCHARGE,
            MODE_DISCHARGE_AGGRESSIVE: STATE_DISCHARGE_AGGRESSIVE,
            MODE_OFF: STATE_OFF,
        }
        return mode_map.get(mode, STATE_IDLE)

    def _calculate_actual_windows(
        self,
        prices: List[Dict[str, Any]],
        charge_windows: List[Dict[str, Any]],
        discharge_windows: List[Dict[str, Any]],
        aggressive_windows: List[Dict[str, Any]],
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
            aggressive_windows: Calculated aggressive discharge windows
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
            # No overrides, return calculated windows as-is (don't combine normal + aggressive)
            return list(charge_windows), list(discharge_windows)

        # Get override configuration (using suffix for tomorrow settings)
        # Get time values and ensure they're in string format
        override_start = config.get(f"time_override_start{suffix}", "")
        override_end = config.get(f"time_override_end{suffix}", "")
        override_mode = config.get(f"time_override_mode{suffix}", MODE_IDLE)

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
            state = STATE_IDLE  # Default

            # Check time override first (highest priority)
            if time_override_enabled and self._is_in_time_range(timestamp, override_start_str, override_end_str):
                state = self._mode_to_state(override_mode)
            # Check price override
            elif price_override_enabled and price <= price_override_threshold:
                state = STATE_CHARGE
            else:
                # Check calculated windows
                for window in aggressive_windows:
                    if self._is_window_active(window, timestamp):
                        state = STATE_DISCHARGE_AGGRESSIVE
                        break

                if state == STATE_IDLE:
                    for window in discharge_windows:
                        if self._is_window_active(window, timestamp):
                            state = STATE_DISCHARGE
                            break

                if state == STATE_IDLE:
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
        new_actual_discharge = [w for w in timeline if w["state"] in [STATE_DISCHARGE, STATE_DISCHARGE_AGGRESSIVE]]

        return new_actual_charge, new_actual_discharge

    def _get_buffer_energy(self, config: Dict[str, Any], is_tomorrow: bool, hass: Any = None) -> float:
        """Get battery starting state for the calculation period.

        Priority (when use_battery_buffer_sensor is ON):
        1. Battery available energy sensor (if configured and valid)
        2. Fall back to manual buffer_kwh value

        Priority (when use_battery_buffer_sensor is OFF):
        1. Manual buffer_kwh value (per-day: today or tomorrow)

        Args:
            config: Configuration dictionary
            is_tomorrow: Whether calculating for tomorrow
            hass: Home Assistant instance (for sensor value lookup)

        Returns:
            Buffer energy in kWh
        """
        use_sensor = config.get("use_battery_buffer_sensor", False)
        sensor_entity = config.get("battery_available_energy_sensor", "")

        # Get per-day buffer value (respects tomorrow_settings_enabled)
        suffix = "_tomorrow" if is_tomorrow and config.get("tomorrow_settings_enabled", False) else ""
        buffer_kwh = config.get(f"battery_buffer_kwh{suffix}", 0.0)

        # If using sensor and we have a valid sensor entity
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
        aggressive_windows: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build time-ordered list of all periods with their window types.

        Args:
            prices: List of all price periods
            charge_windows: Selected charge windows
            discharge_windows: Selected discharge windows
            aggressive_windows: Selected aggressive discharge windows
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
        aggressive_timestamps = {w["timestamp"] for w in aggressive_windows}

        timeline = []
        for price_data in sorted(prices, key=lambda x: x["timestamp"]):
            ts = price_data["timestamp"]

            if ts in charge_timestamps:
                window_type = "charge"
            elif ts in aggressive_timestamps:
                window_type = "aggressive"
            elif ts in discharge_timestamps:
                window_type = "discharge"
            else:
                window_type = "idle"

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

    def _simulate_chronological_costs(
        self,
        timeline: List[Dict[str, Any]],
        config: Dict[str, Any],
        buffer_energy: float,
        limit_discharge: bool = False
    ) -> Dict[str, Any]:
        """Simulate battery state and calculate costs chronologically.

        Args:
            timeline: Time-ordered list of periods with window types
            config: Configuration dictionary
            buffer_energy: Starting battery energy in kWh
            limit_discharge: If True, skip discharge windows that can't be supported

        Returns:
            Dictionary with calculated costs, battery trajectory, and feasibility info
        """
        # Get configuration
        battery_capacity = config.get("battery_capacity", 100.0)  # kWh
        charge_power = config.get("charge_power", 0) / 1000  # W to kW
        discharge_power = config.get("discharge_power", 0) / 1000
        base_usage = config.get("base_usage", 0) / 1000
        battery_rte = config.get("battery_rte", 85) / 100

        # Strategies
        charge_strategy = config.get("base_usage_charge_strategy", "grid_covers_both")
        discharge_strategy = config.get("base_usage_discharge_strategy", "subtract_base")
        idle_strategy = config.get("base_usage_idle_strategy", "grid_covers")
        aggressive_strategy = config.get("base_usage_aggressive_strategy", "same_as_discharge")

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
        feasible_aggressive_windows = []
        feasible_charge_windows = []
        skipped_charge_windows = []

        # Grid usage tracking
        grid_kwh_total = 0.0  # Cumulative grid kWh (charge + uncovered base)

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

            if window_type == "charge":
                # Calculate how much we CAN charge (limited by capacity)
                desired_charge = charge_power * duration_hours
                base_demand = base_usage * duration_hours
                available_capacity = max(0, battery_capacity - battery_state)  # Usable kWh space left
                # RTE does NOT affect charging - only affects buffer size display
                max_grid_draw = available_capacity  # No RTE division - grid draw = stored energy
                actual_charge = min(desired_charge, max_grid_draw)  # Grid draw in kWh

                # Check if charge window is feasible (can charge at least 10% of desired)
                # Skip if battery is too full to charge meaningfully
                if actual_charge >= desired_charge * 0.1:
                    feasible_charge_windows.append(period)

                    # RTE DOES affect battery state - physical reality of charging losses
                    # actual_charge = grid draw (for costs), usable_charge = what battery stores
                    usable_charge = actual_charge * battery_rte  # Battery stores less than grid draw

                    if charge_strategy == "grid_covers_both":
                        # Grid provides both charge power AND base usage
                        # Battery only gains the usable charge
                        battery_state += usable_charge
                        planned_charge_cost += price * (actual_charge + base_demand)
                        grid_kwh_total += actual_charge + base_demand
                    else:  # battery_covers_base
                        # Grid provides charge power, battery covers base usage
                        # Net battery change = usable charge - base usage
                        net_battery_change = usable_charge - base_demand
                        battery_state += net_battery_change
                        planned_charge_cost += price * actual_charge
                        grid_kwh_total += actual_charge
                        # If battery can't cover base (goes negative), grid must cover the shortfall
                        if battery_state < 0:
                            shortfall = -battery_state
                            planned_base_usage_cost += price * shortfall
                            grid_kwh_total += shortfall
                            uncovered_base_kwh += shortfall
                            battery_state = 0

                    battery_state = min(max(0, battery_state), battery_capacity)  # Clamp to valid range
                    actual_charge_kwh += actual_charge

                    _LOGGER.debug(
                        f"CHARGE @ {period['timestamp']}: grid_draw={actual_charge:.2f} kWh, "
                        f"stored={usable_charge:.2f} kWh, "
                        f"capacity_left={available_capacity:.2f} kWh → battery={battery_state:.2f} kWh"
                    )
                else:
                    # Battery too full - skip this charge window
                    skipped_charge_windows.append({
                        "timestamp": period["timestamp"].isoformat() if hasattr(period["timestamp"], 'isoformat') else str(period["timestamp"]),
                        "price": price,
                        "reason": f"Battery nearly full ({battery_state:.2f}/{battery_capacity:.2f} kWh, only {available_capacity:.2f} kWh space)"
                    })
                    _LOGGER.debug(
                        f"CHARGE SKIPPED @ {period['timestamp']}: battery={battery_state:.2f} kWh, "
                        f"capacity={battery_capacity:.2f} kWh, space={available_capacity:.2f} kWh"
                    )

            elif window_type in ("discharge", "aggressive"):
                # Calculate how much we CAN discharge
                # During discharge, battery provides BOTH grid export AND base usage
                desired_discharge = discharge_power * duration_hours
                base_demand = base_usage * duration_hours
                total_drain_needed = desired_discharge + base_demand  # Battery provides both

                if limit_discharge:
                    # Conservative mode: buffer is the STARTING energy, not a floor
                    # All current battery energy is available for discharge (down to 0)
                    available_for_discharge = max(0, battery_state)
                    # Can we provide both discharge AND base usage?
                    if available_for_discharge >= total_drain_needed:
                        actual_discharge = desired_discharge
                        actual_base_from_battery = base_demand
                    else:
                        # Limited availability - prioritize base usage, rest to discharge
                        actual_base_from_battery = min(base_demand, available_for_discharge)
                        actual_discharge = min(desired_discharge, available_for_discharge - actual_base_from_battery)
                else:
                    # Optimistic mode: can discharge all the way to 0
                    if battery_state >= total_drain_needed:
                        actual_discharge = desired_discharge
                        actual_base_from_battery = base_demand
                    else:
                        # Limited - prioritize base usage
                        actual_base_from_battery = min(base_demand, battery_state)
                        actual_discharge = min(desired_discharge, battery_state - actual_base_from_battery)

                total_actual_drain = actual_discharge + actual_base_from_battery

                # Determine which strategy to use
                if window_type == "aggressive":
                    strategy = aggressive_strategy if aggressive_strategy != "same_as_discharge" else discharge_strategy
                else:
                    strategy = discharge_strategy

                if limit_discharge:
                    # Conservative mode: only include if we can do FULL discharge + base
                    # 100% threshold - skip if battery can't support full operation
                    _LOGGER.debug(
                        f"Conservative check @ {period['timestamp']}: "
                        f"battery={battery_state:.2f}, needed={total_drain_needed:.2f}, "
                        f"available={available_for_discharge:.2f}, desired={desired_discharge:.2f}, "
                        f"actual={actual_discharge:.2f}"
                    )
                    if battery_state >= total_drain_needed:
                        if window_type == "discharge":
                            feasible_discharge_windows.append(period)
                        else:
                            feasible_aggressive_windows.append(period)
                        battery_state -= total_actual_drain
                        actual_discharge_kwh += actual_discharge

                        # Calculate revenue based on strategy
                        if strategy == "already_included":
                            # User configured discharge as gross (includes base in their mind)
                            planned_discharge_revenue += sell_price * actual_discharge
                        else:  # subtract_base
                            # Net export = discharge only (base was for house, not grid)
                            planned_discharge_revenue += sell_price * actual_discharge
                    else:
                        _LOGGER.debug(f"  -> SKIPPED: battery={battery_state:.2f} < needed={total_drain_needed:.2f}")
                        skipped_discharge_windows.append({
                            "timestamp": period["timestamp"].isoformat() if hasattr(period["timestamp"], 'isoformat') else str(period["timestamp"]),
                            "price": price,
                            "sell_price": sell_price,
                            "reason": f"Insufficient battery ({battery_state:.2f} kWh < {total_drain_needed:.2f} kWh needed for discharge + base)"
                        })
                        # Even if discharge window is skipped, base usage still drains battery
                        battery_state -= actual_base_from_battery
                        if actual_base_from_battery < base_demand:
                            # Grid fallback: battery can't cover all base usage
                            uncovered = base_demand - actual_base_from_battery
                            planned_base_usage_cost += price * uncovered
                            grid_kwh_total += uncovered
                            uncovered_base_kwh += uncovered
                else:
                    # Optimistic mode: include all, track what we can actually discharge
                    if actual_discharge < desired_discharge * 0.5:
                        feasibility_issues.append(
                            f"{period['timestamp']}: Discharge limited by battery state "
                            f"({actual_discharge:.2f}/{desired_discharge:.2f} kWh)"
                        )

                    battery_state -= total_actual_drain
                    actual_discharge_kwh += actual_discharge

                    # Calculate revenue based on strategy
                    if strategy == "already_included":
                        planned_discharge_revenue += sell_price * actual_discharge
                    else:  # subtract_base
                        planned_discharge_revenue += sell_price * actual_discharge

                battery_state = max(0, battery_state)  # Ensure non-negative

            else:  # idle
                base_demand = base_usage * duration_hours

                if idle_strategy == "grid_covers":
                    # Grid provides all base usage
                    planned_base_usage_cost += price * base_demand
                    grid_kwh_total += base_demand
                elif idle_strategy in ("battery_covers", "battery_covers_limited"):
                    # Battery provides base usage (drain battery)
                    battery_drain = min(base_demand, battery_state)
                    battery_state -= battery_drain
                    uncovered = base_demand - battery_drain
                    if uncovered > 0:
                        # Grid fallback: battery empty, grid must cover the rest
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
            planned_charge_cost + planned_base_usage_cost - planned_discharge_revenue, 3
        )

        _LOGGER.info(
            f"Simulation complete: charged={actual_charge_kwh:.2f} kWh, "
            f"discharged={actual_discharge_kwh:.2f} kWh, final_battery={battery_state:.2f} kWh, "
            f"feasible_charge={len(feasible_charge_windows)}, skipped_charge={len(skipped_charge_windows)}, "
            f"feasible_discharge={len(feasible_discharge_windows)}, "
            f"feasible_aggressive={len(feasible_aggressive_windows)}, "
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
            "feasible_aggressive_windows": feasible_aggressive_windows,
            "feasible_charge_windows": feasible_charge_windows,
            "skipped_charge_windows": skipped_charge_windows,
        }

    def _build_result(
        self,
        prices: List[Dict[str, Any]],
        all_prices: List[Dict[str, Any]],
        charge_windows: List[Dict[str, Any]],
        discharge_windows: List[Dict[str, Any]],
        aggressive_windows: List[Dict[str, Any]],
        current_state: str,
        config: Dict[str, Any],
        is_tomorrow: bool,
        hass: Any = None,
        all_charge_candidates: List[Dict[str, Any]] = None
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
            aggressive_windows,
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
        base_usage = config.get("base_usage", 0) / 1000

        # Get strategies
        charge_strategy = config.get("base_usage_charge_strategy", "grid_covers_both")
        idle_strategy = config.get("base_usage_idle_strategy", "grid_covers")
        discharge_strategy = config.get("base_usage_discharge_strategy", "subtract_base")
        aggressive_strategy = config.get("base_usage_aggressive_strategy", "same_as_discharge")

        # Get sell price config for revenue calculations
        sell_country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
        sell_param_a = config.get("sell_formula_param_a", DEFAULT_SELL_FORMULA_PARAM_A)
        sell_param_b = config.get("sell_formula_param_b", DEFAULT_SELL_FORMULA_PARAM_B)

        # Build price lookup for raw prices
        price_lookup = {p["timestamp"]: p for p in prices}

        # Initialize tracking variables
        completed_charge_cost = 0
        completed_discharge_revenue = 0
        completed_base_usage_cost = 0  # Grid cost for base usage
        completed_base_usage_battery = 0  # Battery kWh used for base usage

        # CHARGE windows: Apply charge strategy
        for w in actual_charge:
            if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time:
                duration_hours = w["duration"] / 60
                if charge_strategy == "grid_covers_both":
                    # Grid provides charge power + base usage
                    completed_charge_cost += w["price"] * duration_hours * (charge_power + base_usage)
                else:  # battery_covers_base
                    # Grid provides charge power only, battery covers base
                    completed_charge_cost += w["price"] * duration_hours * charge_power
                    completed_base_usage_battery += duration_hours * base_usage

        # DISCHARGE/AGGRESSIVE windows: Apply discharge/aggressive strategies
        # Separate by state for strategy application
        for w in actual_discharge:
            if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time:
                duration_hours = w["duration"] / 60

                # Calculate sell price for this window
                price_data = price_lookup.get(w["timestamp"], {})
                raw_price = price_data.get("raw_price", w["price"])
                sell_price = self._calculate_sell_price(
                    raw_price, w["price"], sell_country, sell_param_a, sell_param_b
                )

                # Determine which strategy to use based on window state
                if w.get("state") == STATE_DISCHARGE_AGGRESSIVE:
                    # Aggressive discharge window
                    if aggressive_strategy == "same_as_discharge":
                        strategy = discharge_strategy
                    else:
                        strategy = aggressive_strategy
                else:
                    # Regular discharge window
                    strategy = discharge_strategy

                if strategy == "already_included":
                    # Full discharge power generates revenue at SELL price
                    completed_discharge_revenue += sell_price * duration_hours * discharge_power
                else:  # subtract_base (NoM)
                    # Battery covers base first, exports the rest at SELL price
                    net_export = max(0, discharge_power - base_usage)
                    completed_discharge_revenue += sell_price * duration_hours * net_export
                    completed_base_usage_battery += duration_hours * base_usage

        # IDLE periods: Apply idle strategy
        # Build sets of timestamps for active windows
        charge_timestamps = {w["timestamp"] for w in actual_charge}
        discharge_timestamps = {w["timestamp"] for w in actual_discharge}

        # Use all_prices (not filtered by calculation window) for base usage calculations
        # Base usage should cover full 24h regardless of calculation window
        for price_data in all_prices:
            timestamp = price_data["timestamp"]
            if timestamp + timedelta(minutes=price_data["duration"]) <= current_time:
                # Check if this period is idle (not in any active window)
                is_active = timestamp in charge_timestamps or timestamp in discharge_timestamps

                if not is_active:
                    duration_hours = price_data["duration"] / 60
                    if idle_strategy == "grid_covers":
                        # Grid provides base usage, add to cost
                        completed_base_usage_cost += price_data["price"] * duration_hours * base_usage
                    else:  # battery_covers (NoM)
                        # Battery provides base usage, track battery consumption
                        completed_base_usage_battery += duration_hours * base_usage

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

            # Determine strategy based on state
            if w.get("state") == STATE_DISCHARGE_AGGRESSIVE:
                strategy = aggressive_strategy if aggressive_strategy != "same_as_discharge" else discharge_strategy
            else:
                strategy = discharge_strategy

            if strategy == "already_included":
                planned_discharge_revenue += sell_price * duration_hours * discharge_power
            else:  # subtract_base
                net_export = max(0, discharge_power - base_usage)
                planned_discharge_revenue += sell_price * duration_hours * net_export

        # All idle periods (use all_prices for full 24h coverage)
        for price_data in all_prices:
            timestamp = price_data["timestamp"]
            is_active = timestamp in charge_timestamps or timestamp in discharge_timestamps

            if not is_active:
                duration_hours = price_data["duration"] / 60
                if idle_strategy == "grid_covers":
                    planned_base_usage_cost += price_data["price"] * duration_hours * base_usage

        planned_total_cost = round(planned_charge_cost + planned_base_usage_cost - planned_discharge_revenue, 3)

        # Calculate effective base usage (limited by charged energy when idle strategy is "battery_covers_limited")
        # This is used by the dashboard for Estimated Savings calculation
        limit_savings_enabled = (idle_strategy == "battery_covers_limited")
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
            # Determine which strategy to use based on window state
            if w.get("state") == STATE_DISCHARGE_AGGRESSIVE:
                strategy = aggressive_strategy if aggressive_strategy != "same_as_discharge" else discharge_strategy
            else:
                strategy = discharge_strategy

            if strategy == "already_included":
                # Full discharge power goes to grid
                net_planned_discharge_kwh += duration_hours * discharge_power
            else:  # subtract_base
                # Only net export goes to grid (discharge - base_usage)
                net_planned_discharge_kwh += duration_hours * max(0, discharge_power - base_usage)
        net_planned_discharge_kwh = round(net_planned_discharge_kwh, 3)

        # Calculate effective base usage (limit to usable energy if toggle is on)
        battery_rte_pct = config.get("battery_rte", 85)
        battery_rte_decimal = battery_rte_pct / 100
        # usable_kwh = energy available for base usage = charged - discharged
        # Note: RTE is NOT applied here - it only affects battery state simulation, not cost calculations
        usable_kwh = max(0, net_planned_charge_kwh - net_planned_discharge_kwh)
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

        # Calculate profit percentages (v1.2.0+)
        # profit = spread - RTE_loss
        battery_rte = config.get("battery_rte", 85)
        rte_loss = 100 - battery_rte
        charge_profit_pct = spread_avg - rte_loss  # Buy-buy spread for charging
        discharge_profit_pct = arbitrage_avg - rte_loss  # Buy-sell spread for discharge

        # Get profit thresholds from config
        min_profit_charge = config.get(f"min_profit_charge{suffix}", 10)
        min_profit_discharge = config.get(f"min_profit_discharge{suffix}", 10)
        min_profit_discharge_aggressive = config.get(f"min_profit_discharge_aggressive{suffix}", 10)

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
        usable_kwh = charged_kwh  # No RTE penalty on cost calculations
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

        # True savings (proportional when net_grid covers base, otherwise all savings apply)
        if net_grid_kwh >= base_usage_kwh and net_grid_kwh > 0:
            true_savings = (estimated_savings / net_grid_kwh) * base_usage_kwh
        else:
            # Net grid is less than base usage (or negative) - battery covers base usage
            true_savings = estimated_savings

        # Net post-discharge metrics (battery arbitrage value)
        # Gross values for Battery line matching
        gross_charged_kwh = len(actual_charge) * window_duration_hours * charge_power
        # Note: RTE is NOT applied here - it only affects battery state simulation, not cost/display calculations
        gross_usable_kwh = gross_charged_kwh  # No RTE penalty
        gross_discharged_kwh = len(actual_discharge) * window_duration_hours * discharge_power
        actual_remaining_kwh = gross_usable_kwh - gross_discharged_kwh

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

        # === CHRONOLOGICAL BUFFER TRACKING ===
        # Get buffer energy and limiting mode from config
        buffer_energy = self._get_buffer_energy(config, is_tomorrow, hass)
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
                aggressive_windows,
                config,
                is_tomorrow
            )
            charge_for_chrono = all_actual_charge
            _LOGGER.debug(
                f"Capacity-first: using {len(all_actual_charge)} candidates for chrono "
                f"(vs {len(actual_charge)} limited), will select cheapest {num_charge_windows}"
            )
        else:
            charge_for_chrono = actual_charge

        _LOGGER.debug(
            f"Chronological calc: buffer_energy={buffer_energy:.2f} kWh, "
            f"limit_discharge={limit_discharge}, battery_capacity={battery_capacity:.1f} kWh, "
            f"is_tomorrow={is_tomorrow}, discharge_windows={len(actual_discharge)}"
        )

        # Build chronological timeline and simulate battery state
        chrono_timeline = self._build_chronological_timeline(
            all_prices,  # Use all_prices for full day coverage
            charge_for_chrono,  # Use all candidates for capacity-first selection
            actual_discharge,
            [w for w in actual_discharge if w.get("state") == STATE_DISCHARGE_AGGRESSIVE],
            config
        )

        # When using sensor for TODAY, filter to only FUTURE windows
        # Past windows are already reflected in the sensor reading - don't re-simulate them
        use_sensor = config.get("use_battery_buffer_sensor", False)
        sensor_entity = config.get("battery_available_energy_sensor", "")
        using_sensor_for_today = not is_tomorrow and use_sensor and sensor_entity and hass

        if using_sensor_for_today:
            now = dt_util.now()
            # Keep only windows that haven't started yet (or are currently active)
            # Use a small buffer (1 minute) to include the current window
            future_timeline = []
            for entry in chrono_timeline:
                entry_ts = entry["timestamp"]
                if isinstance(entry_ts, str):
                    entry_ts = dt_util.parse_datetime(entry_ts)
                # Include if window end time is after now (window duration matters)
                window_end = entry_ts + timedelta(minutes=entry["duration"])
                if window_end > now:
                    future_timeline.append(entry)

            _LOGGER.debug(
                f"Sensor mode: filtered timeline from {len(chrono_timeline)} to {len(future_timeline)} future windows"
            )
            chrono_timeline = future_timeline

        # Debug: Log timeline composition before simulation
        charge_count = sum(1 for p in chrono_timeline if p["window_type"] == "charge")
        discharge_count = sum(1 for p in chrono_timeline if p["window_type"] in ("discharge", "aggressive"))
        idle_count = sum(1 for p in chrono_timeline if p["window_type"] == "idle")
        _LOGGER.info(
            f"Timeline before simulation: {len(chrono_timeline)} windows "
            f"(charge={charge_count}, discharge={discharge_count}, idle={idle_count}), "
            f"buffer_start={buffer_energy:.2f} kWh, limit_discharge={limit_discharge}"
        )

        chrono_result = self._simulate_chronological_costs(
            chrono_timeline, config, buffer_energy, limit_discharge
        )

        # Calculate buffer delta (end - start)
        final_battery_state = chrono_result["final_battery_state"]
        buffer_delta = final_battery_state - buffer_energy

        # Find current battery state
        # Priority: real sensor value (for today) > trajectory simulation > buffer_energy
        current_battery_state = buffer_energy  # Default to starting value
        use_sensor = config.get("use_battery_buffer_sensor", False)
        sensor_entity = config.get("battery_available_energy_sensor", "")

        # For today: use real sensor value if available (most accurate)
        if not is_tomorrow and use_sensor and sensor_entity and hass:
            try:
                sensor_state = hass.states.get(sensor_entity)
                if sensor_state and sensor_state.state not in ("unknown", "unavailable", None):
                    current_battery_state = float(sensor_state.state)
            except (ValueError, TypeError):
                pass  # Fall back to trajectory below

        # If no sensor value obtained, use simulated trajectory
        if current_battery_state == buffer_energy:
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

        if skipped_charge or (all_charge_candidates and len(all_charge_candidates) > len(charge_windows)):
            actual_charge = feasible_charge
            # REBUILD grouped windows with filtered charge windows (for dashboard display)
            grouped_charge_windows = self._group_consecutive_windows(
                actual_charge, avg_expensive_sell, is_discharge=False
            )
            # RECALCULATE completed_charge from filtered windows (was calculated before filtering)
            completed_charge = sum(
                1 for w in actual_charge
                if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time
            )
            # RECALCULATE completed_charge_cost from filtered windows
            completed_charge_cost = 0
            for w in actual_charge:
                if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time:
                    duration_hours = w["duration"] / 60
                    if charge_strategy == "grid_covers_both":
                        completed_charge_cost += w["price"] * duration_hours * (charge_power + base_usage)
                    else:  # battery_covers_base
                        completed_charge_cost += w["price"] * duration_hours * charge_power
            _LOGGER.info(
                f"Charge windows filtered: original={original_charge_count}, "
                f"feasible={len(actual_charge)}, skipped={len(skipped_charge)} (battery full), "
                f"completed={completed_charge}"
            )
            for skipped in skipped_charge:
                _LOGGER.debug(f"  Skipped charge: {skipped['timestamp']} - {skipped['reason']}")

        # When conservative discharge mode is enabled, replace discharge windows with only feasible ones
        if limit_discharge:
            original_discharge_count = len(actual_discharge)
            feasible_regular = chrono_result["feasible_discharge_windows"]
            feasible_aggressive = chrono_result.get("feasible_aggressive_windows", [])
            # Combine both regular and aggressive feasible windows for actual_discharge
            actual_discharge = feasible_regular + feasible_aggressive
            # Update aggressive_windows to only include feasible aggressive ones
            aggressive_windows = feasible_aggressive

            # REBUILD grouped windows with filtered discharge windows (for dashboard display)
            grouped_discharge_windows = self._group_consecutive_windows(
                actual_discharge, avg_cheap_buy, is_discharge=True
            )

            # RECALCULATE completed_discharge from filtered windows (was calculated before filtering)
            completed_discharge = sum(
                1 for w in actual_discharge
                if w["timestamp"] + timedelta(minutes=w["duration"]) <= current_time
            )
            # RECALCULATE completed_discharge_revenue from filtered windows
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
                    # Determine which strategy to use based on window state
                    if w.get("state") == STATE_DISCHARGE_AGGRESSIVE:
                        strategy = aggressive_strategy if aggressive_strategy != "same_as_discharge" else discharge_strategy
                    else:
                        strategy = discharge_strategy
                    if strategy == "already_included":
                        completed_discharge_revenue += sell_price * duration_hours * discharge_power
                    else:  # subtract_base
                        net_export = max(0, discharge_power - base_usage)
                        completed_discharge_revenue += sell_price * duration_hours * net_export

            _LOGGER.info(
                f"Conservative mode ACTIVE: buffer={buffer_energy:.2f} kWh, "
                f"original={original_discharge_count} windows, "
                f"feasible={len(actual_discharge)} windows, "
                f"skipped={len(chrono_result['skipped_discharge_windows'])} windows, "
                f"completed={completed_discharge}, "
                f"final_battery={chrono_result['final_battery_state']:.2f} kWh"
            )
            if chrono_result['skipped_discharge_windows']:
                for skipped in chrono_result['skipped_discharge_windows']:
                    _LOGGER.debug(f"  Skipped discharge: {skipped['timestamp']} - {skipped['reason']}")

        # RE-RUN chrono with ELECTED windows after capacity-first selection
        # The first chrono run used ALL candidates to determine feasibility
        # Now we run again with only the elected windows to get correct trajectory
        if all_charge_candidates and len(all_charge_candidates) > len(charge_windows):
            # Rebuild timeline with ELECTED windows only (not all candidates)
            elected_timeline = self._build_chronological_timeline(
                all_prices,
                actual_charge,  # ELECTED windows, not all candidates
                actual_discharge,
                [w for w in actual_discharge if w.get("state") == STATE_DISCHARGE_AGGRESSIVE],
                config
            )
            # Apply same future-only filter if using sensor
            if using_sensor_for_today:
                now = dt_util.now()
                elected_timeline = [
                    entry for entry in elected_timeline
                    if entry["timestamp"] + timedelta(minutes=entry["duration"]) > now
                ]
            # Re-run chrono to get correct trajectory for elected windows
            elected_chrono_result = self._simulate_chronological_costs(
                elected_timeline, config, buffer_energy, limit_discharge
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

            # Re-lookup current battery state from the new trajectory
            current_battery_state = buffer_energy
            if not is_tomorrow and use_sensor and sensor_entity and hass:
                try:
                    sensor_state = hass.states.get(sensor_entity)
                    if sensor_state and sensor_state.state not in ("unknown", "unavailable", None):
                        current_battery_state = float(sensor_state.state)
                except (ValueError, TypeError):
                    pass
            if current_battery_state == buffer_energy:
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

        # ALWAYS recalculate metrics using chrono_result as single source of truth
        # The chrono simulation correctly tracks battery capacity and RTE
        # Initial calculations (before chrono) use bulk formulas that ignore capacity limits
        windows_were_filtered = skipped_charge or (limit_discharge and chrono_result.get('skipped_discharge_windows'))
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
                # Determine which strategy to use based on window state
                if w.get("state") == STATE_DISCHARGE_AGGRESSIVE:
                    strategy = aggressive_strategy if aggressive_strategy != "same_as_discharge" else discharge_strategy
                else:
                    strategy = discharge_strategy
                if strategy == "already_included":
                    net_planned_discharge_kwh += duration_hours * discharge_power
                else:  # subtract_base
                    net_planned_discharge_kwh += duration_hours * max(0, discharge_power - base_usage)
            net_planned_discharge_kwh = round(net_planned_discharge_kwh, 3)

            # Recalculate dependent values
            # Note: RTE is NOT applied here - it only affects battery state simulation, not cost calculations
            usable_kwh = max(0, net_planned_charge_kwh - net_planned_discharge_kwh)
            if limit_savings_enabled:
                effective_base_usage_kwh = min(base_usage_kwh, usable_kwh)

            # Use accurate kWh values from chronological simulation
            # (chrono_result tracks actual grid draw/export accounting for feasibility)
            charged_kwh = chrono_result.get("actual_charge_kwh", 0)  # Grid draw (no RTE)
            discharged_kwh = chrono_result.get("actual_discharge_kwh", 0)  # Grid export
            uncovered_base = chrono_result.get("uncovered_base_kwh", 0)  # Grid covered base when battery empty

            # net_grid_kwh = grid import - grid export
            # grid_kwh_total already includes charge + uncovered base
            net_grid_kwh = chrono_result.get("grid_kwh_total", 0) - discharged_kwh

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
                if w.get("state") == STATE_DISCHARGE_AGGRESSIVE:
                    strategy = aggressive_strategy if aggressive_strategy != "same_as_discharge" else discharge_strategy
                else:
                    strategy = discharge_strategy
                if strategy == "already_included":
                    planned_discharge_revenue += sell_price * duration_hours * discharge_power
                else:  # subtract_base
                    net_export = max(0, discharge_power - base_usage)
                    planned_discharge_revenue += sell_price * duration_hours * net_export

            planned_total_cost = round(planned_charge_cost + planned_base_usage_cost - planned_discharge_revenue, 3)

            # Recalculate uncovered cost after chrono filtering (battery_covers_limited fallback to grid)
            if limit_savings_enabled and usable_kwh < base_usage_kwh:
                uncovered_kwh = base_usage_kwh - usable_kwh
                uncovered_cost = uncovered_kwh * day_avg_price
                planned_total_cost = round(planned_total_cost + uncovered_cost, 3)

            _LOGGER.debug(
                f"Planned costs recalculated from filtered windows: "
                f"charge_cost={planned_charge_cost:.3f}, discharge_revenue={planned_discharge_revenue:.3f}, "
                f"total_cost={planned_total_cost:.3f}"
            )

            # Recalculate savings
            baseline_cost = net_grid_kwh * day_avg_price
            estimated_savings = baseline_cost - planned_total_cost
            if net_grid_kwh >= base_usage_kwh and net_grid_kwh > 0:
                true_savings = (estimated_savings / net_grid_kwh) * base_usage_kwh
            else:
                true_savings = estimated_savings

            _LOGGER.debug(
                f"Gross metrics recalculated: charge={gross_charged_kwh:.2f}, usable={gross_usable_kwh:.2f}, "
                f"discharge={gross_discharged_kwh:.2f}, remaining={actual_remaining_kwh:.2f}, "
                f"net_charge={net_planned_charge_kwh:.2f}, net_discharge={net_planned_discharge_kwh:.2f}, "
                f"net_grid={net_grid_kwh:.2f}"
            )

        # Re-determine current state if any windows were filtered (charge or discharge)
        # If current hour was a window that got filtered, state should become IDLE
        if windows_were_filtered:
            current_state = self._determine_current_state(
                prices,
                actual_charge,  # Use filtered charge windows
                actual_discharge,  # Use filtered discharge windows
                aggressive_windows,  # Use filtered aggressive windows
                config,
                0.0,  # arbitrage_avg not needed for state check
                is_tomorrow
            )

        # Build result
        result = {
            "state": current_state,
            "cheapest_times": [w["timestamp"].isoformat() for w in charge_windows],
            "cheapest_prices": [float(w["price"]) for w in charge_windows],
            "expensive_times": [w["timestamp"].isoformat() for w in discharge_windows],
            "expensive_prices": [float(w["price"]) for w in discharge_windows],
            "expensive_sell_prices": [float(get_sell_price_for_window(w)) for w in discharge_windows],
            "expensive_times_aggressive": [w["timestamp"].isoformat() for w in aggressive_windows],
            "expensive_prices_aggressive": [float(w["price"]) for w in aggressive_windows],
            "expensive_sell_prices_aggressive": [float(get_sell_price_for_window(w)) for w in aggressive_windows],
            "actual_charge_times": [w["timestamp"].isoformat() for w in actual_charge],
            "actual_charge_prices": [float(w["price"]) for w in actual_charge],
            "actual_discharge_times": [w["timestamp"].isoformat() for w in actual_discharge],
            "actual_discharge_prices": [float(get_sell_price_for_window(w)) for w in actual_discharge],  # Use sell prices (discharge = selling to grid)
            "actual_discharge_sell_prices": [float(get_sell_price_for_window(w)) for w in actual_discharge],
            "completed_charge_windows": completed_charge,
            "completed_discharge_windows": completed_discharge,
            "completed_charge_cost": round(completed_charge_cost, 3),
            "completed_discharge_revenue": round(completed_discharge_revenue, 3),
            "completed_base_usage_cost": round(completed_base_usage_cost, 3),
            "completed_base_usage_battery": round(completed_base_usage_battery, 3),
            "uncovered_base_usage_kwh": round(uncovered_kwh, 3),
            "uncovered_base_usage_cost": round(uncovered_cost, 3),
            "total_cost": round(completed_charge_cost + completed_base_usage_cost + completed_uncovered_cost - completed_discharge_revenue, 3),
            "planned_total_cost": planned_total_cost,
            "planned_charge_cost": round(planned_charge_cost, 3),
            "net_planned_charge_kwh": net_planned_charge_kwh,
            "net_planned_discharge_kwh": net_planned_discharge_kwh,
            "effective_base_usage_kwh": effective_base_usage_kwh,
            "base_usage_kwh": round(base_usage_kwh, 3),
            "base_usage_day_cost": round(base_usage_kwh * day_avg_price, 2),  # Daily base usage cost at avg price
            "num_windows": len(charge_windows),
            # Profit-based attributes (v1.2.0+)
            "charge_profit_pct": round(charge_profit_pct, 1),  # Buy-buy profit for charging
            "discharge_profit_pct": round(discharge_profit_pct, 1),  # Buy-sell profit for discharge
            "charge_profit_met": bool(charge_profit_pct >= min_profit_charge),
            "discharge_profit_met": bool(discharge_profit_pct >= min_profit_discharge),
            "aggressive_profit_met": bool(discharge_profit_pct >= min_profit_discharge_aggressive),
            # Legacy spread attributes (deprecated, kept for backwards compatibility)
            "min_spread_required": config.get("min_profit_charge", 10),  # Map to new name
            "spread_percentage": round(arbitrage_avg, 1),  # For operational spread check (sell vs buy)
            "spread_met": bool(charge_profit_pct >= min_profit_charge),  # Map to profit check
            "spread_avg": round(spread_avg, 1),  # Buy vs buy (price volatility)
            "arbitrage_avg": round(arbitrage_avg, 1),  # Sell vs buy (arbitrage margin)
            "actual_spread_avg": round(spread_avg, 1),  # For backwards compatibility
            "discharge_spread_met": bool(discharge_profit_pct >= min_profit_discharge),  # Map to profit check
            "aggressive_discharge_spread_met": bool(discharge_profit_pct >= min_profit_discharge_aggressive),  # Map to profit check
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
            "cost_difference": round((base_usage_kwh * day_avg_price) - planned_total_cost, 2),
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
            # Grid usage tracking (from chronological simulation)
            "grid_kwh_estimated_today": chrono_result.get("grid_kwh_total", 0.0),
            "battery_state_current": round(current_battery_state, 3),
            "battery_state_end_of_day": chrono_result.get("final_battery_state", 0.0),
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
            "expensive_times_aggressive": [],
            "expensive_prices_aggressive": [],
            "expensive_sell_prices_aggressive": [],
            "actual_charge_times": [],
            "actual_charge_prices": [],
            "actual_discharge_times": [],
            "actual_discharge_prices": [],
            "actual_discharge_sell_prices": [],
            "completed_charge_windows": 0,
            "completed_discharge_windows": 0,
            "completed_charge_cost": 0,
            "completed_discharge_revenue": 0,
            "completed_base_usage_cost": 0,
            "completed_base_usage_battery": 0,
            "uncovered_base_usage_kwh": 0,
            "uncovered_base_usage_cost": 0,
            "total_cost": 0,
            "planned_total_cost": 0,
            "planned_charge_cost": 0,
            "net_planned_charge_kwh": 0,
            "net_planned_discharge_kwh": 0,
            "effective_base_usage_kwh": 0,
            "base_usage_kwh": 0,
            "base_usage_day_cost": 0,
            "num_windows": 0,
            # Profit-based attributes (v1.2.0+)
            "charge_profit_pct": 0,
            "discharge_profit_pct": 0,
            "charge_profit_met": False,
            "discharge_profit_met": False,
            "aggressive_profit_met": False,
            # Legacy spread attributes (deprecated)
            "min_spread_required": 0,
            "spread_percentage": 0,
            "spread_met": False,
            "spread_avg": 0,
            "arbitrage_avg": 0,
            "actual_spread_avg": 0,
            "discharge_spread_met": False,
            "aggressive_discharge_spread_met": False,
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
            "limit_discharge_to_buffer": False,
            "skipped_discharge_windows": [],
            "discharge_windows_limited": False,
            "feasibility_issues": [],
            "has_feasibility_issues": False,
        }