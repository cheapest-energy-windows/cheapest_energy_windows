"""Data coordinator for Cheapest Energy Windows."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    LOGGER_NAME,
    UPDATE_INTERVAL,
    CONF_PRICE_SENSOR,
    DEFAULT_PRICE_SENSOR,
    DEFAULT_BASE_USAGE,
    DEFAULT_BASE_USAGE_CHARGE_STRATEGY,
    DEFAULT_BASE_USAGE_NORMAL_STRATEGY,
    DEFAULT_BASE_USAGE_DISCHARGE_STRATEGY,
    DEFAULT_PRICE_COUNTRY,
    DEFAULT_SELL_FORMULA_PARAM_A,
    DEFAULT_SELL_FORMULA_PARAM_B,
    DEFAULT_USE_MIN_SELL_PRICE,
    DEFAULT_MIN_SELL_PRICE,
    DEFAULT_MIN_SELL_PRICE_BYPASS_SPREAD,
    DEFAULT_BUY_FORMULA_PARAM_A,
    DEFAULT_BUY_FORMULA_PARAM_B,
    DEFAULT_VAT_RATE,
    DEFAULT_TAX,
    DEFAULT_ADDITIONAL_COST,
    PREFIX,
)
from .statistics_helper import fetch_day_statistics, get_sensor_state_at_midnight
from .energy_prefs_helper import get_energy_preferences

_LOGGER = logging.getLogger(LOGGER_NAME)


class CEWCoordinator(DataUpdateCoordinator[Dict[str, Any]]):
    """Class to manage fetching Cheapest Energy Windows data."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=UPDATE_INTERVAL,
        )

        self.config_entry = config_entry
        self.price_sensor = config_entry.data.get(CONF_PRICE_SENSOR, DEFAULT_PRICE_SENSOR)

        # Track previous price data to detect changes (Layer 2)
        # Store in hass.data to persist across integration reloads
        persistent_key = f"{DOMAIN}_{config_entry.entry_id}_price_state"
        if persistent_key not in hass.data:
            hass.data[persistent_key] = {
                "previous_raw_today": None,
                "previous_raw_tomorrow": None,
                "last_price_update": None,
                "last_config_update": None,
                "previous_config_hash": None,
                "previous_energy_hash": None,
            }
        self._persistent_state = hass.data[persistent_key]

        # Instance variables (for convenience, but backed by persistent storage)
        self._previous_raw_today: Optional[list] = self._persistent_state["previous_raw_today"]
        self._previous_raw_tomorrow: Optional[list] = self._persistent_state["previous_raw_tomorrow"]
        self._last_price_update: Optional[datetime] = self._persistent_state["last_price_update"]
        self._last_config_update: Optional[datetime] = self._persistent_state["last_config_update"]
        self._previous_config_hash: Optional[str] = self._persistent_state["previous_config_hash"]
        self._previous_energy_hash: Optional[str] = self._persistent_state.get("previous_energy_hash")

    async def _async_update_data(self) -> Dict[str, Any]:
        """Fetch data from price sensor."""
        try:
            # Always use the proxy sensor which normalizes different price sensor formats
            price_sensor = "sensor.cew_price_sensor_proxy"
            price_state = self.hass.states.get(price_sensor)

            if not price_state:
                _LOGGER.warning(f"Price sensor {price_sensor} not found")
                return await self._empty_data(f"Price sensor {price_sensor} not found")

            # Extract price data
            raw_today = price_state.attributes.get("raw_today", [])
            raw_tomorrow = price_state.attributes.get("raw_tomorrow", [])
            tomorrow_valid = price_state.attributes.get("tomorrow_valid", False)

            if not raw_today:
                _LOGGER.warning("No price data available for today")
                return await self._empty_data("No price data available")

            # Get configuration from config entry options (Layer 1: no race conditions)
            config = await self._get_configuration()

            # Layer 2: Detect what changed
            now = dt_util.now()
            price_data_changed = False
            config_changed = False
            is_first_load = False
            scheduled_update = False  # New: track scheduled updates where nothing changed

            # Check if price data changed
            # Compare lengths and a hash of the data for more reliable comparison
            def _price_data_hash(data):
                """Create a simple hash of price data for comparison."""
                if not data:
                    return ""
                # Create hash from length and first/last items
                try:
                    return f"{len(data)}_{data[0].get('value', 0)}_{data[-1].get('value', 0)}"
                except (IndexError, AttributeError, TypeError):
                    return str(len(data))

            def _config_hash(cfg):
                """Create a simple hash of config for comparison."""
                # Convert config dict to a sorted tuple of items for consistent hashing
                try:
                    return str(hash(tuple(sorted((k, str(v)) for k, v in cfg.items()))))
                except (TypeError, AttributeError):
                    return str(cfg)

            current_today_hash = _price_data_hash(raw_today)
            current_tomorrow_hash = _price_data_hash(raw_tomorrow)
            previous_today_hash = _price_data_hash(self._previous_raw_today)
            previous_tomorrow_hash = _price_data_hash(self._previous_raw_tomorrow)

            current_config_hash = _config_hash(config)
            previous_config_hash = self._previous_config_hash

            # Check if this is the first load (no previous data)
            if not previous_today_hash and not previous_tomorrow_hash:
                # First load after restart/reload - treat as initialization
                is_first_load = True
                config_changed = True  # Treat as config change to avoid state transitions
                self._last_config_update = now
                self._persistent_state["last_config_update"] = now
                _LOGGER.info("Coordinator: First load - initializing")
            elif current_today_hash != previous_today_hash or current_tomorrow_hash != previous_tomorrow_hash:
                price_data_changed = True
                self._last_price_update = now
                self._persistent_state["last_price_update"] = now
                _LOGGER.info("Coordinator: Price data changed")
            elif previous_config_hash and current_config_hash != previous_config_hash:
                config_changed = True
                self._last_config_update = now
                self._persistent_state["last_config_update"] = now
                _LOGGER.info("Coordinator: Config changed")
            else:
                # Nothing changed - this is a scheduled update for time-based state changes
                scheduled_update = True

            # Store current price data and config hash for next comparison
            self._previous_raw_today = raw_today.copy() if raw_today else []
            self._previous_raw_tomorrow = raw_tomorrow.copy() if raw_tomorrow else []
            self._previous_config_hash = current_config_hash
            self._persistent_state["previous_raw_today"] = self._previous_raw_today
            self._persistent_state["previous_raw_tomorrow"] = self._previous_raw_tomorrow
            self._persistent_state["previous_config_hash"] = current_config_hash

            # Fetch HA Energy Dashboard statistics if toggle is enabled
            energy_statistics = {}
            solar_forecast_today = None
            solar_forecast_tomorrow = None
            energy_stats_changed = False

            if config.get("use_ha_energy_dashboard", False):
                try:
                    energy_statistics = await fetch_day_statistics(self.hass, config)
                    _LOGGER.debug("HA Energy Dashboard: stats_available=%s", energy_statistics.get("stats_available"))
                except Exception as e:
                    _LOGGER.warning("Failed to fetch HA Energy statistics: %s", e)
                    energy_statistics = {"stats_available": False}

            def _energy_stats_hash(stats):
                """Hash based on which hours have data."""
                if not stats:
                    return ""
                grid_hours = sorted(stats.get("grid_import_hourly", {}).keys())
                solar_hours = sorted(stats.get("solar_hourly", {}).keys())
                return f"g{len(grid_hours)}_{grid_hours}_s{len(solar_hours)}_{solar_hours}"

            current_energy_hash = _energy_stats_hash(energy_statistics)
            previous_energy_hash = self._previous_energy_hash or ""
            # Detect changes in both directions: ON→OFF (cleared) and OFF→ON (populated)
            energy_stats_changed = current_energy_hash != previous_energy_hash

            if energy_stats_changed:
                if current_energy_hash == "":
                    _LOGGER.info("Coordinator: Energy stats cleared (HA Energy disabled)")
                else:
                    _LOGGER.info("Coordinator: Energy stats changed (new hourly data)")
                self._previous_energy_hash = current_energy_hash
                self._persistent_state["previous_energy_hash"] = current_energy_hash

            if config.get("use_ha_energy_dashboard", False):
                try:
                    energy_prefs = await get_energy_preferences(self.hass)
                    forecast_config_entry = energy_prefs.get("solar_forecast_config_entry")

                    if forecast_config_entry:
                        entity_registry = er.async_get(self.hass)
                        for entity in entity_registry.entities.values():
                            if entity.config_entry_id in forecast_config_entry:
                                entity_id = entity.entity_id
                                entity_lower = entity_id.lower()
                                # Look for today/tomorrow forecast sensors
                                # Exclude "remaining" entities - we want total forecast, not remaining
                                if "remaining" in entity_lower:
                                    continue
                                if solar_forecast_today is None and "today" in entity_lower and "energy" in entity_lower:
                                    state = self.hass.states.get(entity_id)
                                    if state and state.state not in ("unknown", "unavailable"):
                                        try:
                                            solar_forecast_today = float(state.state)
                                            _LOGGER.debug("Solar forecast today from %s: %.2f kWh", entity_id, solar_forecast_today)
                                        except (ValueError, TypeError):
                                            pass
                                elif solar_forecast_tomorrow is None and "tomorrow" in entity_lower and "energy" in entity_lower:
                                    state = self.hass.states.get(entity_id)
                                    if state and state.state not in ("unknown", "unavailable"):
                                        try:
                                            solar_forecast_tomorrow = float(state.state)
                                            _LOGGER.debug("Solar forecast tomorrow from %s: %.2f kWh", entity_id, solar_forecast_tomorrow)
                                        except (ValueError, TypeError):
                                            pass
                except Exception as e:
                    _LOGGER.debug("Could not fetch solar forecast from HA Energy: %s", e)

                # Add solar forecast to energy_statistics for calculation_engine
                energy_statistics["solar_forecast_today"] = solar_forecast_today
                energy_statistics["solar_forecast_tomorrow"] = solar_forecast_tomorrow

                solar_forecast_hourly_today = {}
                solar_forecast_hourly_tomorrow = {}

                if forecast_config_entry:
                    try:
                        from homeassistant.components.forecast_solar.energy import async_get_solar_forecast

                        _LOGGER.debug("Fetching hourly solar forecast for entries: %s", forecast_config_entry)
                        today_date = now.date()
                        tomorrow_date = today_date + timedelta(days=1)

                        # Call async_get_solar_forecast for each config entry
                        for entry_id in forecast_config_entry:
                            try:
                                forecast_data = await async_get_solar_forecast(self.hass, entry_id)
                                _LOGGER.debug("Hourly forecast data for %s: %s", entry_id, "received" if forecast_data else "None")

                                if forecast_data:
                                    wh_hours = forecast_data.get("wh_hours", {}) or {}
                                    for timestamp_str, wh_value in wh_hours.items():
                                        try:
                                            ts = datetime.fromisoformat(timestamp_str)
                                            hour = ts.hour
                                            kwh = wh_value / 1000.0

                                            if ts.date() == today_date:
                                                solar_forecast_hourly_today[hour] = solar_forecast_hourly_today.get(hour, 0) + kwh
                                            elif ts.date() == tomorrow_date:
                                                solar_forecast_hourly_tomorrow[hour] = solar_forecast_hourly_tomorrow.get(hour, 0) + kwh
                                        except (ValueError, TypeError):
                                            continue
                            except Exception as e:
                                _LOGGER.debug("Could not fetch forecast for entry %s: %s", entry_id, e)

                        if solar_forecast_hourly_today:
                            _LOGGER.debug("Hourly solar forecast today: %d hours, total %.2f kWh",
                                        len(solar_forecast_hourly_today), sum(solar_forecast_hourly_today.values()))
                        if solar_forecast_hourly_tomorrow:
                            _LOGGER.debug("Hourly solar forecast tomorrow: %d hours, total %.2f kWh",
                                        len(solar_forecast_hourly_tomorrow), sum(solar_forecast_hourly_tomorrow.values()))

                    except ImportError:
                        _LOGGER.debug("async_get_solar_forecast not available in this HA version")
                    except Exception as e:
                        _LOGGER.debug("Could not fetch hourly solar forecast: %s", e)

                energy_statistics["solar_forecast_hourly_today"] = solar_forecast_hourly_today
                energy_statistics["solar_forecast_hourly_tomorrow"] = solar_forecast_hourly_tomorrow

            midnight_battery_state = None
            if config.get("use_battery_buffer_sensor", False):
                sensor_entity = config.get("battery_available_energy_sensor", "")
                if sensor_entity and sensor_entity != "not_configured":
                    try:
                        midnight_battery_state = await get_sensor_state_at_midnight(
                            self.hass, sensor_entity
                        )
                        if midnight_battery_state is not None:
                            _LOGGER.debug("Battery midnight state: %.2f kWh", midnight_battery_state)
                    except Exception as e:
                        _LOGGER.warning("Failed to fetch midnight battery state: %s", e)

            # Process the data with metadata
            return {
                "price_sensor": price_sensor,
                "raw_today": raw_today,
                "raw_tomorrow": raw_tomorrow,
                "tomorrow_valid": tomorrow_valid,
                "config": config,
                "last_update": now,
                # Layer 2: Change tracking metadata
                "price_data_changed": price_data_changed,
                "config_changed": config_changed,
                "is_first_load": is_first_load,
                "scheduled_update": scheduled_update,
                "energy_stats_changed": energy_stats_changed,
                "last_price_update": self._last_price_update,
                "last_config_update": self._last_config_update,
                "energy_statistics": energy_statistics,
                "solar_forecast_today": solar_forecast_today,
                "solar_forecast_tomorrow": solar_forecast_tomorrow,
                "midnight_battery_state": midnight_battery_state,
            }

        except Exception as e:
            _LOGGER.error(f"Coordinator update failed: {e}", exc_info=True)
            raise UpdateFailed(f"Error fetching data: {e}") from e


    async def _get_configuration(self) -> Dict[str, Any]:
        """Get current configuration from config entry options.

        Reading from config_entry.options instead of entity states eliminates
        race conditions where entity states might be temporarily unavailable
        during updates.
        """
        from .const import (
            DEFAULT_CHARGING_WINDOWS,
            DEFAULT_EXPENSIVE_WINDOWS,
            DEFAULT_PERCENTILE_THRESHOLD,
            DEFAULT_MIN_PROFIT_CHARGE,
            DEFAULT_MIN_PROFIT_DISCHARGE,
            DEFAULT_MIN_PRICE_DIFFERENCE,
            DEFAULT_BATTERY_RTE,
            DEFAULT_CHARGE_POWER,
            DEFAULT_DISCHARGE_POWER,
            DEFAULT_PRICE_OVERRIDE_THRESHOLD,
            DEFAULT_QUIET_START,
            DEFAULT_QUIET_END,
            DEFAULT_TIME_OVERRIDE_START,
            DEFAULT_TIME_OVERRIDE_END,
            DEFAULT_CALCULATION_WINDOW_START,
            DEFAULT_CALCULATION_WINDOW_END,
            DEFAULT_DISCHARGE_BUFFER_LIMIT_KWH,
            # Buffer/chronological calculation defaults
            DEFAULT_BATTERY_BUFFER_KWH,
            DEFAULT_BATTERY_CAPACITY,
            DEFAULT_USE_BATTERY_BUFFER_SENSOR,
            DEFAULT_LIMIT_DISCHARGE_TO_BUFFER,
            # Solar production defaults
            DEFAULT_SOLAR_WINDOW_START,
            DEFAULT_SOLAR_WINDOW_END,
            DEFAULT_SOLAR_PRIORITY_STRATEGY,
            DEFAULT_EXPECTED_SOLAR_KWH,
        )

        options = self.config_entry.options

        # Number values with defaults
        config = {
            # Today's configuration
            "charging_windows": float(options.get("charging_windows", DEFAULT_CHARGING_WINDOWS)),
            "expensive_windows": float(options.get("expensive_windows", DEFAULT_EXPENSIVE_WINDOWS)),
            "percentile_threshold": float(options.get("percentile_threshold", DEFAULT_PERCENTILE_THRESHOLD)),
            "min_profit_charge": float(options.get("min_profit_charge", DEFAULT_MIN_PROFIT_CHARGE)),
            "min_profit_discharge": float(options.get("min_profit_discharge", DEFAULT_MIN_PROFIT_DISCHARGE)),
            "min_price_difference": float(options.get("min_price_difference", DEFAULT_MIN_PRICE_DIFFERENCE)),
            "battery_rte": float(options.get("battery_rte", DEFAULT_BATTERY_RTE)),
            "charge_power": float(options.get("charge_power", DEFAULT_CHARGE_POWER)),
            "discharge_power": float(options.get("discharge_power", DEFAULT_DISCHARGE_POWER)),
            # RTE-aware discharge (global battery setting)
            "rte_aware_discharge": bool(options.get("rte_aware_discharge", True)),
            "rte_discharge_margin": float(options.get("rte_discharge_margin", 2)),  # 2% default (as percentage)
            "base_usage": float(options.get("base_usage", DEFAULT_BASE_USAGE)),
            "base_usage_charge_strategy": options.get("base_usage_charge_strategy", DEFAULT_BASE_USAGE_CHARGE_STRATEGY),
            "base_usage_normal_strategy": options.get("base_usage_normal_strategy", DEFAULT_BASE_USAGE_NORMAL_STRATEGY),
            "base_usage_discharge_strategy": options.get("base_usage_discharge_strategy", DEFAULT_BASE_USAGE_DISCHARGE_STRATEGY),
            "price_override_threshold": float(options.get("price_override_threshold", DEFAULT_PRICE_OVERRIDE_THRESHOLD)),
            "discharge_buffer_limit_kwh": float(options.get("discharge_buffer_limit_kwh", DEFAULT_DISCHARGE_BUFFER_LIMIT_KWH)),

            # Tomorrow's configuration
            "charging_windows_tomorrow": float(options.get("charging_windows_tomorrow", DEFAULT_CHARGING_WINDOWS)),
            "expensive_windows_tomorrow": float(options.get("expensive_windows_tomorrow", DEFAULT_EXPENSIVE_WINDOWS)),
            "percentile_threshold_tomorrow": float(options.get("percentile_threshold_tomorrow", DEFAULT_PERCENTILE_THRESHOLD)),
            "min_profit_charge_tomorrow": float(options.get("min_profit_charge_tomorrow", DEFAULT_MIN_PROFIT_CHARGE)),
            "min_profit_discharge_tomorrow": float(options.get("min_profit_discharge_tomorrow", DEFAULT_MIN_PROFIT_DISCHARGE)),
            "price_override_threshold_tomorrow": float(options.get("price_override_threshold_tomorrow", DEFAULT_PRICE_OVERRIDE_THRESHOLD)),

            # Boolean values (switches)
            "automation_enabled": bool(options.get("automation_enabled", True)),
            "tomorrow_settings_enabled": bool(options.get("tomorrow_settings_enabled", False)),
            "midnight_rotation_notifications": bool(options.get("midnight_rotation_notifications", False)),
            "notifications_enabled": bool(options.get("notifications_enabled", True)),
            "quiet_hours_enabled": bool(options.get("quiet_hours_enabled", False)),
            "price_override_enabled": bool(options.get("price_override_enabled", False)),
            "price_override_enabled_tomorrow": bool(options.get("price_override_enabled_tomorrow", False)),
            "time_override_enabled": bool(options.get("time_override_enabled", False)),
            "time_override_enabled_tomorrow": bool(options.get("time_override_enabled_tomorrow", False)),
            "calculation_window_enabled": bool(options.get("calculation_window_enabled", False)),
            "calculation_window_enabled_tomorrow": bool(options.get("calculation_window_enabled_tomorrow", False)),
            "notify_automation_disabled": bool(options.get("notify_automation_disabled", False)),
            "notify_charging": bool(options.get("notify_charging", True)),
            "notify_discharge": bool(options.get("notify_discharge", True)),
            "notify_normal": bool(options.get("notify_normal", False)),

            # String values (selects)
            "pricing_window_duration": options.get("pricing_window_duration", "15_minutes"),
            "time_override_mode": options.get("time_override_mode", "charge"),
            "time_override_mode_tomorrow": options.get("time_override_mode_tomorrow", "charge"),

            # Unified price country
            "price_country": options.get("price_country", DEFAULT_PRICE_COUNTRY),

            # Buy formula parameters
            "buy_formula_param_a": float(options.get("buy_formula_param_a", DEFAULT_BUY_FORMULA_PARAM_A)),
            "buy_formula_param_b": float(options.get("buy_formula_param_b", DEFAULT_BUY_FORMULA_PARAM_B)),

            # Netherlands-specific fields (apply to both buy and sell)
            "vat": float(options.get("vat", DEFAULT_VAT_RATE)),  # Stored as percentage
            "tax": float(options.get("tax", DEFAULT_TAX)),
            "additional_cost": float(options.get("additional_cost", DEFAULT_ADDITIONAL_COST)),

            # Sell formula parameters
            "sell_formula_param_a": float(options.get("sell_formula_param_a", DEFAULT_SELL_FORMULA_PARAM_A)),
            "sell_formula_param_b": float(options.get("sell_formula_param_b", DEFAULT_SELL_FORMULA_PARAM_B)),
            "use_min_sell_price": bool(options.get("use_min_sell_price", DEFAULT_USE_MIN_SELL_PRICE)),
            "min_sell_price": float(options.get("min_sell_price", DEFAULT_MIN_SELL_PRICE)),
            "min_sell_price_bypass_spread": bool(options.get("min_sell_price_bypass_spread", DEFAULT_MIN_SELL_PRICE_BYPASS_SPREAD)),

            # Buffer/chronological calculation settings
            "battery_buffer_kwh": float(options.get("battery_buffer_kwh", DEFAULT_BATTERY_BUFFER_KWH)),
            "battery_buffer_kwh_tomorrow": float(options.get("battery_buffer_kwh_tomorrow", DEFAULT_BATTERY_BUFFER_KWH)),
            "battery_capacity": float(options.get("battery_capacity", DEFAULT_BATTERY_CAPACITY)),
            "use_battery_buffer_sensor": bool(options.get("use_battery_buffer_sensor", DEFAULT_USE_BATTERY_BUFFER_SENSOR)),
            "limit_discharge_to_buffer": bool(options.get("limit_discharge_to_buffer", DEFAULT_LIMIT_DISCHARGE_TO_BUFFER)),
            "use_projected_buffer_tomorrow": bool(options.get("use_projected_buffer_tomorrow", False)),
            "battery_available_energy_sensor": options.get("battery_available_energy_sensor", ""),
            "min_price_diff_enabled": bool(options.get("min_price_diff_enabled", True)),

            # Solar production settings
            "solar_window_start": options.get("solar_window_start", DEFAULT_SOLAR_WINDOW_START),
            "solar_window_end": options.get("solar_window_end", DEFAULT_SOLAR_WINDOW_END),
            "solar_priority_strategy": options.get("solar_priority_strategy", DEFAULT_SOLAR_PRIORITY_STRATEGY),
            "expected_solar_kwh": float(options.get("expected_solar_kwh", DEFAULT_EXPECTED_SOLAR_KWH)),
            "expected_solar_kwh_tomorrow": float(options.get("expected_solar_kwh_tomorrow", DEFAULT_EXPECTED_SOLAR_KWH)),
            "use_solar_forecast": bool(options.get("use_solar_forecast", True)),

            # Auto-optimization settings
            "auto_optimize_strategy": options.get("auto_optimize_strategy", "off"),
            "auto_optimize_strategy_tomorrow": options.get("auto_optimize_strategy_tomorrow", "off"),
            "min_daily_savings": float(options.get("min_daily_savings", 0.50)),
            "min_daily_savings_tomorrow": float(options.get("min_daily_savings_tomorrow", 0.50)),
            "fast_search": options.get("fast_search", True),

            # HA Energy Dashboard integration
            "use_ha_energy_dashboard": bool(options.get("use_ha_energy_dashboard", False)),

            # Time values
            "time_override_start": options.get("time_override_start", DEFAULT_TIME_OVERRIDE_START),
            "time_override_end": options.get("time_override_end", DEFAULT_TIME_OVERRIDE_END),
            "time_override_start_tomorrow": options.get("time_override_start_tomorrow", DEFAULT_TIME_OVERRIDE_START),
            "time_override_end_tomorrow": options.get("time_override_end_tomorrow", DEFAULT_TIME_OVERRIDE_END),
            "calculation_window_start": options.get("calculation_window_start", DEFAULT_CALCULATION_WINDOW_START),
            "calculation_window_end": options.get("calculation_window_end", DEFAULT_CALCULATION_WINDOW_END),
            "calculation_window_start_tomorrow": options.get("calculation_window_start_tomorrow", DEFAULT_CALCULATION_WINDOW_START),
            "calculation_window_end_tomorrow": options.get("calculation_window_end_tomorrow", DEFAULT_CALCULATION_WINDOW_END),
            "quiet_hours_start": options.get("quiet_hours_start", DEFAULT_QUIET_START),
            "quiet_hours_end": options.get("quiet_hours_end", DEFAULT_QUIET_END),
        }

        return config

    async def async_request_refresh(self) -> None:
        """Request an immediate coordinator refresh."""
        await super(CEWCoordinator, self).async_refresh()

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value from the coordinator data."""
        if self.data and "config" in self.data:
            return self.data["config"].get(key, default)
        return default

    async def _empty_data(self, reason: str) -> Dict[str, Any]:
        """Return empty data structure when price sensor is not available."""
        # Still get config so settings are available
        config = await self._get_configuration()

        return {
            "price_sensor": None,
            "raw_today": [],
            "raw_tomorrow": [],
            "tomorrow_valid": False,
            "config": config,
            "last_update": dt_util.now(),
            "error": reason,
            "energy_statistics": {},
            "solar_forecast_today": None,
            "solar_forecast_tomorrow": None,
        }