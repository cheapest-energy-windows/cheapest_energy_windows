"""Sensor platform for Cheapest Energy Windows."""
from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional
import uuid

from homeassistant.components.sensor import (
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .calculation_engine import WindowCalculationEngine
from .const import (
    DOMAIN,
    LOGGER_NAME,
    PREFIX,
    VERSION,
    STATE_CHARGE,
    STATE_DISCHARGE,
    STATE_IDLE,
    STATE_OFF,
    STATE_AVAILABLE,
    STATE_UNAVAILABLE,
    ATTR_CHEAPEST_TIMES,
    ATTR_CHEAPEST_PRICES,
    ATTR_EXPENSIVE_TIMES,
    ATTR_EXPENSIVE_PRICES,
    ATTR_ACTUAL_CHARGE_TIMES,
    ATTR_ACTUAL_CHARGE_PRICES,
    ATTR_ACTUAL_DISCHARGE_TIMES,
    ATTR_ACTUAL_DISCHARGE_PRICES,
    ATTR_COMPLETED_CHARGE_WINDOWS,
    ATTR_COMPLETED_DISCHARGE_WINDOWS,
    ATTR_COMPLETED_CHARGE_COST,
    ATTR_COMPLETED_DISCHARGE_REVENUE,
    ATTR_COMPLETED_SOLAR_EXPORT_REVENUE,
    ATTR_COMPLETED_BASE_USAGE_COST,
    ATTR_COMPLETED_BASE_USAGE_BATTERY,
    ATTR_COMPLETED_CHARGE_KWH,
    ATTR_COMPLETED_DISCHARGE_KWH,
    ATTR_COMPLETED_BASE_GRID_KWH,
    ATTR_COMPLETED_NET_GRID_KWH,
    ATTR_TOTAL_COST,
    ATTR_PLANNED_TOTAL_COST,
    ATTR_PLANNED_CHARGE_COST,
    ATTR_NUM_WINDOWS,
    # Profit-based attributes (v1.2.0+)
    ATTR_CHARGE_PROFIT_PCT,
    ATTR_DISCHARGE_PROFIT_PCT,
    ATTR_CHARGE_PROFIT_MET,
    ATTR_DISCHARGE_PROFIT_MET,
    ATTR_AVG_CHEAP_PRICE,
    ATTR_AVG_EXPENSIVE_PRICE,
    ATTR_CURRENT_PRICE,
    ATTR_PRICE_OVERRIDE_ACTIVE,
    ATTR_TIME_OVERRIDE_ACTIVE,
    ATTR_CURRENT_SELL_PRICE,
    ATTR_SELL_PRICE_COUNTRY,
    # Grid and battery state tracking (v1.2.0+)
    ATTR_GRID_KWH_ESTIMATED_TODAY,
    ATTR_GRID_KWH_ESTIMATED_TOMORROW,
    ATTR_BATTERY_STATE_CURRENT,
    ATTR_BATTERY_STATE_END_OF_DAY,
    ATTR_BATTERY_STATE_END_OF_DAY_VALUE,
    ATTR_BATTERY_STATE_END_OF_TOMORROW,
    DEFAULT_PRICE_COUNTRY,
)
from .coordinator import CEWCoordinator
from .formulas import get_formula

_LOGGER = logging.getLogger(LOGGER_NAME)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Cheapest Energy Windows sensors."""
    coordinator = hass.data[DOMAIN][config_entry.entry_id]["coordinator"]

    sensors = [
        CEWTodaySensor(coordinator, config_entry),
        CEWTomorrowSensor(coordinator, config_entry),
        CEWPriceSensorProxy(hass, coordinator, config_entry),
        CEWLastCalculationSensor(coordinator, config_entry),
    ]

    async_add_entities(sensors)


class CEWBaseSensor(CoordinatorEntity, SensorEntity):
    """Base class for CEW sensors."""

    def __init__(
        self,
        coordinator: CEWCoordinator,
        config_entry: ConfigEntry,
        sensor_type: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.config_entry = config_entry
        self._sensor_type = sensor_type

        # Set unique ID and name
        self._attr_unique_id = f"{PREFIX}{sensor_type}"
        # Force consistent entity_id based on sensor_type, not display name
        self.entity_id = f"sensor.{PREFIX}{sensor_type}"
        self._attr_name = f"CEW {sensor_type.replace('_', ' ').title()}"
        self._attr_has_entity_name = False

        # Initialize state
        self._attr_native_value = STATE_OFF

        # Track previous values to detect changes
        self._previous_state = None
        self._previous_attributes = None

        # Persist automation_enabled across sensor recreations (integration reloads)
        # This allows us to detect actual changes in automation state
        persistent_key = f"{DOMAIN}_{config_entry.entry_id}_sensor_{sensor_type}_state"
        if persistent_key not in coordinator.hass.data:
            coordinator.hass.data[persistent_key] = {
                "previous_automation_enabled": None,
                "previous_calc_config_hash": None,
            }
        self._persistent_sensor_state = coordinator.hass.data[persistent_key]
        self._previous_automation_enabled = self._persistent_sensor_state["previous_automation_enabled"]
        self._previous_calc_config_hash = self._persistent_sensor_state["previous_calc_config_hash"]

    @property
    def device_info(self) -> Dict[str, Any]:
        """Return device information."""
        return {
            "identifiers": {(DOMAIN, self.config_entry.entry_id)},
            "name": "Cheapest Energy Windows",
            "manufacturer": "Community",
            "model": "Energy Optimizer",
            "sw_version": VERSION,
        }

    def _calc_config_hash(self, config: Dict[str, Any], is_tomorrow: bool = False) -> str:
        """Create a hash of config values that affect calculations.

        Only includes values that impact window calculations and current state.
        Excludes notification settings and other non-calculation config.
        """
        suffix = "_tomorrow" if is_tomorrow and config.get("tomorrow_settings_enabled", False) else ""

        # Config values that affect calculations
        calc_values = [
            config.get("automation_enabled", True),
            config.get(f"charging_windows{suffix}", 4),
            config.get(f"expensive_windows{suffix}", 4),
            config.get(f"percentile_threshold{suffix}", 25),
            config.get(f"min_spread{suffix}", 10),
            config.get(f"min_spread_discharge{suffix}", 20),
            config.get(f"min_price_difference{suffix}", 0.05),
            config.get("vat", 0.21),
            config.get("tax", 0.12286),
            config.get("additional_cost", 0.02398),
            config.get("battery_rte", 90),
            config.get("charge_power", 2400),
            config.get("discharge_power", 2400),
            config.get(f"price_override_enabled{suffix}", False),
            config.get(f"price_override_threshold{suffix}", 0.15),
            config.get("pricing_window_duration", "15_minutes"),
            # Calculation window settings affect what windows are selected
            config.get(f"calculation_window_enabled{suffix}", False),
            config.get(f"calculation_window_start{suffix}", "00:00:00"),
            config.get(f"calculation_window_end{suffix}", "23:59:59"),
        ]

        # Add time overrides (these affect current state)
        calc_values.extend([
            config.get(f"time_override_enabled{suffix}", False),
            config.get(f"time_override_start{suffix}", "00:00:00"),
            config.get(f"time_override_end{suffix}", "00:00:00"),
            config.get(f"time_override_mode{suffix}", "charge"),
        ])

        # Create hash from all values
        return str(hash(tuple(str(v) for v in calc_values)))


class CEWTodaySensor(CEWBaseSensor):
    """Sensor for today's energy windows."""

    def __init__(self, coordinator: CEWCoordinator, config_entry: ConfigEntry) -> None:
        """Initialize today sensor."""
        super().__init__(coordinator, config_entry, "today")
        self._calculation_engine = WindowCalculationEngine()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        if not self.coordinator.data:
            if self._previous_state is not None:
                return
            new_state = STATE_OFF
            new_attributes = {}
            self._attr_native_value = new_state
            self._attr_extra_state_attributes = new_attributes
            self._previous_state = new_state
            self._previous_attributes = new_attributes.copy() if new_attributes else None
            self.async_write_ha_state()
            return

        price_data_changed = self.coordinator.data.get("price_data_changed", True)
        config_changed = self.coordinator.data.get("config_changed", False)
        is_first_load = self.coordinator.data.get("is_first_load", False)
        scheduled_update = self.coordinator.data.get("scheduled_update", False)

        config = self.coordinator.data.get("config", {})
        current_automation_enabled = config.get("automation_enabled", True)

        current_calc_config_hash = self._calc_config_hash(config, is_tomorrow=False)
        calc_config_changed = (
            self._previous_calc_config_hash is None or
            self._previous_calc_config_hash != current_calc_config_hash
        )

        # Skip recalculation for non-calculation config changes
        if config_changed and not price_data_changed and not is_first_load and not calc_config_changed and not scheduled_update:
            return

        if calc_config_changed:
            _LOGGER.info("Today: Calculation config changed, recalculating")

        raw_today = self.coordinator.data.get("raw_today", [])

        if raw_today:
            result = self._calculation_engine.calculate_windows(
                raw_today, config, is_tomorrow=False, hass=self.coordinator.hass
            )
            new_state = result.get("state", STATE_OFF)
            new_attributes = self._build_attributes(result)

            # Store today's projected end-of-day battery state for tomorrow's calculation
            if config.get("use_projected_buffer_tomorrow", False):
                today_end_state = result.get("battery_state_end_of_day", 0.0)
                self.coordinator.data["_projected_buffer_tomorrow"] = today_end_state
        else:
            automation_enabled = config.get("automation_enabled", True)
            new_state = STATE_OFF if not automation_enabled else STATE_IDLE
            new_attributes = self._build_attributes({})

        state_changed = new_state != self._previous_state
        attributes_changed = new_attributes != self._previous_attributes

        if state_changed or attributes_changed:
            if state_changed:
                _LOGGER.info(f"Today: {self._previous_state} → {new_state}")

            self._attr_native_value = new_state
            self._attr_extra_state_attributes = new_attributes
            self._previous_state = new_state
            self._previous_attributes = new_attributes.copy() if new_attributes else None
            self._previous_automation_enabled = current_automation_enabled
            self._previous_calc_config_hash = current_calc_config_hash
            self._persistent_sensor_state["previous_automation_enabled"] = current_automation_enabled
            self._persistent_sensor_state["previous_calc_config_hash"] = current_calc_config_hash
            self.async_write_ha_state()
        else:
            self._previous_automation_enabled = current_automation_enabled
            self._previous_calc_config_hash = current_calc_config_hash
            self._persistent_sensor_state["previous_automation_enabled"] = current_automation_enabled
            self._persistent_sensor_state["previous_calc_config_hash"] = current_calc_config_hash

    def _build_attributes(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build sensor attributes from calculation result."""
        # Get last config update time from coordinator data
        last_config_update = self.coordinator.data.get("last_config_update") if self.coordinator.data else None

        return {
            ATTR_CHEAPEST_TIMES: result.get("cheapest_times", []),
            ATTR_CHEAPEST_PRICES: result.get("cheapest_prices", []),
            ATTR_EXPENSIVE_TIMES: result.get("expensive_times", []),
            ATTR_EXPENSIVE_PRICES: result.get("expensive_prices", []),
            ATTR_ACTUAL_CHARGE_TIMES: result.get("actual_charge_times", []),
            ATTR_ACTUAL_CHARGE_PRICES: result.get("actual_charge_prices", []),
            ATTR_ACTUAL_DISCHARGE_TIMES: result.get("actual_discharge_times", []),
            ATTR_ACTUAL_DISCHARGE_PRICES: result.get("actual_discharge_prices", []),
            ATTR_COMPLETED_CHARGE_WINDOWS: result.get("completed_charge_windows", 0),
            ATTR_COMPLETED_DISCHARGE_WINDOWS: result.get("completed_discharge_windows", 0),
            ATTR_COMPLETED_CHARGE_COST: result.get("completed_charge_cost", 0.0),
            ATTR_COMPLETED_DISCHARGE_REVENUE: result.get("completed_discharge_revenue", 0.0),
            ATTR_COMPLETED_SOLAR_EXPORT_REVENUE: result.get("completed_solar_export_revenue", 0.0),
            ATTR_COMPLETED_BASE_USAGE_COST: result.get("completed_base_usage_cost", 0.0),
            ATTR_COMPLETED_BASE_USAGE_BATTERY: result.get("completed_base_usage_battery", 0.0),
            ATTR_COMPLETED_CHARGE_KWH: result.get("completed_charge_kwh", 0.0),
            ATTR_COMPLETED_DISCHARGE_KWH: result.get("completed_discharge_kwh", 0.0),
            ATTR_COMPLETED_BASE_GRID_KWH: result.get("completed_base_grid_kwh", 0.0),
            ATTR_COMPLETED_NET_GRID_KWH: result.get("completed_net_grid_kwh", 0.0),
            ATTR_TOTAL_COST: result.get("total_cost", 0.0),
            ATTR_PLANNED_TOTAL_COST: result.get("planned_total_cost", 0.0),
            ATTR_PLANNED_CHARGE_COST: result.get("planned_charge_cost", 0.0),
            "net_planned_charge_kwh": result.get("net_planned_charge_kwh", 0.0),
            "net_planned_discharge_kwh": result.get("net_planned_discharge_kwh", 0.0),
            ATTR_NUM_WINDOWS: result.get("num_windows", 0),
            # Profit-based attributes (v1.2.0+)
            ATTR_CHARGE_PROFIT_PCT: result.get("charge_profit_pct", 0.0),
            ATTR_DISCHARGE_PROFIT_PCT: result.get("discharge_profit_pct", 0.0),
            ATTR_CHARGE_PROFIT_MET: result.get("charge_profit_met", False),
            ATTR_DISCHARGE_PROFIT_MET: result.get("discharge_profit_met", False),
            "spread_avg": result.get("spread_avg", 0.0),
            "arbitrage_avg": result.get("arbitrage_avg", 0.0),
            ATTR_AVG_CHEAP_PRICE: result.get("avg_cheap_price", 0.0),
            ATTR_AVG_EXPENSIVE_PRICE: result.get("avg_expensive_price", 0.0),
            ATTR_CURRENT_PRICE: result.get("current_price", 0.0),
            ATTR_CURRENT_SELL_PRICE: result.get("current_sell_price", 0.0),
            ATTR_SELL_PRICE_COUNTRY: result.get("price_country", "none"),
            "sell_formula_param_a": result.get("sell_formula_param_a", 0.1),
            "sell_formula_param_b": result.get("sell_formula_param_b", 0.0),
            "expensive_sell_prices": result.get("expensive_sell_prices", []),
            "actual_discharge_sell_prices": result.get("actual_discharge_sell_prices", []),
            ATTR_PRICE_OVERRIDE_ACTIVE: result.get("price_override_active", False),
            ATTR_TIME_OVERRIDE_ACTIVE: result.get("time_override_active", False),
            "last_config_update": last_config_update.isoformat() if last_config_update else None,
            # === DASHBOARD HELPER ATTRIBUTES ===
            "grouped_charge_windows": result.get("grouped_charge_windows", []),
            "grouped_discharge_windows": result.get("grouped_discharge_windows", []),
            "percentile_cheap_avg": result.get("percentile_cheap_avg", 0),
            "percentile_expensive_avg": result.get("percentile_expensive_avg", 0),
            "percentile_expensive_sell_avg": result.get("percentile_expensive_sell_avg", 0),
            "cheap_half_avg": result.get("cheap_half_avg", 0),
            "expensive_half_avg": result.get("expensive_half_avg", 0),
            "day_avg_price": result.get("day_avg_price", 0),
            "net_grid_kwh": result.get("net_grid_kwh", 0),
            "baseline_cost": result.get("baseline_cost", 0),
            "estimated_savings": result.get("estimated_savings", 0),
            "true_savings": result.get("true_savings", 0),
            "gross_charged_kwh": result.get("gross_charged_kwh", 0),
            "gross_usable_kwh": result.get("gross_usable_kwh", 0),
            "gross_discharged_kwh": result.get("gross_discharged_kwh", 0),
            "actual_remaining_kwh": result.get("actual_remaining_kwh", 0),
            "net_post_discharge_eur_kwh": result.get("net_post_discharge_eur_kwh", 0),
            "battery_margin_eur_kwh": result.get("battery_margin_eur_kwh", 0),
            "battery_arbitrage_value": result.get("battery_arbitrage_value", 0),
            "actual_price_kwh": result.get("actual_price_kwh", 0),
            "cost_difference": result.get("cost_difference", 0),
            "charge_power_kw": result.get("charge_power_kw", 0),
            "discharge_power_kw": result.get("discharge_power_kw", 0),
            "base_usage_kw": result.get("base_usage_kw", 0),
            "base_usage_kwh": result.get("base_usage_kwh", 0),
            "base_usage_day_cost": result.get("base_usage_day_cost", 0),
            "window_duration_hours": result.get("window_duration_hours", 0.25),
            # === CHRONOLOGICAL BUFFER TRACKING ===
            "buffer_energy_kwh": result.get("buffer_energy_kwh", 0),
            "final_battery_state_kwh": result.get("final_battery_state_kwh", 0),
            "buffer_delta_kwh": result.get("buffer_delta_kwh", 0),
            "battery_capacity_kwh": result.get("battery_capacity_kwh", 100.0),
            "chrono_charge_kwh": result.get("chrono_charge_kwh", 0),
            "chrono_discharge_kwh": result.get("chrono_discharge_kwh", 0),
            "chrono_uncovered_base_kwh": result.get("chrono_uncovered_base_kwh", 0),
            "limit_discharge_to_buffer": result.get("limit_discharge_to_buffer", False),
            "discharge_buffer_limit_kwh": result.get("discharge_buffer_limit_kwh", 0.0),
            "skipped_discharge_windows": result.get("skipped_discharge_windows", []),
            "discharge_windows_limited": result.get("discharge_windows_limited", False),
            "feasibility_issues": result.get("feasibility_issues", []),
            "has_feasibility_issues": result.get("has_feasibility_issues", False),
            # Grid and battery state tracking (v1.2.0+)
            ATTR_GRID_KWH_ESTIMATED_TODAY: result.get("grid_kwh_estimated_today", 0.0),
            ATTR_BATTERY_STATE_CURRENT: result.get("battery_state_current", 0.0),
            ATTR_BATTERY_STATE_END_OF_DAY: result.get("battery_state_end_of_day", 0.0),
            ATTR_BATTERY_STATE_END_OF_DAY_VALUE: result.get("battery_state_end_of_day_value", 0.0),
            # Solar integration metrics
            "solar_to_battery_kwh": result.get("solar_to_battery_kwh", 0.0),
            "solar_offset_base_kwh": result.get("solar_offset_base_kwh", 0.0),
            "solar_exported_kwh": result.get("solar_exported_kwh", 0.0),
            "solar_export_revenue": result.get("solar_export_revenue", 0.0),
            "solar_total_contribution_kwh": result.get("solar_total_contribution_kwh", 0.0),
            "grid_savings_from_solar": result.get("grid_savings_from_solar", 0.0),
            "expected_solar_kwh": result.get("expected_solar_kwh", 0.0),
            # Detailed battery tracking for energy flow report
            "battery_charged_from_grid_kwh": result.get("battery_charged_from_grid_kwh", 0.0),
            "battery_charged_from_grid_cost": result.get("battery_charged_from_grid_cost", 0.0),
            "battery_charged_from_solar_kwh": result.get("battery_charged_from_solar_kwh", 0.0),
            "battery_charged_avg_price": result.get("battery_charged_avg_price", 0.0),
            "battery_discharged_to_base_kwh": result.get("battery_discharged_to_base_kwh", 0.0),
            "battery_discharged_to_grid_kwh": result.get("battery_discharged_to_grid_kwh", 0.0),
            "battery_discharged_avg_price": result.get("battery_discharged_avg_price", 0.0),
            # RTE (Round-Trip Efficiency) loss tracking
            "rte_loss_kwh": result.get("rte_loss_kwh", 0.0),
            "rte_loss_value": result.get("rte_loss_value", 0.0),
            # RTE-aware discharge tracking
            "rte_preserved_kwh": result.get("rte_preserved_kwh", 0.0),
            "rte_preserved_periods": result.get("rte_preserved_periods", []),
            "rte_breakeven_price": result.get("rte_breakeven_price", 0.0),
        }


class CEWTomorrowSensor(CEWBaseSensor):
    """Sensor for tomorrow's energy windows."""

    def __init__(self, coordinator: CEWCoordinator, config_entry: ConfigEntry) -> None:
        """Initialize tomorrow sensor."""
        super().__init__(coordinator, config_entry, "tomorrow")
        self._calculation_engine = WindowCalculationEngine()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        if not self.coordinator.data:
            # No coordinator data - maintain previous state if we have one
            if self._previous_state is not None:
                return
            else:
                new_state = STATE_OFF
                new_attributes = {}
                self._attr_native_value = new_state
                self._attr_extra_state_attributes = new_attributes
                self._previous_state = new_state
                self._previous_attributes = new_attributes.copy() if new_attributes else None
                self.async_write_ha_state()
                return

        # Layer 3: Check what changed
        price_data_changed = self.coordinator.data.get("price_data_changed", True)
        config_changed = self.coordinator.data.get("config_changed", False)
        is_first_load = self.coordinator.data.get("is_first_load", False)
        scheduled_update = self.coordinator.data.get("scheduled_update", False)

        config = self.coordinator.data.get("config", {})
        current_automation_enabled = config.get("automation_enabled", True)

        # Check if calculation-affecting config changed
        current_calc_config_hash = self._calc_config_hash(config, is_tomorrow=True)
        calc_config_changed = (
            self._previous_calc_config_hash is None or
            self._previous_calc_config_hash != current_calc_config_hash
        )

        # Only skip recalculation for non-calculation config changes
        # Always recalculate for scheduled updates (needed for time-based state changes)
        if config_changed and not price_data_changed and not is_first_load and not calc_config_changed and not scheduled_update:
            return

        # Price data changed OR first run - proceed with recalculation
        tomorrow_valid = self.coordinator.data.get("tomorrow_valid", False)
        raw_tomorrow = self.coordinator.data.get("raw_tomorrow", [])

        if tomorrow_valid and raw_tomorrow:
            # Use today's projected end-of-day as tomorrow's starting buffer (if enabled)
            if config.get("use_projected_buffer_tomorrow", False):
                projected = self.coordinator.data.get("_projected_buffer_tomorrow")
                if projected is not None:
                    config = config.copy()  # Don't mutate original
                    config["_projected_buffer_tomorrow"] = projected

            # Calculate tomorrow's windows
            result = self._calculation_engine.calculate_windows(
                raw_tomorrow, config, is_tomorrow=True, hass=self.coordinator.hass
            )

            # Get calculated state from result (like today sensor does)
            new_state = result.get("state", STATE_OFF)
            new_attributes = self._build_attributes(result)
        else:
            # No tomorrow data yet (Nordpool publishes after 13:00 CET)
            new_state = STATE_OFF
            new_attributes = {}

        # Only update if state or attributes have changed
        state_changed = new_state != self._previous_state
        attributes_changed = new_attributes != self._previous_attributes

        if state_changed or attributes_changed:
            self._attr_native_value = new_state
            self._attr_extra_state_attributes = new_attributes
            self._previous_state = new_state
            self._previous_attributes = new_attributes.copy() if new_attributes else None
            self._previous_automation_enabled = current_automation_enabled
            self._previous_calc_config_hash = current_calc_config_hash
            self._persistent_sensor_state["previous_automation_enabled"] = current_automation_enabled
            self._persistent_sensor_state["previous_calc_config_hash"] = current_calc_config_hash
            self.async_write_ha_state()
        else:
            # Still update tracking even if state didn't change
            self._previous_automation_enabled = current_automation_enabled
            self._previous_calc_config_hash = current_calc_config_hash
            self._persistent_sensor_state["previous_automation_enabled"] = current_automation_enabled
            self._persistent_sensor_state["previous_calc_config_hash"] = current_calc_config_hash

    def _build_attributes(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build sensor attributes for tomorrow."""
        # Get last config update time from coordinator data
        last_config_update = self.coordinator.data.get("last_config_update") if self.coordinator.data else None

        # Tomorrow sensor has fewer attributes (no completed windows, etc.)
        return {
            ATTR_CHEAPEST_TIMES: result.get("cheapest_times", []),
            ATTR_CHEAPEST_PRICES: result.get("cheapest_prices", []),
            ATTR_EXPENSIVE_TIMES: result.get("expensive_times", []),
            ATTR_EXPENSIVE_PRICES: result.get("expensive_prices", []),
            ATTR_ACTUAL_CHARGE_TIMES: result.get("actual_charge_times", []),
            ATTR_ACTUAL_CHARGE_PRICES: result.get("actual_charge_prices", []),
            ATTR_ACTUAL_DISCHARGE_TIMES: result.get("actual_discharge_times", []),
            ATTR_ACTUAL_DISCHARGE_PRICES: result.get("actual_discharge_prices", []),
            ATTR_NUM_WINDOWS: result.get("num_windows", 0),
            # Profit-based attributes (v1.2.0+)
            ATTR_CHARGE_PROFIT_PCT: result.get("charge_profit_pct", 0.0),
            ATTR_DISCHARGE_PROFIT_PCT: result.get("discharge_profit_pct", 0.0),
            ATTR_CHARGE_PROFIT_MET: result.get("charge_profit_met", False),
            ATTR_DISCHARGE_PROFIT_MET: result.get("discharge_profit_met", False),
            "spread_avg": result.get("spread_avg", 0.0),
            "arbitrage_avg": result.get("arbitrage_avg", 0.0),
            ATTR_AVG_CHEAP_PRICE: result.get("avg_cheap_price", 0.0),
            ATTR_AVG_EXPENSIVE_PRICE: result.get("avg_expensive_price", 0.0),
            ATTR_CURRENT_SELL_PRICE: result.get("current_sell_price", 0.0),
            ATTR_SELL_PRICE_COUNTRY: result.get("price_country", "none"),
            "sell_formula_param_a": result.get("sell_formula_param_a", 0.1),
            "sell_formula_param_b": result.get("sell_formula_param_b", 0.0),
            "expensive_sell_prices": result.get("expensive_sell_prices", []),
            "actual_discharge_sell_prices": result.get("actual_discharge_sell_prices", []),
            ATTR_PLANNED_TOTAL_COST: result.get("planned_total_cost", 0.0),
            ATTR_PLANNED_CHARGE_COST: result.get("planned_charge_cost", 0.0),
            "net_planned_charge_kwh": result.get("net_planned_charge_kwh", 0.0),
            "net_planned_discharge_kwh": result.get("net_planned_discharge_kwh", 0.0),
            "last_config_update": last_config_update.isoformat() if last_config_update else None,
            # === DASHBOARD HELPER ATTRIBUTES ===
            "grouped_charge_windows": result.get("grouped_charge_windows", []),
            "grouped_discharge_windows": result.get("grouped_discharge_windows", []),
            "percentile_cheap_avg": result.get("percentile_cheap_avg", 0),
            "percentile_expensive_avg": result.get("percentile_expensive_avg", 0),
            "percentile_expensive_sell_avg": result.get("percentile_expensive_sell_avg", 0),
            "cheap_half_avg": result.get("cheap_half_avg", 0),
            "expensive_half_avg": result.get("expensive_half_avg", 0),
            "day_avg_price": result.get("day_avg_price", 0),
            "net_grid_kwh": result.get("net_grid_kwh", 0),
            "baseline_cost": result.get("baseline_cost", 0),
            "estimated_savings": result.get("estimated_savings", 0),
            "true_savings": result.get("true_savings", 0),
            "gross_charged_kwh": result.get("gross_charged_kwh", 0),
            "gross_usable_kwh": result.get("gross_usable_kwh", 0),
            "gross_discharged_kwh": result.get("gross_discharged_kwh", 0),
            "actual_remaining_kwh": result.get("actual_remaining_kwh", 0),
            "net_post_discharge_eur_kwh": result.get("net_post_discharge_eur_kwh", 0),
            "battery_margin_eur_kwh": result.get("battery_margin_eur_kwh", 0),
            "battery_arbitrage_value": result.get("battery_arbitrage_value", 0),
            "actual_price_kwh": result.get("actual_price_kwh", 0),
            "cost_difference": result.get("cost_difference", 0),
            "charge_power_kw": result.get("charge_power_kw", 0),
            "discharge_power_kw": result.get("discharge_power_kw", 0),
            "base_usage_kw": result.get("base_usage_kw", 0),
            "base_usage_kwh": result.get("base_usage_kwh", 0),
            "base_usage_day_cost": result.get("base_usage_day_cost", 0),
            "window_duration_hours": result.get("window_duration_hours", 0.25),
            # === CHRONOLOGICAL BUFFER TRACKING ===
            "buffer_energy_kwh": result.get("buffer_energy_kwh", 0),
            "final_battery_state_kwh": result.get("final_battery_state_kwh", 0),
            "buffer_delta_kwh": result.get("buffer_delta_kwh", 0),
            "battery_capacity_kwh": result.get("battery_capacity_kwh", 100.0),
            "chrono_charge_kwh": result.get("chrono_charge_kwh", 0),
            "chrono_discharge_kwh": result.get("chrono_discharge_kwh", 0),
            "chrono_uncovered_base_kwh": result.get("chrono_uncovered_base_kwh", 0),
            "limit_discharge_to_buffer": result.get("limit_discharge_to_buffer", False),
            "discharge_buffer_limit_kwh": result.get("discharge_buffer_limit_kwh", 0.0),
            "skipped_discharge_windows": result.get("skipped_discharge_windows", []),
            "discharge_windows_limited": result.get("discharge_windows_limited", False),
            "feasibility_issues": result.get("feasibility_issues", []),
            "has_feasibility_issues": result.get("has_feasibility_issues", False),
            # Grid and battery state tracking (v1.2.0+) - tomorrow specific
            ATTR_GRID_KWH_ESTIMATED_TOMORROW: result.get("grid_kwh_estimated_today", 0.0),  # Reuse today's field
            ATTR_BATTERY_STATE_END_OF_TOMORROW: result.get("battery_state_end_of_day", 0.0),  # Reuse today's field
            # Solar integration metrics
            "solar_to_battery_kwh": result.get("solar_to_battery_kwh", 0.0),
            "solar_offset_base_kwh": result.get("solar_offset_base_kwh", 0.0),
            "solar_exported_kwh": result.get("solar_exported_kwh", 0.0),
            "solar_export_revenue": result.get("solar_export_revenue", 0.0),
            "solar_total_contribution_kwh": result.get("solar_total_contribution_kwh", 0.0),
            "grid_savings_from_solar": result.get("grid_savings_from_solar", 0.0),
            "expected_solar_kwh": result.get("expected_solar_kwh", 0.0),
            # Detailed battery tracking for energy flow report
            "battery_charged_from_grid_kwh": result.get("battery_charged_from_grid_kwh", 0.0),
            "battery_charged_from_grid_cost": result.get("battery_charged_from_grid_cost", 0.0),
            "battery_charged_from_solar_kwh": result.get("battery_charged_from_solar_kwh", 0.0),
            "battery_charged_avg_price": result.get("battery_charged_avg_price", 0.0),
            "battery_discharged_to_base_kwh": result.get("battery_discharged_to_base_kwh", 0.0),
            "battery_discharged_to_grid_kwh": result.get("battery_discharged_to_grid_kwh", 0.0),
            "battery_discharged_avg_price": result.get("battery_discharged_avg_price", 0.0),
            # RTE (Round-Trip Efficiency) loss tracking
            "rte_loss_kwh": result.get("rte_loss_kwh", 0.0),
            "rte_loss_value": result.get("rte_loss_value", 0.0),
            # RTE-aware discharge tracking
            "rte_preserved_kwh": result.get("rte_preserved_kwh", 0.0),
            "rte_preserved_periods": result.get("rte_preserved_periods", []),
            "rte_breakeven_price": result.get("rte_breakeven_price", 0.0),
        }


class CEWPriceSensorProxy(SensorEntity):
    """Proxy sensor that mirrors the configured price sensor.

    This allows the dashboard to use a consistent sensor entity_id
    regardless of which price sensor the user has configured.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: CEWCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the proxy sensor."""
        self.hass = hass
        self.coordinator = coordinator
        self.config_entry = config_entry

        self._attr_unique_id = f"{PREFIX}price_sensor_proxy"
        # Force consistent entity_id based on key, not display name
        self.entity_id = f"sensor.{PREFIX}price_sensor_proxy"
        self._attr_name = "CEW Price Sensor Proxy"
        self._attr_has_entity_name = False
        self._attr_native_value = None
        # Initialize with empty lists to prevent template errors during initial load
        self._attr_extra_state_attributes = {
            "raw_today": [],
            "raw_tomorrow": [],
            "calculated_today": [],
            "calculated_tomorrow": [],
            "calculated_sell_today": [],
            "calculated_sell_tomorrow": [],
            "tomorrow_valid": False,
        }

        # Calculation engine for price calculations
        self._calculation_engine = WindowCalculationEngine()

    @property
    def device_info(self):
        """Return device information."""
        return {
            "identifiers": {(DOMAIN, self.config_entry.entry_id)},
            "name": "Cheapest Energy Windows",
            "manufacturer": "Community",
            "model": "Energy Optimizer",
            "sw_version": VERSION,
        }

    @property
    def should_poll(self) -> bool:
        """No polling needed - updates come from coordinator."""
        return False

    def _detect_sensor_format(self, attributes):
        """Detect sensor format type."""
        if "raw_today" in attributes and "raw_tomorrow" in attributes:
            return "nordpool"
        elif "prices_today" in attributes or "prices_tomorrow" in attributes:
            return "entsoe"
        # Future: Add more formats here
        return None

    def _normalize_entsoe_to_nordpool(self, attributes):
        """Convert ENTSO-E format to Nord Pool format."""
        from datetime import timedelta
        normalized = {}

        # Convert prices_today to raw_today
        if "prices_today" in attributes and attributes["prices_today"]:
            raw_today = []
            for item in attributes["prices_today"]:
                time_str = item.get("time", "")
                parsed = dt_util.parse_datetime(time_str)
                if parsed:
                    # Convert UTC to local timezone
                    local_time = dt_util.as_local(parsed)
                    end_time = local_time + timedelta(minutes=15)
                    raw_today.append({
                        "start": local_time.isoformat(),
                        "end": end_time.isoformat(),
                        "value": item.get("price", 0)
                    })
            normalized["raw_today"] = raw_today
        else:
            normalized["raw_today"] = []

        # Convert prices_tomorrow to raw_tomorrow
        if "prices_tomorrow" in attributes and attributes["prices_tomorrow"]:
            raw_tomorrow = []
            for item in attributes["prices_tomorrow"]:
                time_str = item.get("time", "")
                parsed = dt_util.parse_datetime(time_str)
                if parsed:
                    # Convert UTC to local timezone
                    local_time = dt_util.as_local(parsed)
                    end_time = local_time + timedelta(minutes=15)
                    raw_tomorrow.append({
                        "start": local_time.isoformat(),
                        "end": end_time.isoformat(),
                        "value": item.get("price", 0)
                    })
            normalized["raw_tomorrow"] = raw_tomorrow
            normalized["tomorrow_valid"] = True
        else:
            normalized["raw_tomorrow"] = []
            normalized["tomorrow_valid"] = False

        # Pass through other attributes we might need
        for key, value in attributes.items():
            if key not in ["prices_today", "prices_tomorrow", "prices", "raw_prices"]:
                normalized[key] = value

        return normalized

    def _calculate_prices(self, raw_prices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate buy prices using the configured formula.

        Args:
            raw_prices: List of raw price data with 'start', 'end', 'value' keys

        Returns:
            List of calculated prices with same structure as input
        """
        if not raw_prices:
            return []

        # Get current config from coordinator
        config = self.coordinator.data.get("config", {}) if self.coordinator.data else {}

        # Extract parameters needed by _calculate_buy_price
        country = config.get("price_country", "netherlands")
        param_a = config.get("buy_formula_param_a", 0.1)
        param_b = config.get("buy_formula_param_b", 0.0)
        vat = config.get("vat", 21)
        tax = config.get("tax", 0.12286)
        additional_cost = config.get("additional_cost", 0.02398)

        calculated = []
        for item in raw_prices:
            spot_price = item.get("value", 0)

            # Use calculation engine's buy price method with all required parameters
            calculated_price = self._calculation_engine._calculate_buy_price(
                spot_price, country, param_a, param_b, vat, tax, additional_cost
            )

            calculated.append({
                "start": item.get("start"),
                "end": item.get("end"),
                "value": calculated_price
            })

        return calculated

    def _calculate_sell_prices(self, raw_prices: List[Dict[str, Any]], buy_prices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate sell prices using the configured formula.

        Args:
            raw_prices: List of raw spot price data
            buy_prices: List of calculated buy prices (needed for sell calculation)

        Returns:
            List of calculated sell prices with same structure
        """
        if not raw_prices or not buy_prices:
            return []

        # Get current config from coordinator
        config = self.coordinator.data.get("config", {}) if self.coordinator.data else {}

        # Extract parameters needed by _calculate_sell_price
        sell_country = config.get("price_country", "netherlands")
        sell_param_a = config.get("sell_formula_param_a", 0.1)
        sell_param_b = config.get("sell_formula_param_b", 0.0)

        calculated = []
        for idx, item in enumerate(raw_prices):
            if idx < len(buy_prices):
                spot_price = item.get("value", 0)
                buy_price = buy_prices[idx].get("value", 0)

                # Use calculation engine's sell price method
                calculated_sell_price = self._calculation_engine._calculate_sell_price(
                    spot_price, buy_price, sell_country, sell_param_a, sell_param_b
                )

                calculated.append({
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "value": calculated_sell_price
                })

        return calculated

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        if not self.coordinator.data:
            return

        # Get the configured price sensor entity_id
        price_sensor_entity = self.hass.states.get(f"text.{PREFIX}price_sensor_entity")
        if not price_sensor_entity:
            _LOGGER.warning("Price sensor entity text input not found")
            return

        price_sensor_id = price_sensor_entity.state
        if not price_sensor_id or price_sensor_id == "":
            _LOGGER.warning("Price sensor entity not configured")
            return

        # Get the actual price sensor state
        price_sensor = self.hass.states.get(price_sensor_id)
        if not price_sensor:
            _LOGGER.warning(f"Configured price sensor {price_sensor_id} not found")
            self._attr_native_value = STATE_UNAVAILABLE
            self.async_write_ha_state()
            return

        # Mirror the state
        self._attr_native_value = price_sensor.state

        # Detect format and normalize if needed
        sensor_format = self._detect_sensor_format(price_sensor.attributes)

        if sensor_format == "entsoe":
            _LOGGER.debug(f"Detected ENTSO-E format from {price_sensor_id}, normalizing to Nord Pool format")
            self._attr_extra_state_attributes = self._normalize_entsoe_to_nordpool(price_sensor.attributes)
        elif sensor_format == "nordpool":
            _LOGGER.debug(f"Detected Nord Pool format from {price_sensor_id}, passing through")
            self._attr_extra_state_attributes = dict(price_sensor.attributes)
        else:
            _LOGGER.warning(f"Unknown price sensor format from {price_sensor_id}, passing through as-is")
            self._attr_extra_state_attributes = dict(price_sensor.attributes)

        # Add calculated prices using the configured formula
        # Use `or []` to handle both missing keys AND None values
        raw_today = self._attr_extra_state_attributes.get("raw_today") or []
        raw_tomorrow = self._attr_extra_state_attributes.get("raw_tomorrow") or []

        calculated_buy_today = self._calculate_prices(raw_today)
        calculated_buy_tomorrow = self._calculate_prices(raw_tomorrow)

        self._attr_extra_state_attributes["calculated_today"] = calculated_buy_today
        self._attr_extra_state_attributes["calculated_tomorrow"] = calculated_buy_tomorrow

        # Calculate sell prices (uses buy prices as input)
        calculated_sell_today = self._calculate_sell_prices(raw_today, calculated_buy_today)
        calculated_sell_tomorrow = self._calculate_sell_prices(raw_tomorrow, calculated_buy_tomorrow)

        self._attr_extra_state_attributes["calculated_sell_today"] = calculated_sell_today
        self._attr_extra_state_attributes["calculated_sell_tomorrow"] = calculated_sell_tomorrow

        # Add formula info for dashboard display using the formula registry
        config = self.coordinator.data.get("config", {}) if self.coordinator.data else {}
        country = config.get("price_country", DEFAULT_PRICE_COUNTRY)
        formula = get_formula(country)

        self._attr_extra_state_attributes["price_country"] = country
        if formula:
            self._attr_extra_state_attributes["price_country_display"] = formula.name
            self._attr_extra_state_attributes["buy_formula_description"] = f"Buy: {formula.buy_formula_description}"
            self._attr_extra_state_attributes["sell_formula_description"] = f"Sell: {formula.sell_formula_description}"
            self._attr_extra_state_attributes["active_params"] = [p.key for p in formula.params]
            self._attr_extra_state_attributes["buy_equals_sell"] = (formula.sell_formula_description.lower() == "same as buy price")
        else:
            # Fallback for unknown country
            self._attr_extra_state_attributes["price_country_display"] = country
            self._attr_extra_state_attributes["buy_formula_description"] = "Buy: spot price"
            self._attr_extra_state_attributes["sell_formula_description"] = "Sell: spot price"
            self._attr_extra_state_attributes["active_params"] = []
            self._attr_extra_state_attributes["buy_equals_sell"] = True

        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        # Subscribe to coordinator updates
        self.async_on_remove(
            self.coordinator.async_add_listener(self._handle_coordinator_update)
        )

        # Do initial update
        self._handle_coordinator_update()


class CEWLastCalculationSensor(CoordinatorEntity, SensorEntity):
    """Sensor that tracks calculation updates with unique state values.

    This sensor generates a unique random value on every coordinator update
    to trigger chart refreshes via a hidden series in the dashboard.
    Using random values ensures state changes are always detected,
    even with rapid consecutive updates.
    """

    def __init__(
        self,
        coordinator: CEWCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.config_entry = config_entry
        self._attr_unique_id = f"{PREFIX}last_calculation"
        # Force consistent entity_id based on key, not display name
        self.entity_id = f"sensor.{PREFIX}last_calculation"
        self._attr_name = "CEW Last Calculation"
        self._attr_has_entity_name = False
        self._attr_icon = "mdi:refresh"

        # Initialize with random value
        self._attr_native_value = str(uuid.uuid4())[:8]

    @property
    def device_info(self):
        """Return device information."""
        return {
            "identifiers": {(DOMAIN, self.config_entry.entry_id)},
            "name": "Cheapest Energy Windows",
            "manufacturer": "Community",
            "model": "Energy Optimizer",
            "sw_version": VERSION,
        }

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        if not self.coordinator.data:
            return

        # Only update when calculations actually change
        # Coordinator polls every 10s for state transitions, but this sensor
        # only updates when price data changes or config changes to avoid
        # unnecessary chart refreshes
        price_data_changed = self.coordinator.data.get("price_data_changed", False)
        config_changed = self.coordinator.data.get("config_changed", False)

        if price_data_changed or config_changed:
            # Actual calculation occurred - generate new unique value
            self._attr_native_value = str(uuid.uuid4())[:8]
            self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        # Subscribe to coordinator updates
        self.async_on_remove(
            self.coordinator.async_add_listener(self._handle_coordinator_update)
        )

        # Do initial update
        self._handle_coordinator_update()