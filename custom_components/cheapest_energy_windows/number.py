"""Number entities for Cheapest Energy Windows."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity import EntityCategory

from .const import (
    CALCULATION_AFFECTING_KEYS,
    DOMAIN,
    LOGGER_NAME,
    PREFIX,
    VERSION,
    DEFAULT_CHARGING_WINDOWS,
    DEFAULT_EXPENSIVE_WINDOWS,
    DEFAULT_CHEAP_PERCENTILE,
    DEFAULT_EXPENSIVE_PERCENTILE,
    DEFAULT_MIN_SPREAD,
    DEFAULT_MIN_SPREAD_DISCHARGE,
    DEFAULT_AGGRESSIVE_DISCHARGE_SPREAD,
    DEFAULT_MIN_PRICE_DIFFERENCE,
    DEFAULT_ADDITIONAL_COST,
    DEFAULT_TAX,
    DEFAULT_VAT_RATE,
    DEFAULT_BASE_USAGE,
    DEFAULT_BATTERY_RTE,
    DEFAULT_CHARGE_POWER,
    DEFAULT_DISCHARGE_POWER,
    DEFAULT_BATTERY_MIN_SOC_DISCHARGE,
    DEFAULT_BATTERY_MIN_SOC_AGGRESSIVE_DISCHARGE,
    DEFAULT_PRICE_OVERRIDE_THRESHOLD,
)

_LOGGER = logging.getLogger(LOGGER_NAME)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Cheapest Energy Windows number entities."""

    numbers = []

    # Today's configuration
    numbers.extend([
        CEWNumber(
            hass, config_entry, "charging_windows", "Charging Windows",
            0, 96, DEFAULT_CHARGING_WINDOWS, 1, "windows",
            "mdi:window-open", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "expensive_windows", "Expensive Windows",
            0, 96, DEFAULT_EXPENSIVE_WINDOWS, 1, "windows",
            "mdi:window-closed", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "cheap_percentile", "Cheap Percentile",
            1, 50, DEFAULT_CHEAP_PERCENTILE, 1, "%",
            "mdi:percent", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "expensive_percentile", "Expensive Percentile",
            1, 50, DEFAULT_EXPENSIVE_PERCENTILE, 1, "%",
            "mdi:percent", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "min_spread", "Min Spread",
            0, 200, DEFAULT_MIN_SPREAD, 1, "%",
            "mdi:arrow-expand-horizontal", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "min_spread_discharge", "Min Spread Discharge",
            0, 200, DEFAULT_MIN_SPREAD_DISCHARGE, 1, "%",
            "mdi:arrow-expand-horizontal", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "aggressive_discharge_spread", "Aggressive Discharge Spread",
            0, 300, DEFAULT_AGGRESSIVE_DISCHARGE_SPREAD, 1, "%",
            "mdi:arrow-expand-horizontal", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "min_price_difference", "Min Price Difference",
            0, 0.5, DEFAULT_MIN_PRICE_DIFFERENCE, 0.01, "EUR/kWh",
            "mdi:cash-minus", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "additional_cost", "Additional Cost",
            0, 0.5, DEFAULT_ADDITIONAL_COST, 0.01, "EUR/kWh",
            "mdi:cash-plus", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "tax", "Tax",
            0, 0.5, DEFAULT_TAX, 0.01, "EUR/kWh",
            "mdi:cash-100", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "vat", "VAT",
            0, 50, DEFAULT_VAT_RATE, 0.1, "%",
            "mdi:percent", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "battery_rte", "Battery RTE",
            50, 100, DEFAULT_BATTERY_RTE, 1, "%",
            "mdi:battery-sync", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "charge_power", "Charge Power",
            0, 10000, DEFAULT_CHARGE_POWER, 100, "W",
            "mdi:lightning-bolt", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "discharge_power", "Discharge Power",
            0, 10000, DEFAULT_DISCHARGE_POWER, 100, "W",
            "mdi:lightning-bolt", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "base_usage", "Base Usage",
            0, 5000, DEFAULT_BASE_USAGE, 100, "W",
            "mdi:home-lightning-bolt", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "price_override_threshold", "Price Override Threshold",
            0, 0.5, DEFAULT_PRICE_OVERRIDE_THRESHOLD, 0.01, "EUR/kWh",
            "mdi:cash-lock", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "battery_min_soc_discharge",
            "Battery Min SOC Discharge",
            0, 100, DEFAULT_BATTERY_MIN_SOC_DISCHARGE, 1, "%",
            "mdi:battery-low", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "battery_min_soc_aggressive_discharge",
            "Battery Min SOC Aggressive Discharge",
            0, 100, DEFAULT_BATTERY_MIN_SOC_AGGRESSIVE_DISCHARGE, 1, "%",
            "mdi:battery-alert", NumberMode.BOX
        ),
    ])

    # Tomorrow's configuration
    tomorrow_configs = [
        ("charging_windows_tomorrow", "Charging Windows Tomorrow", DEFAULT_CHARGING_WINDOWS, 96, "windows"),
        ("expensive_windows_tomorrow", "Expensive Windows Tomorrow", DEFAULT_EXPENSIVE_WINDOWS, 96, "windows"),
        ("cheap_percentile_tomorrow", "Cheap Percentile Tomorrow", DEFAULT_CHEAP_PERCENTILE, 50, "%"),
        ("expensive_percentile_tomorrow", "Expensive Percentile Tomorrow", DEFAULT_EXPENSIVE_PERCENTILE, 50, "%"),
        ("min_spread_tomorrow", "Min Spread Tomorrow", DEFAULT_MIN_SPREAD, 200, "%"),
        ("min_spread_discharge_tomorrow", "Min Spread Discharge Tomorrow", DEFAULT_MIN_SPREAD_DISCHARGE, 200, "%"),
        ("aggressive_discharge_spread_tomorrow", "Aggressive Discharge Spread Tomorrow", DEFAULT_AGGRESSIVE_DISCHARGE_SPREAD, 300, "%"),
        ("min_price_difference_tomorrow", "Min Price Difference Tomorrow", DEFAULT_MIN_PRICE_DIFFERENCE, 0.5, "EUR/kWh"),
        ("price_override_threshold_tomorrow", "Price Override Threshold Tomorrow", DEFAULT_PRICE_OVERRIDE_THRESHOLD, 0.5, "EUR/kWh"),
    ]

    for key, name, default, max_val, unit in tomorrow_configs:
        min_val = 1 if "percentile" in key else 0 if "windows" in key else 0
        step = 1 if "%" in unit or "windows" in unit else 0.01
        numbers.append(
            CEWNumber(
                hass, config_entry, key, name,
                min_val, max_val, default, step, unit,
                "mdi:calendar-clock", NumberMode.BOX
            )
        )

    async_add_entities(numbers)


class CEWNumber(NumberEntity):
    """Representation of a CEW number entity."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        key: str,
        name: str,
        min_value: float,
        max_value: float,
        initial_value: float,
        step: float,
        unit: str,
        icon: str,
        mode: NumberMode,
    ) -> None:
        """Initialize the number entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._key = key
        self._attr_name = f"CEW {name}"
        self._attr_unique_id = f"{PREFIX}{key}"
        self._attr_native_min_value = min_value
        self._attr_native_max_value = max_value
        self._attr_native_step = step
        self._attr_native_unit_of_measurement = unit
        self._attr_icon = icon
        self._attr_mode = mode
        self._attr_has_entity_name = False

        # Load value from config entry options, fallback to initial
        self._attr_native_value = config_entry.options.get(key, initial_value)

    @property
    def entity_registry_enabled_default(self) -> bool:
        """Return if the entity should be enabled when first added to the entity registry."""
        return True

    @property
    def device_info(self):
        """Return device information."""
        return {
            "identifiers": {(DOMAIN, self._config_entry.entry_id)},
            "name": "Cheapest Energy Windows",
            "manufacturer": "Community",
            "model": "Energy Optimizer",
            "sw_version": VERSION,
        }

    async def async_set_native_value(self, value: float) -> None:
        """Update the value."""
        self._attr_native_value = value

        # Save to config entry options
        new_options = dict(self._config_entry.options)
        new_options[self._key] = value
        self.hass.config_entries.async_update_entry(
            self._config_entry,
            options=new_options
        )

        self.async_write_ha_state()

        # Only trigger coordinator update for numbers that affect calculations
        # Check against the centralized registry of calculation-affecting keys
        if self._key in CALCULATION_AFFECTING_KEYS:
            if DOMAIN in self.hass.data and self._config_entry.entry_id in self.hass.data[DOMAIN]:
                coordinator = self.hass.data[DOMAIN][self._config_entry.entry_id].get("coordinator")
                if coordinator:
                    _LOGGER.debug(f"Number {self._key} affects calculations, triggering coordinator refresh")
                    await coordinator.async_request_refresh()
        else:
            _LOGGER.debug(f"Number {self._key} doesn't affect calculations, skipping coordinator refresh")