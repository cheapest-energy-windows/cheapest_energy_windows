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
    DEFAULT_PERCENTILE_THRESHOLD,
    DEFAULT_MIN_PROFIT_CHARGE,
    DEFAULT_MIN_PROFIT_DISCHARGE,
    DEFAULT_MIN_PRICE_DIFFERENCE,
    DEFAULT_BASE_USAGE,
    DEFAULT_BATTERY_RTE,
    DEFAULT_CHARGE_POWER,
    DEFAULT_DISCHARGE_POWER,
    DEFAULT_DISCHARGE_BUFFER_LIMIT_KWH,
    DEFAULT_PRICE_OVERRIDE_THRESHOLD,
    DEFAULT_MIN_SELL_PRICE,
    DEFAULT_SELL_FORMULA_PARAM_A,
    DEFAULT_SELL_FORMULA_PARAM_B,
    DEFAULT_BUY_FORMULA_PARAM_A,
    DEFAULT_BUY_FORMULA_PARAM_B,
    DEFAULT_VAT_RATE,
    DEFAULT_TAX,
    DEFAULT_ADDITIONAL_COST,
    DEFAULT_PRICE_COUNTRY,
    # Buffer/chronological calculation defaults
    DEFAULT_BATTERY_BUFFER_KWH,
    DEFAULT_BATTERY_CAPACITY,
    # Solar production
    DEFAULT_EXPECTED_SOLAR_KWH,
)
from .formulas import get_formula, is_param_active

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
            hass, config_entry, "percentile_threshold", "Percentile Threshold",
            1, 50, DEFAULT_PERCENTILE_THRESHOLD, 1, "%",
            "mdi:percent", NumberMode.BOX
        ),
        # Profit thresholds
        CEWNumber(
            hass, config_entry, "min_profit_charge", "Min Profit Charge",
            -100, 200, DEFAULT_MIN_PROFIT_CHARGE, 1, "%",
            "mdi:percent-circle", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "min_profit_discharge", "Min Profit Discharge",
            -100, 200, DEFAULT_MIN_PROFIT_DISCHARGE, 1, "%",
            "mdi:percent-circle", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "min_price_difference", "Min Price Difference",
            0, 0.5, DEFAULT_MIN_PRICE_DIFFERENCE, 0.001, "EUR/kWh",
            "mdi:cash-minus", NumberMode.BOX
        ),
        # Formula parameters - availability depends on selected country
        # Belgium/Other: Uses param_a, param_b
        # Netherlands: Uses vat, tax, additional_cost
        CEWFormulaParamNumber(
            hass, config_entry, "buy_formula_param_a", "Cost (A)",
            -0.1, 0.5, DEFAULT_BUY_FORMULA_PARAM_A, 0.001, "EUR/kWh",
            "mdi:alpha-a-circle", NumberMode.BOX
        ),
        CEWFormulaParamNumber(
            hass, config_entry, "buy_formula_param_b", "Multiplier (B)",
            0.0, 2.0, DEFAULT_BUY_FORMULA_PARAM_B, 0.01, "",
            "mdi:alpha-b-circle", NumberMode.BOX
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
        # RTE-aware discharge margin (global battery setting)
        CEWNumber(
            hass, config_entry, "rte_discharge_margin", "RTE Discharge Margin",
            0, 50, 2, 1, "%",  # 0-50%, default 2%, step 1%
            "mdi:percent", NumberMode.BOX
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
            hass, config_entry, "discharge_buffer_limit_kwh",
            "Discharge Buffer Limit",
            0, 100, DEFAULT_DISCHARGE_BUFFER_LIMIT_KWH, 0.1, "kWh",
            "mdi:battery-lock", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "min_sell_price", "Minimum Sell Price",
            -0.5, 1.0, DEFAULT_MIN_SELL_PRICE, 0.01, "EUR/kWh",
            "mdi:cash-lock", NumberMode.BOX
        ),
        # Buffer/chronological calculation entities
        CEWNumber(
            hass, config_entry, "battery_buffer_kwh", "Battery Buffer Energy",
            0, 100, DEFAULT_BATTERY_BUFFER_KWH, 0.1, "kWh",
            "mdi:battery-plus", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "battery_capacity", "Battery Capacity",
            0.1, 200, DEFAULT_BATTERY_CAPACITY, 0.1, "kWh",
            "mdi:battery", NumberMode.BOX
        ),
        # Minimum usable battery threshold - below this level, battery is considered depleted
        CEWNumber(
            hass, config_entry, "battery_min_usable_kwh", "Battery Min Usable",
            0, 5, 0.2, 0.1, "kWh",
            "mdi:battery-low", NumberMode.BOX
        ),
        # Solar production entities
        CEWNumber(
            hass, config_entry, "expected_solar_kwh", "Expected Solar Today",
            0, 100, DEFAULT_EXPECTED_SOLAR_KWH, 0.1, "kWh",
            "mdi:solar-power", NumberMode.BOX
        ),
        CEWNumber(
            hass, config_entry, "expected_solar_kwh_tomorrow", "Expected Solar Tomorrow",
            0, 100, DEFAULT_EXPECTED_SOLAR_KWH, 0.1, "kWh",
            "mdi:solar-power", NumberMode.BOX
        ),
        # Sell formula parameters (same structure as buy for Belgium/Other)
        # Formula: SELL = (B × spot − A)
        CEWFormulaParamNumber(
            hass, config_entry, "sell_formula_param_a", "Sell Cost (A)",
            -0.1, 0.5, DEFAULT_SELL_FORMULA_PARAM_A, 0.001, "EUR/kWh",
            "mdi:alpha-a-circle", NumberMode.BOX
        ),
        CEWFormulaParamNumber(
            hass, config_entry, "sell_formula_param_b", "Sell Multiplier (B)",
            0.0, 2.0, DEFAULT_SELL_FORMULA_PARAM_B, 0.01, "",
            "mdi:alpha-b-circle", NumberMode.BOX
        ),
        # Netherlands formula parameters (VAT/tax/additional cost)
        # VAT is used by all countries, but tax/additional_cost are Netherlands-specific
        CEWFormulaParamNumber(
            hass, config_entry, "vat", "VAT Rate",
            0, 100, DEFAULT_VAT_RATE, 1, "%",
            "mdi:percent", NumberMode.BOX
        ),
        CEWFormulaParamNumber(
            hass, config_entry, "tax", "Energy Tax",
            0, 1.0, DEFAULT_TAX, 0.001, "EUR/kWh",
            "mdi:cash-plus", NumberMode.BOX
        ),
        CEWFormulaParamNumber(
            hass, config_entry, "additional_cost", "Additional Cost",
            0, 1.0, DEFAULT_ADDITIONAL_COST, 0.001, "EUR/kWh",
            "mdi:cash-plus", NumberMode.BOX
        ),

        # Auto-optimization minimum savings threshold
        CEWNumber(
            hass, config_entry, "min_daily_savings", "Min Daily Savings",
            0, 5.0, 0.50, 0.05, "EUR",
            "mdi:currency-eur", NumberMode.BOX
        ),
    ])

    # Tomorrow's configuration
    tomorrow_configs = [
        ("charging_windows_tomorrow", "Charging Windows Tomorrow", DEFAULT_CHARGING_WINDOWS, 96, "windows"),
        ("expensive_windows_tomorrow", "Expensive Windows Tomorrow", DEFAULT_EXPENSIVE_WINDOWS, 96, "windows"),
        ("percentile_threshold_tomorrow", "Percentile Threshold Tomorrow", DEFAULT_PERCENTILE_THRESHOLD, 50, "%"),
        ("min_profit_charge_tomorrow", "Min Profit Charge Tomorrow", DEFAULT_MIN_PROFIT_CHARGE, 200, "%"),
        ("min_profit_discharge_tomorrow", "Min Profit Discharge Tomorrow", DEFAULT_MIN_PROFIT_DISCHARGE, 200, "%"),
        ("price_override_threshold_tomorrow", "Price Override Threshold Tomorrow", DEFAULT_PRICE_OVERRIDE_THRESHOLD, 0.5, "EUR/kWh"),
        ("battery_buffer_kwh_tomorrow", "Battery Buffer Energy Tomorrow", DEFAULT_BATTERY_BUFFER_KWH, 100, "kWh"),
    ]

    for key, name, default, max_val, unit in tomorrow_configs:
        min_val = 1 if "percentile" in key else -100 if "min_profit" in key else 0 if "windows" in key else 0
        step = 1 if "%" in unit or "windows" in unit else 0.1 if "kWh" in unit else 0.001
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
        # Force consistent entity_id based on key, not name
        # This prevents entity_id changes when entity names are updated
        self.entity_id = f"number.{PREFIX}{key}"
        self._attr_native_min_value = min_value
        self._attr_native_max_value = max_value
        self._attr_native_step = step
        self._attr_native_unit_of_measurement = unit
        self._attr_icon = icon
        self._attr_mode = mode
        self._attr_has_entity_name = False

        # Load value from config entry options, with fallback to data for backwards compatibility
        # (values may be in data for existing installations that haven't been migrated)
        self._attr_native_value = config_entry.options.get(
            key,
            config_entry.data.get(key, initial_value)
        )

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


class CEWFormulaParamNumber(CEWNumber):
    """Number entity for formula parameters that checks availability based on country.

    This entity will show as unavailable when the current country's formula
    doesn't use this parameter.
    """

    @property
    def available(self) -> bool:
        """Return True if the parameter is used by the current country's formula."""
        country = self._config_entry.options.get(
            "price_country",
            self._config_entry.data.get("price_country", DEFAULT_PRICE_COUNTRY)
        )
        return is_param_active(country, self._key)