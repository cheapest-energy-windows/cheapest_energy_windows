"""Text entities for Cheapest Energy Windows."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.text import TextEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CALCULATION_AFFECTING_KEYS,
    DOMAIN,
    LOGGER_NAME,
    PREFIX,
    VERSION,
)

_LOGGER = logging.getLogger(LOGGER_NAME)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Cheapest Energy Windows text entities."""

    texts = [
        CEWText(
            hass,
            config_entry,
            "price_sensor_entity",
            "Price Sensor Entity",
            config_entry.data.get("price_sensor", ""),
            "mdi:identifier",
            r"^sensor\.[a-z0-9_]+$"
        ),
        # Battery configuration entities (optional)
        CEWText(
            hass,
            config_entry,
            "battery_soc_sensor",
            "Battery SoC Sensor",
            "not_configured",
            "mdi:battery-50",
            None  # No pattern validation for optional entities
        ),
        CEWText(
            hass,
            config_entry,
            "battery_available_energy_sensor",
            "Battery Available Energy Sensor",
            "not_configured",
            "mdi:battery",
            None
        ),
        CEWText(
            hass,
            config_entry,
            "battery_daily_charge_sensor",
            "Battery Daily Charge Sensor",
            "not_configured",
            "mdi:lightning-bolt",
            None
        ),
        CEWText(
            hass,
            config_entry,
            "battery_daily_discharge_sensor",
            "Battery Daily Discharge Sensor",
            "not_configured",
            "mdi:lightning-bolt-outline",
            None
        ),
        CEWText(
            hass,
            config_entry,
            "battery_power_sensor",
            "Battery Power Sensor",
            "not_configured",
            "mdi:flash",
            None
        ),
        CEWText(
            hass,
            config_entry,
            "battery_system_name",
            "Battery System Name",
            "My Battery System",
            "mdi:battery",
            None
        ),
        # Battery operation automations
        CEWText(
            hass,
            config_entry,
            "battery_normal_action",
            "Battery Normal Action",
            config_entry.options.get("battery_normal_action", "not_configured"),
            "mdi:robot",
            r"^(automation\.|script\.|scene\.)[a-z0-9_]+$|^not_configured$"
        ),
        CEWText(
            hass,
            config_entry,
            "battery_charge_action",
            "Battery Charge Action",
            config_entry.options.get("battery_charge_action", "not_configured"),
            "mdi:robot",
            r"^(automation\.|script\.|scene\.)[a-z0-9_]+$|^not_configured$"
        ),
        CEWText(
            hass,
            config_entry,
            "battery_discharge_action",
            "Battery Discharge Action",
            config_entry.options.get("battery_discharge_action", "not_configured"),
            "mdi:robot",
            r"^(automation\.|script\.|scene\.)[a-z0-9_]+$|^not_configured$"
        ),
        CEWText(
            hass,
            config_entry,
            "battery_off_action",
            "Battery Off Action",
            config_entry.options.get("battery_off_action", "not_configured"),
            "mdi:robot",
            r"^(automation\.|script\.|scene\.)[a-z0-9_]+$|^not_configured$"
        ),
        # Solar forecast sensors (from Forecast.Solar integration)
        CEWText(
            hass,
            config_entry,
            "solar_forecast_sensor",
            "Solar Forecast Sensor (Today)",
            "not_configured",
            "mdi:solar-power",
            None
        ),
        CEWText(
            hass,
            config_entry,
            "solar_forecast_sensor_tomorrow",
            "Solar Forecast Sensor (Tomorrow)",
            "not_configured",
            "mdi:solar-power",
            None
        ),
    ]

    async_add_entities(texts)


class CEWText(TextEntity):
    """Representation of a CEW text entity."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        key: str,
        name: str,
        default: str,
        icon: str,
        pattern: str | None = None,
    ) -> None:
        """Initialize the text entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._key = key
        self._attr_name = f"CEW {name}"
        self._attr_unique_id = f"{PREFIX}{key}"
        # Force consistent entity_id based on key, not display name
        self.entity_id = f"text.{PREFIX}{key}"
        self._attr_icon = icon
        self._attr_pattern = pattern
        self._attr_has_entity_name = False
        self._attr_native_min = 1
        self._attr_native_max = 255

        # Load value from config entry options, fallback to default
        self._attr_native_value = config_entry.options.get(key, default)

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

    async def async_set_value(self, value: str) -> None:
        """Set the text value."""
        self._attr_native_value = value

        # Save to config entry options
        new_options = dict(self._config_entry.options)
        new_options[self._key] = value
        self.hass.config_entries.async_update_entry(
            self._config_entry,
            options=new_options
        )

        # Also update the main data if this is the price sensor
        if self._key == "price_sensor_entity":
            new_data = dict(self._config_entry.data)
            new_data["price_sensor"] = value
            self.hass.config_entries.async_update_entry(
                self._config_entry,
                data=new_data
            )

        self.async_write_ha_state()

        # Trigger coordinator update for calculation-affecting text entities
        if self._key in CALCULATION_AFFECTING_KEYS or self._key == "price_sensor_entity":
            if DOMAIN in self.hass.data and self._config_entry.entry_id in self.hass.data[DOMAIN]:
                coordinator = self.hass.data[DOMAIN][self._config_entry.entry_id].get("coordinator")
                if coordinator:
                    _LOGGER.debug(f"Text {self._key} affects calculations, triggering coordinator refresh")
                    await coordinator.async_request_refresh()