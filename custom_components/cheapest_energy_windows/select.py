"""Select entities for Cheapest Energy Windows."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.select import SelectEntity
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
    """Set up Cheapest Energy Windows select entities."""

    selects = []

    # Define all select entities
    select_configs = [
        ("pricing_window_duration", "Pricing Window Duration", ["15_minutes", "1_hour"], "15_minutes", "mdi:timer"),
        ("time_override_mode", "Time Override Mode", ["idle", "charge", "discharge", "discharge_aggressive", "off"], "idle", "mdi:toggle-switch"),
        ("time_override_mode_tomorrow", "Time Override Mode Tomorrow", ["idle", "charge", "discharge", "discharge_aggressive", "off"], "idle", "mdi:toggle-switch"),
        ("base_usage_charge_strategy", "Base Usage - Charge Strategy", ["grid_covers_both", "battery_covers_base"], "grid_covers_both", "mdi:battery-charging"),
        ("base_usage_idle_strategy", "Base Usage - Idle Strategy", ["grid_covers", "battery_covers"], "grid_covers", "mdi:home-lightning-bolt"),
        ("base_usage_discharge_strategy", "Base Usage - Discharge Strategy", ["already_included", "subtract_base"], "subtract_base", "mdi:battery-arrow-down"),
        ("base_usage_aggressive_strategy", "Base Usage - Aggressive Strategy", ["same_as_discharge", "already_included", "subtract_base"], "same_as_discharge", "mdi:battery-alert"),
    ]

    for key, name, options, default, icon in select_configs:
        selects.append(
            CEWSelect(hass, config_entry, key, name, options, default, icon)
        )

    async_add_entities(selects)


class CEWSelect(SelectEntity):
    """Representation of a CEW select entity."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        key: str,
        name: str,
        options: list[str],
        default: str,
        icon: str,
    ) -> None:
        """Initialize the select entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._key = key
        self._attr_name = f"CEW {name}"
        self._attr_unique_id = f"{PREFIX}{key}"
        self._attr_options = options
        self._attr_icon = icon
        self._attr_has_entity_name = False

        # Load value from config entry options, fallback to default
        self._attr_current_option = config_entry.options.get(key, default)
        if self._attr_current_option not in options:
            self._attr_current_option = default

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

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        self._attr_current_option = option

        # Save to config entry options
        new_options = dict(self._config_entry.options)
        new_options[self._key] = option
        self.hass.config_entries.async_update_entry(
            self._config_entry,
            options=new_options
        )

        self.async_write_ha_state()

        # Only trigger coordinator update for selects that affect calculations
        # Check against the centralized registry of calculation-affecting keys
        if self._key in CALCULATION_AFFECTING_KEYS:
            if DOMAIN in self.hass.data and self._config_entry.entry_id in self.hass.data[DOMAIN]:
                coordinator = self.hass.data[DOMAIN][self._config_entry.entry_id].get("coordinator")
                if coordinator:
                    _LOGGER.debug(f"Select {self._key} affects calculations, triggering coordinator refresh")
                    await coordinator.async_request_refresh()
        else:
            _LOGGER.debug(f"Select {self._key} doesn't affect calculations, skipping coordinator refresh")