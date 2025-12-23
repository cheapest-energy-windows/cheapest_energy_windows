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
    DEFAULT_PRICE_COUNTRY,
    DEFAULT_SOLAR_PRIORITY_STRATEGY,
    SOLAR_PRIORITY_OPTIONS,
)
from .formulas import get_country_options, get_formula

_LOGGER = logging.getLogger(LOGGER_NAME)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Cheapest Energy Windows select entities."""

    selects = []

    # Get country options dynamically from formula registry
    # Returns list of (id, display_name) tuples
    country_options = get_country_options()
    country_display_names = [name for _, name in country_options]

    # Get default country display name
    default_formula = get_formula(DEFAULT_PRICE_COUNTRY)
    default_country_display = default_formula.name if default_formula else "Netherlands"

    # Define all select entities
    select_configs = [
        ("pricing_window_duration", "Pricing Window Duration", ["15_minutes", "1_hour"], "15_minutes", "mdi:timer"),
        ("time_override_mode", "Time Override Mode", ["idle", "charge", "discharge", "off"], "idle", "mdi:toggle-switch"),
        ("time_override_mode_tomorrow", "Time Override Mode Tomorrow", ["idle", "charge", "discharge", "off"], "idle", "mdi:toggle-switch"),
        ("base_usage_charge_strategy", "Base Usage: During Charging", ["grid_covers_both", "battery_covers_base"], "grid_covers_both", "mdi:battery-charging"),
        ("base_usage_idle_strategy", "Base Usage: During Idle", ["battery_covers_limited", "battery_covers", "grid_covers"], "battery_covers_limited", "mdi:home-lightning-bolt"),
        ("base_usage_discharge_strategy", "Base Usage: During Discharge", ["already_included", "subtract_base"], "subtract_base", "mdi:battery-arrow-down"),
        ("price_country", "Price Formula", country_display_names, default_country_display, "mdi:map-marker"),
        # Solar production priority strategy
        ("solar_priority_strategy", "Solar Priority", ["base_then_grid", "base_then_battery"], DEFAULT_SOLAR_PRIORITY_STRATEGY, "mdi:solar-power-variant"),
        # Auto-optimization strategy
        ("auto_optimize_strategy", "Auto Optimize Strategy", ["off", "minimize_cost", "maximize_value"], "off", "mdi:auto-fix"),
        ("auto_optimize_strategy_tomorrow", "Auto Optimize Strategy Tomorrow", ["off", "minimize_cost", "maximize_value"], "off", "mdi:auto-fix"),
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
        # Force consistent entity_id based on key, not display name
        self.entity_id = f"select.{PREFIX}{key}"
        self._attr_options = options
        self._attr_icon = icon
        self._attr_has_entity_name = False

        # Set translation key to enable HA's native option translations
        if key.startswith("base_usage_") or key == "solar_priority_strategy" or key.startswith("auto_optimize_strategy"):
            self._attr_translation_key = key

        # Load value from config entry options, with fallback to data for backwards compatibility
        # (values may be in data for existing installations that haven't been migrated)
        stored_value = config_entry.options.get(
            key,
            config_entry.data.get(key, default)
        )

        # For price_country, convert stored internal value to display name
        if key == "price_country":
            # Convert internal value (e.g., "netherlands") to display name (e.g., "Netherlands")
            formula = get_formula(stored_value)
            self._attr_current_option = formula.name if formula else stored_value
        else:
            self._attr_current_option = stored_value

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

    @property
    def state(self) -> str | None:
        """Return the entity state."""
        # For price_country, current_option is already the display name
        return self._attr_current_option

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        # For price_country, the option is a display name - store display name in entity,
        # but save internal value to config entry
        self._attr_current_option = option

        # Determine what value to save to config entry
        if self._key == "price_country":
            # Convert display name to internal value for storage
            # Look up internal ID from formula registry
            country_options = get_country_options()
            save_value = option  # Default to option if not found
            for country_id, display_name in country_options:
                if display_name == option:
                    save_value = country_id
                    break
        else:
            save_value = option

        # Save to config entry options
        new_options = dict(self._config_entry.options)
        new_options[self._key] = save_value
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