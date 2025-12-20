"""Time entities for Cheapest Energy Windows."""
from __future__ import annotations

import logging
from datetime import time

from homeassistant.components.time import TimeEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CALCULATION_AFFECTING_KEYS,
    DOMAIN,
    LOGGER_NAME,
    PREFIX,
    VERSION,
    DEFAULT_QUIET_START,
    DEFAULT_QUIET_END,
    DEFAULT_TIME_OVERRIDE_START,
    DEFAULT_TIME_OVERRIDE_END,
    DEFAULT_CALCULATION_WINDOW_START,
    DEFAULT_CALCULATION_WINDOW_END,
    DEFAULT_SOLAR_WINDOW_START,
    DEFAULT_SOLAR_WINDOW_END,
)

_LOGGER = logging.getLogger(LOGGER_NAME)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Cheapest Energy Windows time entities."""

    times = []

    # Define all time entities
    time_configs = [
        ("time_override_start", "Time Override Start", DEFAULT_TIME_OVERRIDE_START, "mdi:clock-start"),
        ("time_override_end", "Time Override End", DEFAULT_TIME_OVERRIDE_END, "mdi:clock-end"),
        ("time_override_start_tomorrow", "Time Override Start Tomorrow", DEFAULT_TIME_OVERRIDE_START, "mdi:clock-start"),
        ("time_override_end_tomorrow", "Time Override End Tomorrow", DEFAULT_TIME_OVERRIDE_END, "mdi:clock-end"),
        ("calculation_window_start", "Calculation Window Start", DEFAULT_CALCULATION_WINDOW_START, "mdi:window-closed-variant"),
        ("calculation_window_end", "Calculation Window End", DEFAULT_CALCULATION_WINDOW_END, "mdi:window-closed-variant"),
        ("calculation_window_start_tomorrow", "Calculation Window Start Tomorrow", DEFAULT_CALCULATION_WINDOW_START, "mdi:window-open-variant"),
        ("calculation_window_end_tomorrow", "Calculation Window End Tomorrow", DEFAULT_CALCULATION_WINDOW_END, "mdi:window-open-variant"),
        ("quiet_hours_start", "Quiet Hours Start", DEFAULT_QUIET_START, "mdi:volume-off"),
        ("quiet_hours_end", "Quiet Hours End", DEFAULT_QUIET_END, "mdi:volume-high"),
        # Solar production window
        ("solar_window_start", "Solar Window Start", DEFAULT_SOLAR_WINDOW_START, "mdi:weather-sunset-up"),
        ("solar_window_end", "Solar Window End", DEFAULT_SOLAR_WINDOW_END, "mdi:weather-sunset-down"),
    ]

    for key, name, default, icon in time_configs:
        times.append(
            CEWTime(hass, config_entry, key, name, default, icon)
        )

    async_add_entities(times)


class CEWTime(TimeEntity):
    """Representation of a CEW time entity."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        key: str,
        name: str,
        default: str,
        icon: str,
    ) -> None:
        """Initialize the time entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._key = key
        self._attr_name = f"CEW {name}"
        self._attr_unique_id = f"{PREFIX}{key}"
        # Force consistent entity_id based on key, not display name
        self.entity_id = f"time.{PREFIX}{key}"
        self._attr_icon = icon
        self._attr_has_entity_name = False

        # Load value from config entry options, with fallback to data for backwards compatibility
        # (values may be in data for existing installations that haven't been migrated)
        time_str = config_entry.options.get(
            key,
            config_entry.data.get(key, default)
        )
        self._attr_native_value = self._parse_time(time_str)

    def _parse_time(self, time_str: str) -> time | None:
        """Parse time string to time object."""
        if not time_str:
            return None
        try:
            parts = time_str.split(":")
            return time(int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return None

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

    async def async_set_value(self, value: time) -> None:
        """Set the time value."""
        self._attr_native_value = value

        # Save to config entry options as string
        time_str = value.strftime("%H:%M:%S") if value else ""
        new_options = dict(self._config_entry.options)
        new_options[self._key] = time_str
        self.hass.config_entries.async_update_entry(
            self._config_entry,
            options=new_options
        )

        self.async_write_ha_state()

        # Only trigger coordinator update for times that affect calculations
        # Check against the centralized registry of calculation-affecting keys
        if self._key in CALCULATION_AFFECTING_KEYS:
            if DOMAIN in self.hass.data and self._config_entry.entry_id in self.hass.data[DOMAIN]:
                coordinator = self.hass.data[DOMAIN][self._config_entry.entry_id].get("coordinator")
                if coordinator:
                    _LOGGER.debug(f"Time {self._key} affects calculations, triggering coordinator refresh")
                    await coordinator.async_request_refresh()
        else:
            _LOGGER.debug(f"Time {self._key} is UI/notification only (quiet hours), skipping coordinator refresh")