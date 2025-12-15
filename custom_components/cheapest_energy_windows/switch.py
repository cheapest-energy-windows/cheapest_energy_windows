"""Switch entities for Cheapest Energy Windows."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.switch import SwitchEntity
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
    """Set up Cheapest Energy Windows switch entities."""

    switches = []

    # Define all boolean switches
    switch_configs = [
        ("automation_enabled", "Automation Enabled", True, "mdi:toggle-switch"),
        ("tomorrow_settings_enabled", "Tomorrow Settings Enabled", False, "mdi:calendar-clock"),
        ("midnight_rotation_notifications", "Midnight Rotation Notifications", False, "mdi:bell-ring"),
        ("notifications_enabled", "Notifications Enabled", True, "mdi:bell"),
        ("quiet_hours_enabled", "Quiet Hours Enabled", False, "mdi:volume-off"),
        ("price_override_enabled", "Price Override Enabled", False, "mdi:cash-lock"),
        ("price_override_enabled_tomorrow", "Price Override Enabled Tomorrow", False, "mdi:cash-lock-open"),
        ("time_override_enabled", "Time Override Enabled", False, "mdi:clock-edit"),
        ("time_override_enabled_tomorrow", "Time Override Enabled Tomorrow", False, "mdi:clock-edit"),
        ("calculation_window_enabled", "Calculation Window Enabled", False, "mdi:window-closed-variant"),
        ("calculation_window_enabled_tomorrow", "Calculation Window Enabled Tomorrow", False, "mdi:window-open-variant"),
        ("notify_automation_disabled", "Notify Automation Disabled", True, "mdi:bell-off"),
        ("notify_charging", "Notify Charging", True, "mdi:battery-charging"),
        ("notify_discharge", "Notify Discharge", True, "mdi:battery-arrow-up"),
        ("notify_discharge_aggressive", "Notify Discharge Aggressive", True, "mdi:battery-alert"),
        ("notify_idle", "Notify Idle", True, "mdi:battery"),
        ("notify_off", "Notify Off", True, "mdi:battery-off"),
        ("battery_use_soc_safety", "Battery Use SOC Safety", False, "mdi:shield-battery"),
        ("use_min_sell_price", "Use Minimum Sell Price", False, "mdi:cash-check"),
        ("min_sell_price_bypass_spread", "Min Sell Price Bypasses Spread", False, "mdi:skip-forward"),
        ("min_price_diff_enabled", "Min Buy Price Diff Enabled", True, "mdi:cash-check"),
        ("min_price_diff_enabled_tomorrow", "Min Buy Price Diff Enabled Tomorrow", True, "mdi:cash-check"),
        # Note: Arbitrage Protection removed in v1.2.0
        # Profit thresholds now naturally control window qualification
    ]

    for key, name, default, icon in switch_configs:
        switches.append(
            CEWSwitch(hass, config_entry, key, name, default, icon)
        )

    async_add_entities(switches)


class CEWSwitch(SwitchEntity):
    """Representation of a CEW switch entity."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        key: str,
        name: str,
        default: bool,
        icon: str,
    ) -> None:
        """Initialize the switch entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._key = key
        self._attr_name = f"CEW {name}"
        self._attr_unique_id = f"{PREFIX}{key}"
        # Force consistent entity_id based on key, not display name
        self.entity_id = f"switch.{PREFIX}{key}"
        self._attr_icon = icon
        self._attr_has_entity_name = False

        # Load value from config entry options, with fallback to data for backwards compatibility
        # (values may be in data for existing installations that haven't been migrated)
        self._attr_is_on = config_entry.options.get(
            key,
            config_entry.data.get(key, default)
        )

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

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the switch on."""
        self._attr_is_on = True
        await self._save_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the switch off."""
        self._attr_is_on = False
        await self._save_state()

    async def _save_state(self) -> None:
        """Save the state to config entry."""
        new_options = dict(self._config_entry.options)
        new_options[self._key] = self._attr_is_on

        _LOGGER.debug(f"Switch {self._key}: Saving state={self._attr_is_on} to config entry")
        _LOGGER.debug(f"Switch {self._key}: Before update, config_entry.options[{self._key}]={self._config_entry.options.get(self._key, 'NOT SET')}")

        self.hass.config_entries.async_update_entry(
            self._config_entry,
            options=new_options
        )

        _LOGGER.debug(f"Switch {self._key}: After update, config_entry.options[{self._key}]={self._config_entry.options.get(self._key, 'NOT SET')}")

        self.async_write_ha_state()

        # Only trigger coordinator refresh for switches that affect calculations
        # Check against the centralized registry of calculation-affecting keys
        if self._key in CALCULATION_AFFECTING_KEYS:
            if DOMAIN in self.hass.data and self._config_entry.entry_id in self.hass.data[DOMAIN]:
                coordinator = self.hass.data[DOMAIN][self._config_entry.entry_id].get("coordinator")
                if coordinator:
                    _LOGGER.debug(f"Switch {self._key} affects calculations, triggering coordinator refresh")
                    await coordinator.async_request_refresh()
        else:
            _LOGGER.debug(f"Switch {self._key} is UI/notification only, skipping coordinator refresh")