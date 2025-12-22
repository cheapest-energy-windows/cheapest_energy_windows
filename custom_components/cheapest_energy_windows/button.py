"""Button entities for Cheapest Energy Windows."""
from __future__ import annotations

import logging

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
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
    """Set up Cheapest Energy Windows button entities."""
    buttons = [
        CEWRecalculateButton(hass, config_entry),
    ]
    async_add_entities(buttons)


class CEWRecalculateButton(ButtonEntity):
    """Button to trigger manual recalculation/optimization."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the button entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._attr_name = "CEW Recalculate"
        self._attr_unique_id = f"{PREFIX}recalculate"
        self.entity_id = f"button.{PREFIX}recalculate"
        self._attr_icon = "mdi:refresh"
        self._attr_has_entity_name = False

        # Link to device
        self._attr_device_info = {
            "identifiers": {(DOMAIN, config_entry.entry_id)},
            "name": "Cheapest Energy Windows",
            "manufacturer": "Community",
            "model": "Energy Optimizer",
            "sw_version": VERSION,
        }

    async def async_press(self) -> None:
        """Handle button press - trigger recalculation."""
        _LOGGER.info("Manual recalculation triggered via button")

        # Set force recalculation flag that persists across coordinator refreshes
        self.hass.data[DOMAIN][self._config_entry.entry_id]["force_recalculation"] = True

        # Get the coordinator and trigger refresh
        coordinator = self.hass.data[DOMAIN][self._config_entry.entry_id]["coordinator"]
        await coordinator.async_request_refresh()
