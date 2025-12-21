"""Automation handler for Cheapest Energy Windows."""
from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event

from .const import (
    LOGGER_NAME,
    PREFIX,
)

_LOGGER = logging.getLogger(LOGGER_NAME)


async def async_setup_automation(hass: HomeAssistant) -> "AutomationHandler":
    """Set up automation handler."""
    handler = AutomationHandler(hass)
    await handler.async_setup()
    return handler


class AutomationHandler:
    """Handles automations for Cheapest Energy Windows."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the automation handler."""
        self.hass = hass
        self._state_listener = None

    async def async_setup(self) -> None:
        """Set up automation handlers."""
        # Set up state change listener (for logging and filtering only)
        await self._setup_state_listener()

        _LOGGER.info("Automation handlers set up successfully")

    async def async_shutdown(self) -> None:
        """Shut down automation handlers."""
        if self._state_listener:
            self._state_listener()
            self._state_listener = None

        _LOGGER.info("Automation handlers shut down")

    async def _setup_state_listener(self) -> None:
        """Set up state change listener for the cew_today sensor."""
        @callback
        async def state_changed(event):
            """Handle state change events."""
            new_state = event.data.get("new_state")
            if not new_state:
                return

            # Filter out invalid states
            if new_state.state in ["unknown", "unavailable"]:
                return

            # Get old state
            old_state = event.data.get("old_state")
            old_state_value = old_state.state if old_state else None

            # Skip if state hasn't actually changed
            if old_state_value == new_state.state:
                return

            # Log the state change
            _LOGGER.debug(
                f"CEW state changed from {old_state_value} to {new_state.state}"
            )

        # Subscribe to state changes
        self._state_listener = async_track_state_change_event(
            self.hass,
            f"sensor.{PREFIX}today",
            state_changed
        )

        _LOGGER.debug("State change listener registered")