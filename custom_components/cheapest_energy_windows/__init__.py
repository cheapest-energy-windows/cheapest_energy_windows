"""The Cheapest Energy Windows integration."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, device_registry as dr

from .const import (
    DOMAIN,
    PLATFORMS,
    PREFIX,
    VERSION,
    LOGGER_NAME,
    EVENT_SETTINGS_ROTATED,
)
from .coordinator import CEWCoordinator
from .services import async_setup_services
from .automation_handler import async_setup_automation

_LOGGER = logging.getLogger(LOGGER_NAME)

# Config entry only integration
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

# Migration mapping for "idle" to "normal" rename
_IDLE_TO_NORMAL_MIGRATION = {
    "base_usage_idle_strategy": "base_usage_normal_strategy",
    "notify_idle": "notify_normal",
    "battery_idle_action": "battery_normal_action",
}


async def _async_migrate_idle_to_normal(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Migrate old 'idle' config keys to 'normal'."""
    needs_update = False
    new_options = dict(entry.options)
    new_data = dict(entry.data)

    # Check and migrate options
    for old_key, new_key in _IDLE_TO_NORMAL_MIGRATION.items():
        if old_key in new_options:
            new_options[new_key] = new_options.pop(old_key)
            needs_update = True
            _LOGGER.info(f"Migrating config option '{old_key}' → '{new_key}'")

    # Check and migrate data (for backward compatibility)
    for old_key, new_key in _IDLE_TO_NORMAL_MIGRATION.items():
        if old_key in new_data:
            new_data[new_key] = new_data.pop(old_key)
            needs_update = True
            _LOGGER.info(f"Migrating config data '{old_key}' → '{new_key}'")

    # Also migrate time_override_mode value from "idle" to "normal"
    if new_options.get("time_override_mode") == "idle":
        new_options["time_override_mode"] = "normal"
        needs_update = True
        _LOGGER.info("Migrating time_override_mode value 'idle' → 'normal'")
    if new_options.get("time_override_mode_tomorrow") == "idle":
        new_options["time_override_mode_tomorrow"] = "normal"
        needs_update = True
        _LOGGER.info("Migrating time_override_mode_tomorrow value 'idle' → 'normal'")

    if needs_update:
        hass.config_entries.async_update_entry(
            entry,
            options=new_options,
            data=new_data
        )
        _LOGGER.info("Configuration migration from 'idle' to 'normal' complete")

        # Create a persistent notification to warn users
        await hass.services.async_call(
            "persistent_notification",
            "create",
            {
                "title": "CEW: 'Idle' renamed to 'Normal'",
                "message": (
                    "⚠️ **Breaking Change**: The 'idle' state has been renamed to 'normal'.\n\n"
                    "Your configuration has been automatically migrated.\n\n"
                    "**Action Required:**\n"
                    "- Update any custom automations that trigger on `sensor.cew_today` state 'idle' → 'normal'\n"
                    "- Entity IDs have changed:\n"
                    "  - `switch.cew_notify_idle` → `switch.cew_notify_normal`\n"
                    "  - `text.cew_battery_idle_action` → `text.cew_battery_normal_action`\n"
                    "  - `select.cew_base_usage_idle_strategy` → `select.cew_base_usage_normal_strategy`\n\n"
                    "This notification will not appear again."
                ),
                "notification_id": "cew_idle_to_normal_migration"
            }
        )


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the Cheapest Energy Windows component."""
    # This is called when the component is set up through configuration.yaml
    # We only support config_flow, so we just return True
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Cheapest Energy Windows from a config entry."""
    _LOGGER.info("Setting up Cheapest Energy Windows integration")

    # Migrate old "idle" keys to "normal"
    await _async_migrate_idle_to_normal(hass, entry)

    # Store domain data
    hass.data.setdefault(DOMAIN, {})

    # Create device registry entry
    device_registry = dr.async_get(hass)
    device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={(DOMAIN, entry.entry_id)},
        manufacturer="Community",
        model="Energy Optimizer",
        name="Cheapest Energy Windows",
        sw_version=VERSION,
    )

    # Set up the coordinator for data fetching
    coordinator = CEWCoordinator(hass, entry)

    # Store coordinator BEFORE platforms so they can access it
    # Initialize per-day recalculation and window flags for reboot survival
    hass.data[DOMAIN][entry.entry_id] = {
        "coordinator": coordinator,
        # Per-day recalculation control flags
        "force_recalculation_today": False,
        "force_recalculation_tomorrow": False,
        # Flags to clear windows when auto-optimizer is toggled OFF
        "clear_windows_today": False,
        "clear_windows_tomorrow": False,
        # Storage for rotated windows from midnight service
        "rotated_windows_today": None,
    }

    # Set up platforms FIRST so entities exist
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # NOW do the first coordinator refresh after entities exist
    await coordinator.async_config_entry_first_refresh()

    # Set up services
    await async_setup_services(hass)

    # Update automation template (ensures users always have latest features/fixes)
    from .services import async_create_notification_automation
    try:
        success, message = await async_create_notification_automation(hass)
        if success:
            _LOGGER.info(f"Automation template updated: {message}")
        else:
            _LOGGER.warning(f"Automation template update failed: {message}")
    except Exception as e:
        _LOGGER.error(f"Error updating automation template: {e}")
        # Don't fail setup if automation update fails

    # Set up automation handler
    automation_handler = await async_setup_automation(hass)

    # Store automation handler for cleanup
    hass.data[DOMAIN][entry.entry_id]["automation_handler"] = automation_handler

    # Register update listener
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    _LOGGER.info("Integration setup complete")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading Cheapest Energy Windows integration")

    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        # Shut down automation handler
        automation_handler = hass.data[DOMAIN][entry.entry_id].get("automation_handler")
        if automation_handler:
            await automation_handler.async_shutdown()

        # Clean up domain data
        hass.data[DOMAIN].pop(entry.entry_id)

        # Clean up persistent coordinator state
        persistent_key = f"{DOMAIN}_{entry.entry_id}_price_state"
        if persistent_key in hass.data:
            hass.data.pop(persistent_key)

        # Clean up sensor persistent states
        for sensor_type in ["today", "tomorrow"]:
            sensor_key = f"{DOMAIN}_{entry.entry_id}_sensor_{sensor_type}_state"
            if sensor_key in hass.data:
                hass.data.pop(sensor_key)

        _LOGGER.info("Cleared persistent state")

        # Clean up services if this was the last instance
        if not hass.data[DOMAIN]:
            # Unregister services
            hass.services.async_remove(DOMAIN, "rotate_tomorrow_settings")
            hass.services.async_remove(DOMAIN, "trigger_battery_action")
            _LOGGER.info("Services unregistered successfully")

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the config entry."""
    _LOGGER.info("Reloading Cheapest Energy Windows integration")

    # Clear formula registry cache before unload
    from .formulas import clear_registry
    clear_registry()

    # Standard reload: unload + setup
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update options."""
    _LOGGER.debug("Options updated, checking if reload needed")

    # Only reload if critical settings changed that require entity recreation
    # Most config changes are handled by coordinator refresh without reload
    # This prevents entity destruction/recreation on every config change

    # Currently, we don't reload automatically - entities handle their own updates
    # Future: Add logic here to reload ONLY if price_sensor_entity changed
    pass


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.info(f"Migrating configuration from version {config_entry.version}")

    # No migrations needed yet for version 1
    if config_entry.version == 1:
        return True

    _LOGGER.error(f"Unknown config version {config_entry.version}")
    return False


