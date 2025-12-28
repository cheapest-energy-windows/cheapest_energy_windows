"""Helper to read HA Energy Dashboard configured sensors."""
from __future__ import annotations

import logging
from typing import Any, Dict

from homeassistant.core import HomeAssistant

from .const import LOGGER_NAME

_LOGGER = logging.getLogger(LOGGER_NAME)


async def get_energy_preferences(hass: HomeAssistant) -> Dict[str, Any]:
    """
    Get energy preferences from HA Energy Dashboard.

    Returns configured sensors for grid, solar, and battery.
    Uses the energy component's async_get_manager internally.

    Returns:
        Dict with discovered sensors:
        {
            "grid_consumption_sensors": ["sensor.xxx", ...],
            "grid_return_sensors": ["sensor.xxx", ...],
            "solar_production_sensors": ["sensor.xxx", ...],
            "battery_charge_sensors": ["sensor.xxx", ...],
            "battery_discharge_sensors": ["sensor.xxx", ...],
        }
    """
    result = {
        "grid_consumption_sensors": [],
        "grid_return_sensors": [],
        "solar_production_sensors": [],
        "battery_charge_sensors": [],
        "battery_discharge_sensors": [],
        # Solar forecast config entry
        "solar_forecast_config_entry": None,
    }

    try:
        # Import energy manager
        from homeassistant.components.energy import async_get_manager

        manager = await async_get_manager(hass)

        if not manager or not manager.data:
            _LOGGER.debug("No Energy Dashboard configuration found")
            return result

        # Parse energy sources
        energy_sources = manager.data.get("energy_sources", [])
        for source in energy_sources:
            source_type = source.get("type")

            if source_type == "grid":
                # Grid consumption (flow_from = import from grid)
                for flow in source.get("flow_from", []):
                    stat_id = flow.get("stat_energy_from")
                    if stat_id:
                        result["grid_consumption_sensors"].append(stat_id)
                # Grid return/export (flow_to = export to grid)
                for flow in source.get("flow_to", []):
                    stat_id = flow.get("stat_energy_to")
                    if stat_id:
                        result["grid_return_sensors"].append(stat_id)

            elif source_type == "solar":
                stat_id = source.get("stat_energy_from")
                if stat_id:
                    result["solar_production_sensors"].append(stat_id)
                # Solar forecast config entry
                forecast_entry = source.get("config_entry_solar_forecast")
                if forecast_entry and not result["solar_forecast_config_entry"]:
                    result["solar_forecast_config_entry"] = forecast_entry

            elif source_type == "battery":
                # Battery charge = energy TO battery (stat_energy_to)
                charge_id = source.get("stat_energy_to")
                if charge_id:
                    result["battery_charge_sensors"].append(charge_id)
                # Battery discharge = energy FROM battery (stat_energy_from)
                discharge_id = source.get("stat_energy_from")
                if discharge_id:
                    result["battery_discharge_sensors"].append(discharge_id)

        _LOGGER.debug(
            "Energy Dashboard sensors discovered: %d grid, %d solar, %d battery charge, %d battery discharge, forecast=%s",
            len(result["grid_consumption_sensors"]),
            len(result["solar_production_sensors"]),
            len(result["battery_charge_sensors"]),
            len(result["battery_discharge_sensors"]),
            result["solar_forecast_config_entry"],
        )

        return result

    except ImportError:
        _LOGGER.warning("Energy component not available")
        return result
    except Exception as e:
        _LOGGER.warning("Failed to get energy preferences: %s", e)
        return result
