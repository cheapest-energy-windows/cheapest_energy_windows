"""Helper to fetch hourly statistics from HA recorder."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import LOGGER_NAME
from .energy_prefs_helper import get_energy_preferences

_LOGGER = logging.getLogger(LOGGER_NAME)


async def is_recorder_available(hass: HomeAssistant) -> bool:
    """Check if the recorder component is available."""
    return "recorder" in hass.config.components


async def fetch_hourly_statistics(
    hass: HomeAssistant,
    entity_id: str,
    start_time: datetime,
    end_time: datetime,
) -> Dict[int, float]:
    """
    Fetch hourly kWh deltas for a sensor.

    Uses 'sum' statistic type and calculates delta between consecutive hours
    (sensors are cumulative/always increasing).

    Args:
        hass: Home Assistant instance
        entity_id: The entity ID to fetch statistics for
        start_time: Start of the period
        end_time: End of the period

    Returns:
        Dict mapping hour (0-23) to kWh consumed in that hour
    """
    try:
        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.statistics import statistics_during_period

        # Convert to UTC timestamps for the API
        start_utc = dt_util.as_utc(start_time)
        end_utc = dt_util.as_utc(end_time)

        # Use recorder's executor for proper database access
        recorder = get_instance(hass)

        # Try direct call first (newer HA versions may handle threading internally)
        try:
            stats = statistics_during_period(
                hass,
                start_time=start_utc,
                end_time=end_utc,
                statistic_ids={entity_id},
                period="hour",
                units=None,
                types={"sum"},
            )
        except Exception as e:
            _LOGGER.warning("Direct call failed (%s), trying executor", e)
            # Fallback to executor job
            def _fetch_stats():
                return statistics_during_period(
                    hass,
                    start_time=start_utc,
                    end_time=end_utc,
                    statistic_ids={entity_id},
                    period="hour",
                    units=None,
                    types={"sum"},
                )
            stats = await recorder.async_add_executor_job(_fetch_stats)

        # Calculate deltas between consecutive hours
        hourly: Dict[int, float] = {}
        sensor_stats = stats.get(entity_id, [])

        if not sensor_stats:
            _LOGGER.debug("No statistics found for %s", entity_id)
            return hourly

        # Sort by start time
        sorted_stats = sorted(sensor_stats, key=lambda x: x["start"])

        prev_sum = None
        for stat in sorted_stats:
            current_sum = stat.get("sum")
            if current_sum is None:
                continue

            # Convert start time to local time for hour lookup
            # start can be either a float (Unix timestamp) or a datetime
            start = stat["start"]
            if isinstance(start, (int, float)):
                # Convert Unix timestamp to datetime
                start_dt = datetime.fromtimestamp(start, tz=dt_util.UTC)
            else:
                start_dt = start
            local_time = dt_util.as_local(start_dt)
            hour = local_time.hour

            if prev_sum is not None:
                delta = current_sum - prev_sum
                if delta >= 0:  # Sanity check - should never be negative
                    hourly[hour] = delta
                else:
                    _LOGGER.debug(
                        "Negative delta for %s at hour %d: %s -> %s",
                        entity_id, hour, prev_sum, current_sum
                    )

            prev_sum = current_sum

        return hourly

    except ImportError as e:
        _LOGGER.warning("Recorder statistics not available: %s", e)
        return {}
    except Exception as e:
        _LOGGER.exception("Failed to fetch statistics for %s: %s", entity_id, e)
        return {}


async def fetch_combined_hourly_statistics(
    hass: HomeAssistant,
    entity_ids: List[str],
    start_time: datetime,
    end_time: datetime,
) -> Dict[int, float]:
    """
    Fetch and combine hourly statistics for multiple entities.

    Energy Dashboard may have multiple sensors for the same category
    (e.g., 2 grid import sensors). Sum them together.

    Args:
        hass: Home Assistant instance
        entity_ids: List of entity IDs to fetch and combine
        start_time: Start of the period
        end_time: End of the period

    Returns:
        Dict mapping hour (0-23) to combined kWh consumed in that hour
    """
    combined: Dict[int, float] = {}

    for entity_id in entity_ids:
        hourly = await fetch_hourly_statistics(hass, entity_id, start_time, end_time)
        for hour, kwh in hourly.items():
            combined[hour] = combined.get(hour, 0) + kwh

    return combined


async def fetch_day_statistics(
    hass: HomeAssistant,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fetch all energy statistics for today from HA Energy Dashboard.

    Automatically reads configured sensors from Energy Dashboard
    instead of requiring manual entity configuration.

    Args:
        hass: Home Assistant instance
        config: CEW configuration dict with energy_use_* toggles

    Returns:
        Dict with hourly data for consumption, solar, and battery
    """
    now = dt_util.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    result: Dict[str, Any] = {
        "consumption_hourly": {},  # {hour: kwh_delta}
        "solar_hourly": {},
        "battery_charge_hourly": {},
        "battery_discharge_hourly": {},
        "stats_available": False,
        "current_hour": now.hour,
        "avg_hourly_consumption": 0.0,
        "avg_hourly_solar": 0.0,
        "avg_hourly_battery_charge": 0.0,
        "avg_hourly_battery_discharge": 0.0,
        # Discovered sensor names for dashboard feedback
        "consumption_sensor": "none",
        "solar_sensor": "none",
        "battery_sensor": "none",
    }

    # Check if any energy dashboard feature is enabled
    use_consumption = config.get("energy_use_consumption", False)
    use_solar = config.get("energy_use_solar", False)
    use_battery = config.get("energy_use_battery", False)

    if not (use_consumption or use_solar or use_battery):
        _LOGGER.debug("No energy dashboard features enabled")
        return result

    # Check if recorder is available
    if not await is_recorder_available(hass):
        _LOGGER.warning("Recorder not available for energy statistics")
        return result

    # Get configured sensors from Energy Dashboard
    energy_prefs = await get_energy_preferences(hass)
    if not energy_prefs:
        _LOGGER.debug("No Energy Dashboard preferences available")
        return result

    # Fetch consumption if enabled
    if use_consumption:
        sensors = energy_prefs.get("grid_consumption_sensors", [])
        if sensors:
            result["consumption_sensor"] = sensors[0] if len(sensors) == 1 else f"{len(sensors)} sensors"
            result["consumption_hourly"] = await fetch_combined_hourly_statistics(
                hass, sensors, today_start, now
            )
            if result["consumption_hourly"]:
                result["stats_available"] = True
                result["consumption_source"] = "today"
                # Calculate average for future hour estimates
                total = sum(result["consumption_hourly"].values())
                hours = len(result["consumption_hourly"])
                result["avg_hourly_consumption"] = total / hours if hours > 0 else 0
                _LOGGER.debug(
                    "Consumption: %d hours, avg %.3f kWh/hr",
                    hours, result["avg_hourly_consumption"]
                )
            else:
                # No data for today yet (midnight) - fetch yesterday for average
                yesterday_start = today_start - timedelta(days=1)
                yesterday_end = today_start  # End of yesterday
                yesterday_hourly = await fetch_combined_hourly_statistics(
                    hass, sensors, yesterday_start, yesterday_end
                )
                if yesterday_hourly:
                    result["stats_available"] = True
                    result["consumption_source"] = "yesterday"
                    total = sum(yesterday_hourly.values())
                    hours = len(yesterday_hourly)
                    result["avg_hourly_consumption"] = total / hours if hours > 0 else 0
                    _LOGGER.debug(
                        "Consumption (yesterday fallback): %d hours, avg %.3f kWh/hr",
                        hours, result["avg_hourly_consumption"]
                    )
        else:
            _LOGGER.info("Energy Dashboard consumption enabled but no grid sensor configured")

    # Fetch solar if enabled
    if use_solar:
        sensors = energy_prefs.get("solar_production_sensors", [])
        if sensors:
            result["solar_sensor"] = sensors[0] if len(sensors) == 1 else f"{len(sensors)} sensors"
            result["solar_hourly"] = await fetch_combined_hourly_statistics(
                hass, sensors, today_start, now
            )
            if result["solar_hourly"]:
                result["stats_available"] = True
                result["solar_source"] = "today"
                total = sum(result["solar_hourly"].values())
                hours = len(result["solar_hourly"])
                result["avg_hourly_solar"] = total / hours if hours > 0 else 0
                _LOGGER.debug(
                    "Solar: %d hours, avg %.3f kWh/hr",
                    hours, result["avg_hourly_solar"]
                )
            else:
                # No data for today yet (midnight) - fetch yesterday for average
                yesterday_start = today_start - timedelta(days=1)
                yesterday_end = today_start
                yesterday_hourly = await fetch_combined_hourly_statistics(
                    hass, sensors, yesterday_start, yesterday_end
                )
                if yesterday_hourly:
                    result["stats_available"] = True
                    result["solar_source"] = "yesterday"
                    total = sum(yesterday_hourly.values())
                    hours = len(yesterday_hourly)
                    result["avg_hourly_solar"] = total / hours if hours > 0 else 0
                    _LOGGER.debug(
                        "Solar (yesterday fallback): %d hours, avg %.3f kWh/hr",
                        hours, result["avg_hourly_solar"]
                    )
        else:
            _LOGGER.info("Energy Dashboard solar enabled but no solar sensor configured")

    # Fetch battery if enabled
    if use_battery:
        charge_sensors = energy_prefs.get("battery_charge_sensors", [])
        discharge_sensors = energy_prefs.get("battery_discharge_sensors", [])

        if charge_sensors or discharge_sensors:
            result["battery_sensor"] = "configured"
            yesterday_start = today_start - timedelta(days=1)
            yesterday_end = today_start

            if charge_sensors:
                result["battery_charge_hourly"] = await fetch_combined_hourly_statistics(
                    hass, charge_sensors, today_start, now
                )
                if result["battery_charge_hourly"]:
                    result["stats_available"] = True
                    result["battery_charge_source"] = "today"
                    total = sum(result["battery_charge_hourly"].values())
                    hours = len(result["battery_charge_hourly"])
                    result["avg_hourly_battery_charge"] = total / hours if hours > 0 else 0
                else:
                    # Fallback to yesterday
                    yesterday_hourly = await fetch_combined_hourly_statistics(
                        hass, charge_sensors, yesterday_start, yesterday_end
                    )
                    if yesterday_hourly:
                        result["stats_available"] = True
                        result["battery_charge_source"] = "yesterday"
                        total = sum(yesterday_hourly.values())
                        hours = len(yesterday_hourly)
                        result["avg_hourly_battery_charge"] = total / hours if hours > 0 else 0

            if discharge_sensors:
                result["battery_discharge_hourly"] = await fetch_combined_hourly_statistics(
                    hass, discharge_sensors, today_start, now
                )
                if result["battery_discharge_hourly"]:
                    result["stats_available"] = True
                    result["battery_discharge_source"] = "today"
                    total = sum(result["battery_discharge_hourly"].values())
                    hours = len(result["battery_discharge_hourly"])
                    result["avg_hourly_battery_discharge"] = total / hours if hours > 0 else 0
                else:
                    # Fallback to yesterday
                    yesterday_hourly = await fetch_combined_hourly_statistics(
                        hass, discharge_sensors, yesterday_start, yesterday_end
                    )
                    if yesterday_hourly:
                        result["stats_available"] = True
                        result["battery_discharge_source"] = "yesterday"
                        total = sum(yesterday_hourly.values())
                        hours = len(yesterday_hourly)
                        result["avg_hourly_battery_discharge"] = total / hours if hours > 0 else 0
        else:
            _LOGGER.info("Energy Dashboard battery enabled but no battery sensors configured")

    return result
