"""Helper to fetch hourly statistics from HA recorder."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import LOGGER_NAME
from .energy_prefs_helper import get_energy_preferences

_LOGGER = logging.getLogger(LOGGER_NAME)


def calculate_real_consumption(
    grid_import: Dict[int, float],
    grid_export: Dict[int, float],
    solar: Dict[int, float],
    battery_charge: Dict[int, float],
    battery_discharge: Dict[int, float],
) -> Dict[int, float]:
    """
    Calculate real household consumption from energy flows.

    Formula: real_consumption = solar + grid_import - grid_export + battery_discharge - battery_charge

    This calculates what the house actually consumed, regardless of energy source.
    - Solar produced goes to house consumption
    - Grid import goes to house consumption
    - Grid export means excess (subtract from consumption)
    - Battery discharge covers house consumption
    - Battery charge consumes from solar/grid (subtract from house consumption)

    Args:
        grid_import: Dict of hour -> kWh imported from grid
        grid_export: Dict of hour -> kWh exported to grid
        solar: Dict of hour -> kWh produced by solar
        battery_charge: Dict of hour -> kWh charged to battery
        battery_discharge: Dict of hour -> kWh discharged from battery

    Returns:
        Dict of hour -> kWh real household consumption
    """
    # Collect all hours that have any data
    all_hours = set()
    for hourly in [grid_import, grid_export, solar, battery_charge, battery_discharge]:
        all_hours.update(hourly.keys())

    real_consumption: Dict[int, float] = {}
    for hour in sorted(all_hours):
        consumption = (
            solar.get(hour, 0)
            + grid_import.get(hour, 0)
            - grid_export.get(hour, 0)
            + battery_discharge.get(hour, 0)
            - battery_charge.get(hour, 0)
        )
        # Consumption should never be negative (would indicate meter error)
        real_consumption[hour] = max(0, consumption)

    return real_consumption


def calculate_weighted_consumption_avg(
    today_avg: float,
    today_hours: int,
    yesterday_avg: float,
    day_before_avg: float,
) -> Tuple[float, str]:
    """
    Calculate weighted 72h average consumption.

    The weighting formula progressively increases today's weight as more hours pass:
    - weight_today = hours_with_data / 24 (0.0 to 1.0, grows linearly)
    - Remaining weight split: 70% yesterday, 30% day-before

    This provides a stable baseline at midnight (uses historical data) that
    self-corrects as the day progresses (today's data takes over).

    IMPORTANT: At midnight (today_hours < 1), we maintain continuity by using
    yesterday's full average as the primary source. This prevents a sudden jump
    when the day changes.

    Example weights by hour:
    - Hour 0 (no data yet): today=0%, yesterday=70%, day_before=30%
    - Hour 1:  today=4%, yesterday=67%, day_before=29%
    - Hour 12: today=54%, yesterday=32%, day_before=14%
    - Hour 23: today=100%, yesterday=0%, day_before=0%

    Args:
        today_avg: Average consumption (kWh/hr) for today's hours
        today_hours: Number of hours with data today (0-24)
        yesterday_avg: Average consumption (kWh/hr) for yesterday (full 24h)
        day_before_avg: Average consumption (kWh/hr) for day before (full 24h)

    Returns:
        Tuple of (weighted_avg, source_description)
    """
    # Handle midnight edge case: when today has no data yet, use yesterday
    # as the primary reference to maintain continuity across midnight
    if today_hours < 1 and yesterday_avg > 0:
        # At midnight, yesterday's average should be the baseline
        # This is what "today" was just an hour ago, so it maintains continuity
        weight_today = 0.0
        weight_yesterday = 0.7
        weight_day_before = 0.3
    else:
        # Normal case: calculate today's weight (grows as day progresses)
        weight_today = today_hours / 24 if today_hours > 0 else 0.0
        remaining_weight = 1 - weight_today

        # Split remaining weight: 70% yesterday, 30% day-before
        weight_yesterday = remaining_weight * 0.7
        weight_day_before = remaining_weight * 0.3

    # Handle missing data - redistribute weights
    if yesterday_avg == 0 and day_before_avg == 0:
        # No historical data - use today only
        if today_avg > 0:
            return (today_avg, "today_only")
        return (0.0, "no_data")

    if yesterday_avg == 0:
        # No yesterday - redistribute to day-before
        weight_day_before += weight_yesterday
        weight_yesterday = 0

    if day_before_avg == 0:
        # No day-before - redistribute to yesterday
        weight_yesterday += weight_day_before
        weight_day_before = 0

    # Calculate weighted average
    weighted = (
        today_avg * weight_today +
        yesterday_avg * weight_yesterday +
        day_before_avg * weight_day_before
    )

    return (weighted, "72h_weighted")


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
    """
    try:
        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.statistics import statistics_during_period

        # Convert to UTC timestamps for the API
        start_utc = dt_util.as_utc(start_time)
        end_utc = dt_util.as_utc(end_time)

        recorder = get_instance(hass)

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

        # Always run in executor – no direct call from async
        stats = await recorder.async_add_executor_job(_fetch_stats)

        # Calculate deltas between consecutive hours
        hourly: Dict[int, float] = {}
        sensor_stats = stats.get(entity_id, [])

        if not sensor_stats:
            _LOGGER.debug("No statistics found for %s", entity_id)
            return hourly

        sorted_stats = sorted(sensor_stats, key=lambda x: x["start"])

        prev_sum = None
        for stat in sorted_stats:
            current_sum = stat.get("sum")
            if current_sum is None:
                continue

            start = stat["start"]
            if isinstance(start, (int, float)):
                start_dt = datetime.fromtimestamp(start, tz=dt_util.UTC)
            else:
                start_dt = start
            local_time = dt_util.as_local(start_dt)
            hour = local_time.hour

            if prev_sum is not None:
                delta = current_sum - prev_sum
                if delta >= 0:
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

    Calculates real consumption using the formula:
    real_consumption = solar + grid_import - grid_export + battery_discharge - battery_charge

    Args:
        hass: Home Assistant instance
        config: CEW configuration dict with energy_use_* toggles

    Returns:
        Dict with hourly data for consumption, solar, and battery
    """
    now = dt_util.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    result: Dict[str, Any] = {
        # Raw hourly data from sensors
        "grid_import_hourly": {},  # {hour: kwh_delta}
        "grid_export_hourly": {},
        "solar_hourly": {},
        "battery_charge_hourly": {},
        "battery_discharge_hourly": {},
        # Calculated real consumption (the formula result)
        "real_consumption_hourly": {},
        # Averages for display and future hour estimates
        "avg_grid_import": 0.0,
        "avg_grid_export": 0.0,
        "avg_solar": 0.0,
        "avg_battery_charge": 0.0,
        "avg_battery_discharge": 0.0,
        "avg_real_consumption": 0.0,
        "total_battery_charge_kwh": 0.0,
        "total_battery_discharge_kwh": 0.0,
        "total_solar_production_kwh": 0.0,
        # Metadata
        "stats_available": False,
        "current_hour": now.hour,
        "hours_with_data": 0,
        "consumption_source": "manual",  # "today", "yesterday", or "manual"
        # Weighted 72h consumption average (for stable baseline)
        "today_avg_consumption": 0.0,
        "today_hours_with_data": 0,
        "yesterday_avg_consumption": 0.0,
        "day_before_avg_consumption": 0.0,
        "weighted_avg_consumption": 0.0,
        "weighted_consumption_source": "manual",
        # Discovered sensor info for dashboard feedback
        "sensors": {
            "grid_import": [],
            "grid_export": [],
            "solar": [],
            "battery_charge": [],
            "battery_discharge": [],
        },
        # Legacy keys for backward compatibility
        "consumption_hourly": {},  # Alias for grid_import_hourly
        "avg_hourly_consumption": 0.0,  # Alias for avg_grid_import
        "avg_hourly_solar": 0.0,
        "avg_hourly_battery_charge": 0.0,
        "avg_hourly_battery_discharge": 0.0,
        "consumption_sensor": "none",
        "solar_sensor": "none",
        "battery_sensor": "none",
    }

    use_ha_energy = config.get("use_ha_energy_dashboard", False)

    if not use_ha_energy:
        _LOGGER.debug("HA Energy Dashboard disabled - using manual values")
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

    yesterday_start = today_start - timedelta(days=1)
    yesterday_end = today_start
    stats_start = today_start - timedelta(hours=1)
    use_yesterday_fallback = False

    # Grid consumption (import)
    import_sensors = energy_prefs.get("grid_consumption_sensors", [])
    export_sensors = energy_prefs.get("grid_return_sensors", [])

    if import_sensors:
        result["sensors"]["grid_import"] = import_sensors
        result["consumption_sensor"] = import_sensors[0] if len(import_sensors) == 1 else f"{len(import_sensors)} sensors"

        result["grid_import_hourly"] = await fetch_combined_hourly_statistics(
            hass, import_sensors, stats_start, now
        )
        # Also update legacy key
        result["consumption_hourly"] = result["grid_import_hourly"]

        if result["grid_import_hourly"]:
            result["stats_available"] = True
            result["consumption_source"] = "today"
            total = sum(result["grid_import_hourly"].values())
            hours = len(result["grid_import_hourly"])
            result["avg_grid_import"] = total / hours if hours > 0 else 0
            result["avg_hourly_consumption"] = result["avg_grid_import"]
            result["hours_with_data"] = hours
            _LOGGER.debug("Grid import: %d hours, avg %.3f kWh/hr", hours, result["avg_grid_import"])
        else:
            # No data for today yet - use yesterday as fallback
            use_yesterday_fallback = True
            yesterday_hourly = await fetch_combined_hourly_statistics(
                hass, import_sensors, yesterday_start, yesterday_end
            )
            if yesterday_hourly:
                result["stats_available"] = True
                result["consumption_source"] = "yesterday"
                result["grid_import_hourly"] = yesterday_hourly
                result["consumption_hourly"] = yesterday_hourly
                total = sum(yesterday_hourly.values())
                hours = len(yesterday_hourly)
                result["avg_grid_import"] = total / hours if hours > 0 else 0
                result["avg_hourly_consumption"] = result["avg_grid_import"]
                result["hours_with_data"] = hours
                _LOGGER.debug("Grid import (yesterday fallback): %d hours, avg %.3f kWh/hr", hours, result["avg_grid_import"])
    else:
        _LOGGER.info("HA Energy Dashboard: no grid import sensor configured")

    # Grid export
    if export_sensors:
        result["sensors"]["grid_export"] = export_sensors

        if not use_yesterday_fallback:
            result["grid_export_hourly"] = await fetch_combined_hourly_statistics(
                hass, export_sensors, stats_start, now
            )
        else:
            result["grid_export_hourly"] = await fetch_combined_hourly_statistics(
                hass, export_sensors, yesterday_start, yesterday_end
            )

        if result["grid_export_hourly"]:
            total = sum(result["grid_export_hourly"].values())
            hours = len(result["grid_export_hourly"])
            result["avg_grid_export"] = total / hours if hours > 0 else 0
            _LOGGER.debug("Grid export: avg %.3f kWh/hr", result["avg_grid_export"])

    # Solar production
    sensors = energy_prefs.get("solar_production_sensors", [])
    if sensors:
        result["sensors"]["solar"] = sensors
        result["solar_sensor"] = sensors[0] if len(sensors) == 1 else f"{len(sensors)} sensors"

        # Always try to fetch today's solar first (independent of grid fallback)
        result["solar_hourly"] = await fetch_combined_hourly_statistics(
            hass, sensors, stats_start, now
        )

        if result["solar_hourly"]:
            result["stats_available"] = True
            result["solar_source"] = "today"
            total = sum(result["solar_hourly"].values())
            hours = len(result["solar_hourly"])
            result["avg_solar"] = total / hours if hours > 0 else 0
            result["avg_hourly_solar"] = result["avg_solar"]
            result["total_solar_production_kwh"] = total
            _LOGGER.debug("Solar: total %.3f kWh, avg %.3f kWh/hr", total, result["avg_solar"])
        else:
            # No solar data for today (normal at night/early morning)
            result["solar_source"] = "today"
            result["total_solar_production_kwh"] = 0.0
            result["avg_solar"] = 0.0
            result["avg_hourly_solar"] = 0.0
            _LOGGER.debug("Solar: no data for today yet (0 kWh)")

    # Battery charge/discharge
    charge_sensors = energy_prefs.get("battery_charge_sensors", [])
    discharge_sensors = energy_prefs.get("battery_discharge_sensors", [])

    if charge_sensors or discharge_sensors:
        result["battery_sensor"] = "configured"

        if charge_sensors:
            result["sensors"]["battery_charge"] = charge_sensors

            # Always fetch TODAY's battery data - never fallback to yesterday
            # Battery totals should be 0 until actual data comes in for today
            # Use today_start (midnight) not stats_start to avoid getting yesterday's late-night data
            result["battery_charge_hourly"] = await fetch_combined_hourly_statistics(
                hass, charge_sensors, today_start, now
            )

            if result["battery_charge_hourly"]:
                result["stats_available"] = True
                result["battery_charge_source"] = "today"
                total = sum(result["battery_charge_hourly"].values())
                hours = len(result["battery_charge_hourly"])
                result["avg_battery_charge"] = total / hours if hours > 0 else 0
                result["avg_hourly_battery_charge"] = result["avg_battery_charge"]
                result["total_battery_charge_kwh"] = total
                _LOGGER.debug("Battery charge: total %.3f kWh, avg %.3f kWh/hr", total, result["avg_battery_charge"])
            else:
                # No battery data for today yet - keep totals at 0
                result["battery_charge_source"] = "today"
                result["total_battery_charge_kwh"] = 0.0
                _LOGGER.debug("Battery charge: no data for today yet, total = 0")

        if discharge_sensors:
            result["sensors"]["battery_discharge"] = discharge_sensors

            # Always fetch TODAY's battery data - never fallback to yesterday
            # Battery totals should be 0 until actual data comes in for today
            # Use today_start (midnight) not stats_start to avoid getting yesterday's late-night data
            result["battery_discharge_hourly"] = await fetch_combined_hourly_statistics(
                hass, discharge_sensors, today_start, now
            )

            if result["battery_discharge_hourly"]:
                result["stats_available"] = True
                result["battery_discharge_source"] = "today"
                total = sum(result["battery_discharge_hourly"].values())
                hours = len(result["battery_discharge_hourly"])
                result["avg_battery_discharge"] = total / hours if hours > 0 else 0
                result["avg_hourly_battery_discharge"] = result["avg_battery_discharge"]
                result["total_battery_discharge_kwh"] = total
                _LOGGER.debug("Battery discharge: total %.3f kWh, avg %.3f kWh/hr", total, result["avg_battery_discharge"])
            else:
                # No battery data for today yet - keep totals at 0
                result["battery_discharge_source"] = "today"
                result["total_battery_discharge_kwh"] = 0.0
                _LOGGER.debug("Battery discharge: no data for today yet, total = 0")

    # Calculate real consumption using the formula
    # real_consumption = solar + grid_import - grid_export + battery_discharge - battery_charge
    if result["stats_available"]:
        result["real_consumption_hourly"] = calculate_real_consumption(
            grid_import=result["grid_import_hourly"],
            grid_export=result["grid_export_hourly"],
            solar=result["solar_hourly"],
            battery_charge=result["battery_charge_hourly"],
            battery_discharge=result["battery_discharge_hourly"],
        )

        if result["real_consumption_hourly"]:
            total = sum(result["real_consumption_hourly"].values())
            hours = len(result["real_consumption_hourly"])
            result["avg_real_consumption"] = total / hours if hours > 0 else 0
            result["hours_with_data"] = max(result["hours_with_data"], hours)
            _LOGGER.debug(
                "Real consumption calculated: %d hours, avg %.3f kWh/hr",
                hours, result["avg_real_consumption"]
            )

    # Calculate weighted 72h consumption average for stable baseline
    # This fetches yesterday and day-before data to provide stable baseline early in the day
    if import_sensors and result["stats_available"]:
        # Today's average (already calculated)
        today_avg = result["avg_real_consumption"]

        # IMPORTANT: If we're using yesterday fallback, today has no actual data yet
        # Don't count yesterday's hours as today's hours - that would break the weighted average
        if use_yesterday_fallback:
            today_hours = 0  # No actual today data - we're using yesterday as fallback
        else:
            today_hours = len(result["real_consumption_hourly"]) if result["real_consumption_hourly"] else 0

        # Store today's data
        result["today_avg_consumption"] = today_avg
        result["today_hours_with_data"] = today_hours

        # Fetch yesterday (full 24h) - only if not already using yesterday fallback
        yesterday_avg = 0.0
        day_before_avg = 0.0

        # Always fetch yesterday's full data for accurate weighted average
        # Even when use_yesterday_fallback is True, we need all components (solar, battery)
        # because the fallback only copied grid data, not solar/battery
        # NOTE: We extend start by 1 hour to capture the baseline sum needed for hour 0's delta
        yesterday_fetch_start = yesterday_start - timedelta(hours=1)
        yesterday_import = await fetch_combined_hourly_statistics(
            hass, import_sensors, yesterday_fetch_start, yesterday_end
        )
        yesterday_export = await fetch_combined_hourly_statistics(
            hass, export_sensors, yesterday_fetch_start, yesterday_end
        ) if export_sensors else {}
        yesterday_solar = await fetch_combined_hourly_statistics(
            hass, energy_prefs.get("solar_production_sensors", []), yesterday_fetch_start, yesterday_end
        ) if energy_prefs.get("solar_production_sensors") else {}
        yesterday_charge = await fetch_combined_hourly_statistics(
            hass, charge_sensors, yesterday_fetch_start, yesterday_end
        ) if charge_sensors else {}
        yesterday_discharge = await fetch_combined_hourly_statistics(
            hass, discharge_sensors, yesterday_fetch_start, yesterday_end
        ) if discharge_sensors else {}

        yesterday_consumption = calculate_real_consumption(
            yesterday_import, yesterday_export, yesterday_solar,
            yesterday_charge, yesterday_discharge
        )
        if yesterday_consumption:
            yesterday_avg = sum(yesterday_consumption.values()) / len(yesterday_consumption)

        result["yesterday_avg_consumption"] = yesterday_avg

        # Fetch day before yesterday (full 24h)
        day_before_start = today_start - timedelta(days=2)
        day_before_end = yesterday_start
        # NOTE: We extend start by 1 hour to capture the baseline sum needed for hour 0's delta
        day_before_fetch_start = day_before_start - timedelta(hours=1)

        day_before_import = await fetch_combined_hourly_statistics(
            hass, import_sensors, day_before_fetch_start, day_before_end
        )
        day_before_export = await fetch_combined_hourly_statistics(
            hass, export_sensors, day_before_fetch_start, day_before_end
        ) if export_sensors else {}
        day_before_solar = await fetch_combined_hourly_statistics(
            hass, energy_prefs.get("solar_production_sensors", []), day_before_fetch_start, day_before_end
        ) if energy_prefs.get("solar_production_sensors") else {}
        day_before_charge = await fetch_combined_hourly_statistics(
            hass, charge_sensors, day_before_fetch_start, day_before_end
        ) if charge_sensors else {}
        day_before_discharge = await fetch_combined_hourly_statistics(
            hass, discharge_sensors, day_before_fetch_start, day_before_end
        ) if discharge_sensors else {}

        day_before_consumption = calculate_real_consumption(
            day_before_import, day_before_export, day_before_solar,
            day_before_charge, day_before_discharge
        )
        if day_before_consumption:
            day_before_avg = sum(day_before_consumption.values()) / len(day_before_consumption)

        result["day_before_avg_consumption"] = day_before_avg

        # Calculate weighted average
        weighted_avg, weighted_source = calculate_weighted_consumption_avg(
            today_avg, today_hours, yesterday_avg, day_before_avg
        )
        result["weighted_avg_consumption"] = weighted_avg
        result["weighted_consumption_source"] = weighted_source

        _LOGGER.debug(
            "Weighted consumption: today=%.3f (%dh), yesterday=%.3f, day_before=%.3f -> weighted=%.3f (%s)",
            today_avg, today_hours, yesterday_avg, day_before_avg, weighted_avg, weighted_source
        )

    return result


async def get_sensor_state_at_midnight(
    hass: HomeAssistant,
    entity_id: str,
) -> float | None:
    """Get sensor state at midnight (start of today) from recorder history.

    Used to get battery sensor value at the start of the day for accurate
    full-day simulation. Falls back to current state if history unavailable.

    Args:
        hass: Home Assistant instance
        entity_id: Battery sensor entity ID (e.g., sensor.battery_soc)

    Returns:
        Sensor value at midnight as float, or None if unavailable
    """
    if not entity_id or entity_id == "not_configured":
        return None

    try:
        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.history import state_changes_during_period

        now = dt_util.now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # Search window: midnight to 1 hour after
        end_time = midnight + timedelta(hours=1)

        recorder = get_instance(hass)

        def _fetch_history():
            return state_changes_during_period(
                hass,
                start_time=midnight,
                end_time=end_time,
                entity_id=entity_id,
                include_start_time_state=True,
                no_attributes=True,
            )

        states = await recorder.async_add_executor_job(_fetch_history)

        if not states or entity_id not in states:
            _LOGGER.debug("No history found for %s at midnight", entity_id)
            return None

        entity_states = states[entity_id]
        if not entity_states:
            return None

        # Get the first state (closest to midnight)
        first_state = entity_states[0]
        try:
            value = float(first_state.state)
            _LOGGER.debug("Battery state at midnight for %s: %.2f", entity_id, value)
            return value
        except (ValueError, TypeError):
            _LOGGER.debug("Could not parse state %s for %s", first_state.state, entity_id)
            return None

    except ImportError as e:
        _LOGGER.warning("Recorder history not available: %s", e)
        return None
    except Exception as e:
        _LOGGER.warning("Failed to get midnight state for %s: %s", entity_id, e)
        return None
