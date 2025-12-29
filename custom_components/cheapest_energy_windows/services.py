"""Services for Cheapest Energy Windows."""
from __future__ import annotations

import logging
from pathlib import Path
import yaml

from homeassistant.core import HomeAssistant, ServiceCall
import homeassistant.helpers.config_validation as cv
from homeassistant.util import dt as dt_util
import voluptuous as vol

from .const import (
    DOMAIN,
    LOGGER_NAME,
    PREFIX,
    SERVICE_ROTATE_SETTINGS,
    EVENT_SETTINGS_ROTATED,
)

_LOGGER = logging.getLogger(LOGGER_NAME)

# Service schemas
SERVICE_ROTATE_SCHEMA = vol.Schema({})


async def async_create_notification_automation(hass: HomeAssistant) -> tuple[bool, str]:
    """Create the notification automation in automations.yaml.

    Returns:
        tuple[bool, str]: (Success status, Message)
    """
    try:
        automation_id = f"{DOMAIN}_notifications"

        # Get the path to automations.yaml
        automations_path = hass.config.path("automations.yaml")
        _LOGGER.info(f"Automations file path: {automations_path}")

        # Read existing automations (non-blocking)
        existing_automations = []
        if Path(automations_path).exists():
            try:
                def read_automations_file():
                    with open(automations_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            return yaml.safe_load(content) or []
                    return []

                existing_automations = await hass.async_add_executor_job(read_automations_file)

                if not isinstance(existing_automations, list):
                    existing_automations = [existing_automations]

                # Remove existing automation if present (to update with latest template)
                existing_automations = [
                    auto for auto in existing_automations
                    if not (isinstance(auto, dict) and auto.get("id") == automation_id)
                ]
                _LOGGER.info(f"Updating automation {automation_id} with latest template")

            except yaml.YAMLError as e:
                _LOGGER.error(f"Error parsing existing automations.yaml: {e}")
                return False, f"Failed to parse existing automations: {e}"

        # Load automation template from automation_template.yaml
        # This is the single source of truth for the automation structure
        template_path = Path(__file__).parent / "automation_template.yaml"
        _LOGGER.info(f"Loading automation template from: {template_path}")

        try:
            def read_template_file():
                with open(template_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    return yaml.safe_load(content)

            template_automation = await hass.async_add_executor_job(read_template_file)

            # The template is already in the correct format, just need to add the ID
            if not isinstance(template_automation, dict):
                raise ValueError("automation_template.yaml must contain a single automation dictionary")

            # Set the automation ID (template might have a different one)
            template_automation["id"] = automation_id

            new_automation = template_automation
            _LOGGER.info(f"Successfully loaded automation template (using notify.notify notifications)")

        except Exception as e:
            _LOGGER.error(f"Error loading automation_template.yaml: {e}")
            _LOGGER.warning("Falling back to basic automation structure")

            # Fallback to minimal automation if template can't be loaded
            new_automation = {
            "id": automation_id,
            "alias": "CEW - Battery Control Automation",
            "description": "Battery control automation for Cheapest Energy Windows (add your battery actions to each trigger)",
            "mode": "queued",
            "max": 10,
            "trigger": [
                {
                    "platform": "state",
                    "entity_id": f"sensor.{PREFIX}today",
                    "to": "charge",
                    "from": ["discharge", "normal", "off"],
                    "id": "charge_start"
                },
                {
                    "platform": "state",
                    "entity_id": f"sensor.{PREFIX}today",
                    "to": "discharge",
                    "from": ["charge", "normal", "off"],
                    "id": "discharge_start"
                },
                {
                    "platform": "state",
                    "entity_id": f"sensor.{PREFIX}today",
                    "to": "normal",
                    "from": ["charge", "discharge", "off"],
                    "id": "normal_start"
                },
                {
                    "platform": "state",
                    "entity_id": f"sensor.{PREFIX}today",
                    "to": "off",
                    "from": ["charge", "discharge", "normal"],
                    "id": "automation_disabled"
                }
            ],
            "condition": [],
            "action": [
                {
                    "choose": [
                        {
                            "conditions": [
                                {"condition": "trigger", "id": "charge_start"}
                            ],
                            "sequence": [
                                {
                                    "service": "persistent_notification.create",
                                    "data": {
                                        "title": "CEW Battery Action Needed",
                                        "message": (
                                            "⚠️ CHARGE trigger fired but no battery action configured.\n\n"
                                            "Edit this automation and add your battery CHARGE action here.\n"
                                            "Example: Turn on battery charge mode, set charge power, etc."
                                        ),
                                        "notification_id": "cew_charge_action_needed"
                                    }
                                }
                            ]
                        },
                        {
                            "conditions": [
                                {"condition": "trigger", "id": "discharge_start"}
                            ],
                            "sequence": [
                                {
                                    "service": "persistent_notification.create",
                                    "data": {
                                        "title": "CEW Battery Action Needed",
                                        "message": (
                                            "⚠️ DISCHARGE trigger fired but no battery action configured.\n\n"
                                            "Edit this automation and add your battery DISCHARGE action here.\n"
                                            "Example: Turn on battery discharge mode, set discharge power, etc."
                                        ),
                                        "notification_id": "cew_discharge_action_needed"
                                    }
                                }
                            ]
                        },
                        {
                            "conditions": [
                                {"condition": "trigger", "id": "normal_start"}
                            ],
                            "sequence": [
                                {
                                    "service": "persistent_notification.create",
                                    "data": {
                                        "title": "CEW Battery Action Needed",
                                        "message": (
                                            "⚠️ NORMAL trigger fired but no battery action configured.\n\n"
                                            "Edit this automation and add your battery NORMAL action here.\n"
                                            "Example: Set battery to standby mode, 0W charge/discharge, etc."
                                        ),
                                        "notification_id": "cew_normal_action_needed"
                                    }
                                }
                            ]
                        },
                        {
                            "conditions": [
                                {"condition": "trigger", "id": "automation_disabled"}
                            ],
                            "sequence": [
                                {
                                    "service": "persistent_notification.create",
                                    "data": {
                                        "title": "CEW Automation Disabled",
                                        "message": (
                                            "CEW automation has been disabled.\n\n"
                                            "You can add a battery STOP/MANUAL mode action here if needed."
                                        ),
                                        "notification_id": "cew_automation_disabled"
                                    }
                                }
                            ]
                        }
                    ],
                    "default": []
                }
            ]
        }

        # Add to existing automations
        existing_automations.append(new_automation)

        # Write to file (non-blocking)
        def write_automations_file():
            with open(automations_path, "w", encoding="utf-8") as f:
                yaml.dump(existing_automations, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        await hass.async_add_executor_job(write_automations_file)

        _LOGGER.info(f"Successfully wrote automation to {automations_path}")

        # Reload automations
        try:
            await hass.services.async_call(
                "automation",
                "reload",
                {},
                blocking=True
            )
            _LOGGER.info("Automations reloaded successfully")
            return True, "Automation created successfully!"
        except Exception as e:
            _LOGGER.warning(f"Failed to reload automations: {e}. Automation will load on next restart.")
            return True, "Automation created! Please restart Home Assistant to activate it."

    except Exception as e:
        _LOGGER.error(f"Error creating automation: {e}", exc_info=True)
        return False, f"Failed to create automation: {str(e)}"


async def async_setup_services(hass: HomeAssistant) -> None:
    """Set up services for Cheapest Energy Windows."""

    async def handle_rotate_settings(call: ServiceCall) -> None:
        """Handle the rotate_settings service call."""
        _LOGGER.info("Rotating tomorrow settings to today")

        # List of settings to rotate
        settings_to_rotate = [
            # Window counts
            ("charging_windows_tomorrow", "charging_windows"),
            ("expensive_windows_tomorrow", "expensive_windows"),
            # Percentile threshold
            ("percentile_threshold_tomorrow", "percentile_threshold"),
            # Profit thresholds (replaced old min_spread settings)
            ("min_profit_charge_tomorrow", "min_profit_charge"),
            ("min_profit_discharge_tomorrow", "min_profit_discharge"),
            # Price override
            ("price_override_threshold_tomorrow", "price_override_threshold"),
            # Auto-optimizer min savings threshold
            ("min_daily_savings_tomorrow", "min_daily_savings"),
        ]

        # Rotate number settings
        for tomorrow_key, today_key in settings_to_rotate:
            tomorrow_entity = f"number.{PREFIX}{tomorrow_key}"
            today_entity = f"number.{PREFIX}{today_key}"

            tomorrow_state = hass.states.get(tomorrow_entity)
            if tomorrow_state:
                await hass.services.async_call(
                    "number",
                    "set_value",
                    {"entity_id": today_entity, "value": float(tomorrow_state.state)},
                    blocking=True,
                )
                _LOGGER.debug(f"Rotated {tomorrow_key} -> {today_key}: {tomorrow_state.state}")

        # Rotate boolean settings
        boolean_settings = [
            ("price_override_enabled_tomorrow", "price_override_enabled"),
            ("time_override_enabled_tomorrow", "time_override_enabled"),
            ("calculation_window_enabled_tomorrow", "calculation_window_enabled"),
        ]

        for tomorrow_key, today_key in boolean_settings:
            tomorrow_entity = f"switch.{PREFIX}{tomorrow_key}"
            today_entity = f"switch.{PREFIX}{today_key}"

            tomorrow_state = hass.states.get(tomorrow_entity)
            if tomorrow_state:
                service = "turn_on" if tomorrow_state.state == "on" else "turn_off"
                await hass.services.async_call(
                    "switch",
                    service,
                    {"entity_id": today_entity},
                    blocking=True,
                )
                _LOGGER.debug(f"Rotated {tomorrow_key} -> {today_key}: {tomorrow_state.state}")

        # Rotate select settings
        select_settings = [
            ("time_override_mode_tomorrow", "time_override_mode"),
            ("auto_optimize_strategy_tomorrow", "auto_optimize_strategy"),
        ]

        for tomorrow_key, today_key in select_settings:
            tomorrow_entity = f"select.{PREFIX}{tomorrow_key}"
            today_entity = f"select.{PREFIX}{today_key}"

            tomorrow_state = hass.states.get(tomorrow_entity)
            if tomorrow_state:
                await hass.services.async_call(
                    "select",
                    "select_option",
                    {"entity_id": today_entity, "option": tomorrow_state.state},
                    blocking=True,
                )
                _LOGGER.debug(f"Rotated {tomorrow_key} -> {today_key}: {tomorrow_state.state}")

        # Rotate datetime settings
        datetime_settings = [
            ("time_override_start_tomorrow", "time_override_start"),
            ("time_override_end_tomorrow", "time_override_end"),
            ("calculation_window_start_tomorrow", "calculation_window_start"),
            ("calculation_window_end_tomorrow", "calculation_window_end"),
        ]

        for tomorrow_key, today_key in datetime_settings:
            tomorrow_entity = f"time.{PREFIX}{tomorrow_key}"
            today_entity = f"time.{PREFIX}{today_key}"

            tomorrow_state = hass.states.get(tomorrow_entity)
            if tomorrow_state:
                await hass.services.async_call(
                    "time",
                    "set_value",
                    {"entity_id": today_entity, "time": tomorrow_state.state},
                    blocking=True,
                )
                _LOGGER.debug(f"Rotated {tomorrow_key} -> {today_key}: {tomorrow_state.state}")

        # Rotate calculated windows: tomorrow -> today
        # This preserves the calculated windows across midnight
        _LOGGER.info("Rotating calculated windows: tomorrow -> today")

        tomorrow_sensor = hass.states.get(f"sensor.{PREFIX}tomorrow")
        if tomorrow_sensor and tomorrow_sensor.attributes:
            # Find the entry_id - should be the first (and only) one
            if DOMAIN in hass.data and hass.data[DOMAIN]:
                entry_id = list(hass.data[DOMAIN].keys())[0]
                hass.data[DOMAIN][entry_id]["rotated_windows_today"] = {
                    "actual_charge_times": tomorrow_sensor.attributes.get("actual_charge_times", []),
                    "actual_charge_prices": tomorrow_sensor.attributes.get("actual_charge_prices", []),
                    "actual_discharge_times": tomorrow_sensor.attributes.get("actual_discharge_times", []),
                    "actual_discharge_prices": tomorrow_sensor.attributes.get("actual_discharge_prices", []),
                    "grouped_charge_windows": tomorrow_sensor.attributes.get("grouped_charge_windows", []),
                    "grouped_discharge_windows": tomorrow_sensor.attributes.get("grouped_discharge_windows", []),
                    "charge_window_count": tomorrow_sensor.attributes.get("charge_window_count", 0),
                    "discharge_window_count": tomorrow_sensor.attributes.get("discharge_window_count", 0),
                    "total_charge_cost": tomorrow_sensor.attributes.get("total_charge_cost", 0),
                    "total_discharge_revenue": tomorrow_sensor.attributes.get("total_discharge_revenue", 0),
                    "net_benefit": tomorrow_sensor.attributes.get("net_benefit", 0),
                    "windows_calculated": tomorrow_sensor.attributes.get("windows_calculated", False),
                    "calculation_complete": tomorrow_sensor.attributes.get("calculation_complete", False),
                    "rotation_timestamp": dt_util.now().isoformat(),
                }
                _LOGGER.info("Window rotation complete - tomorrow's windows stored for today")
        else:
            _LOGGER.debug("No tomorrow sensor or attributes to rotate")

        # Fire event
        hass.bus.async_fire(EVENT_SETTINGS_ROTATED, {})

        _LOGGER.info("Settings rotation complete")

    async def handle_trigger_battery_action(call: ServiceCall) -> None:
        """Handle triggering battery mode actions."""
        mode = call.data.get("mode")

        # Map mode to text entity
        mode_entity_map = {
            "normal": f"text.{PREFIX}battery_normal_action",
            "charge": f"text.{PREFIX}battery_charge_action",
            "discharge": f"text.{PREFIX}battery_discharge_action",
            "off": f"text.{PREFIX}battery_off_action",
        }

        text_entity = mode_entity_map.get(mode)
        if not text_entity:
            _LOGGER.error(f"Invalid mode: {mode}")
            return

        # Get the configured automation/script/scene entity_id
        text_state = hass.states.get(text_entity)
        if not text_state:
            _LOGGER.error(f"Text entity not found: {text_entity}")
            return

        target_entity = text_state.state
        if not target_entity or target_entity == "not_configured":
            _LOGGER.warning(f"No action configured for mode: {mode}")
            return

        # Determine service based on entity type
        if target_entity.startswith("automation."):
            service = "automation.trigger"
        elif target_entity.startswith("script."):
            service = "script.turn_on"
        elif target_entity.startswith("scene."):
            service = "scene.turn_on"
        else:
            _LOGGER.error(f"Unsupported entity type: {target_entity}")
            return

        # Call the service
        domain, service_name = service.split(".")
        await hass.services.async_call(
            domain,
            service_name,
            {"entity_id": target_entity},
            blocking=False,
        )
        _LOGGER.info(f"Triggered {mode} action: {target_entity}")

    # Register services
    hass.services.async_register(
        DOMAIN,
        SERVICE_ROTATE_SETTINGS,
        handle_rotate_settings,
        schema=SERVICE_ROTATE_SCHEMA,
    )

    hass.services.async_register(
        DOMAIN,
        "trigger_battery_action",
        handle_trigger_battery_action,
        schema=vol.Schema({
            vol.Required("mode"): vol.In(["normal", "charge", "discharge", "off"]),
        }),
    )

    _LOGGER.info("Services registered successfully")