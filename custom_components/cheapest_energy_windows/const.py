"""Constants for Cheapest Energy Windows."""
from datetime import timedelta
from typing import Final

# Domain
DOMAIN: Final = "cheapest_energy_windows"
PREFIX: Final = "cew_"
VERSION: Final = "1.2.0"

# Platforms
PLATFORMS: Final = ["sensor", "number", "select", "switch", "time", "text"]

# Configuration keys
CONF_PRICE_SENSOR: Final = "price_sensor"
CONF_VAT_RATE: Final = "vat_rate"
CONF_TAX: Final = "tax"
CONF_ADDITIONAL_COST: Final = "additional_cost"
CONF_BATTERY_SYSTEM_NAME: Final = "battery_system_name"
CONF_BATTERY_SOC_SENSOR: Final = "battery_soc_sensor"
CONF_BATTERY_ENERGY_SENSOR: Final = "battery_available_energy_sensor"
CONF_BATTERY_CHARGE_SENSOR: Final = "battery_daily_charge_sensor"
CONF_BATTERY_DISCHARGE_SENSOR: Final = "battery_daily_discharge_sensor"
CONF_BATTERY_POWER_SENSOR: Final = "battery_power_sensor"
CONF_BASE_USAGE: Final = "base_usage"
CONF_BASE_USAGE_CHARGE_STRATEGY: Final = "base_usage_charge_strategy"
CONF_BASE_USAGE_IDLE_STRATEGY: Final = "base_usage_idle_strategy"
CONF_BASE_USAGE_DISCHARGE_STRATEGY: Final = "base_usage_discharge_strategy"

# Buffer/chronological calculation configuration keys
CONF_BATTERY_BUFFER_KWH: Final = "battery_buffer_kwh"
CONF_BATTERY_BUFFER_ENERGY_SENSOR: Final = "battery_buffer_energy_sensor"
CONF_BATTERY_CAPACITY: Final = "battery_capacity"
CONF_LIMIT_DISCHARGE_TO_BUFFER: Final = "limit_discharge_to_buffer"

# Unified price country configuration key
CONF_PRICE_COUNTRY: Final = "price_country"

# Sell price configuration keys
CONF_MIN_SELL_PRICE: Final = "min_sell_price"
CONF_USE_MIN_SELL_PRICE: Final = "use_min_sell_price"
CONF_MIN_SELL_PRICE_BYPASS_SPREAD: Final = "min_sell_price_bypass_spread"
CONF_SELL_FORMULA_PARAM_A: Final = "sell_formula_param_a"
CONF_SELL_FORMULA_PARAM_B: Final = "sell_formula_param_b"

# Buy price configuration keys
CONF_BUY_FORMULA_PARAM_A: Final = "buy_formula_param_a"
CONF_BUY_FORMULA_PARAM_B: Final = "buy_formula_param_b"

# Profit threshold settings (replaced old spread settings)
# Profit = Spread - RTE_Loss, where RTE_Loss = 100 - battery_rte
CONF_MIN_PROFIT_CHARGE: Final = "min_profit_charge"
CONF_MIN_PROFIT_DISCHARGE: Final = "min_profit_discharge"

# Default values
DEFAULT_PRICE_SENSOR: Final = ""
DEFAULT_VAT_RATE: Final = 21
DEFAULT_TAX: Final = 0.12286
DEFAULT_ADDITIONAL_COST: Final = 0.02398
DEFAULT_CHARGING_WINDOWS: Final = 6
DEFAULT_EXPENSIVE_WINDOWS: Final = 3
DEFAULT_PERCENTILE_THRESHOLD: Final = 25
# Profit thresholds (profit = spread - RTE_loss, default 10%)
DEFAULT_MIN_PROFIT_CHARGE: Final = 10
DEFAULT_MIN_PROFIT_DISCHARGE: Final = 10

DEFAULT_MIN_PRICE_DIFFERENCE: Final = 0.05
DEFAULT_PRICE_OVERRIDE_THRESHOLD: Final = 0.15
DEFAULT_BATTERY_RTE: Final = 85
DEFAULT_CHARGE_POWER: Final = 800
DEFAULT_DISCHARGE_POWER: Final = 800
DEFAULT_BATTERY_SYSTEM_NAME: Final = "My Battery System"
DEFAULT_DISCHARGE_BUFFER_LIMIT_KWH: Final = 0.0
DEFAULT_QUIET_START: Final = "22:00:00"
DEFAULT_QUIET_END: Final = "07:00:00"
DEFAULT_TIME_OVERRIDE_START: Final = "00:00:00"
DEFAULT_TIME_OVERRIDE_END: Final = "00:00:00"
DEFAULT_CALCULATION_WINDOW_START: Final = "00:00:00"
DEFAULT_CALCULATION_WINDOW_END: Final = "23:59:59"
DEFAULT_BASE_USAGE: Final = 500
DEFAULT_BASE_USAGE_CHARGE_STRATEGY: Final = "grid_covers_both"
DEFAULT_BASE_USAGE_IDLE_STRATEGY: Final = "battery_covers_limited"
DEFAULT_BASE_USAGE_DISCHARGE_STRATEGY: Final = "subtract_base"

# Buffer/chronological calculation defaults
DEFAULT_BATTERY_BUFFER_KWH: Final = 0.0
DEFAULT_BATTERY_CAPACITY: Final = 10.0
DEFAULT_LIMIT_DISCHARGE_TO_BUFFER: Final = False
DEFAULT_USE_BATTERY_BUFFER_SENSOR: Final = False

# Solar production integration defaults
CONF_SOLAR_WINDOW_START: Final = "solar_window_start"
CONF_SOLAR_WINDOW_END: Final = "solar_window_end"
CONF_SOLAR_PRIORITY_STRATEGY: Final = "solar_priority_strategy"
CONF_EXPECTED_SOLAR_KWH: Final = "expected_solar_kwh"

DEFAULT_SOLAR_WINDOW_START: Final = "09:00:00"
DEFAULT_SOLAR_WINDOW_END: Final = "19:00:00"
DEFAULT_SOLAR_PRIORITY_STRATEGY: Final = "base_then_grid"
DEFAULT_EXPECTED_SOLAR_KWH: Final = 0.0

# Solar strategy options
SOLAR_STRATEGY_BASE_THEN_GRID: Final = "base_then_grid"
SOLAR_STRATEGY_BASE_THEN_BATTERY: Final = "base_then_battery"
SOLAR_PRIORITY_OPTIONS: Final = [SOLAR_STRATEGY_BASE_THEN_GRID, SOLAR_STRATEGY_BASE_THEN_BATTERY]


# Default price country (used as fallback)
# Country formulas are now defined in the formulas/ subpackage
# See formulas/__init__.py for the registry and auto-discovery
DEFAULT_PRICE_COUNTRY: Final = "netherlands"

# Sell price defaults
DEFAULT_MIN_SELL_PRICE: Final = 0.0
DEFAULT_USE_MIN_SELL_PRICE: Final = False
DEFAULT_MIN_SELL_PRICE_BYPASS_SPREAD: Final = False

# Formula parameter defaults (for backward compatibility)
# These are now defined per-country in formulas/*.py
# Kept here for migration and fallback purposes
DEFAULT_BUY_FORMULA_PARAM_A: Final = 0.009  # Cost (A) in EUR/kWh
DEFAULT_BUY_FORMULA_PARAM_B: Final = 1.0    # Multiplier (B)
DEFAULT_SELL_FORMULA_PARAM_A: Final = 0.009  # Cost (A) in EUR/kWh
DEFAULT_SELL_FORMULA_PARAM_B: Final = 1.0    # Multiplier (B)

# Base usage strategy options
BASE_USAGE_CHARGE_OPTIONS: Final = ["grid_covers_both", "battery_covers_base"]
BASE_USAGE_IDLE_OPTIONS: Final = ["grid_covers", "battery_covers"]
BASE_USAGE_DISCHARGE_OPTIONS: Final = ["already_included", "subtract_base"]

# Update intervals
UPDATE_INTERVAL: Final = timedelta(seconds=10)

# Sensor states
STATE_CHARGE: Final = "charge"
STATE_DISCHARGE: Final = "discharge"
STATE_IDLE: Final = "idle"
STATE_OFF: Final = "off"
STATE_AVAILABLE: Final = "available"
STATE_UNAVAILABLE: Final = "unavailable"

# Battery modes for time overrides
MODE_IDLE: Final = "idle"
MODE_CHARGE: Final = "charge"
MODE_DISCHARGE: Final = "discharge"
MODE_OFF: Final = "off"

# Time override modes list
TIME_OVERRIDE_MODES: Final = [MODE_IDLE, MODE_CHARGE, MODE_DISCHARGE, MODE_OFF]

# Pricing window duration options
PRICING_15_MINUTES: Final = "15_minutes"
PRICING_1_HOUR: Final = "1_hour"
PRICING_WINDOW_OPTIONS: Final = [PRICING_15_MINUTES, PRICING_1_HOUR]

# Attribute names for sensors
ATTR_CHEAPEST_TIMES: Final = "cheapest_times"
ATTR_CHEAPEST_PRICES: Final = "cheapest_prices"
ATTR_EXPENSIVE_TIMES: Final = "expensive_times"
ATTR_EXPENSIVE_PRICES: Final = "expensive_prices"
ATTR_ACTUAL_CHARGE_TIMES: Final = "actual_charge_times"
ATTR_ACTUAL_CHARGE_PRICES: Final = "actual_charge_prices"
ATTR_ACTUAL_DISCHARGE_TIMES: Final = "actual_discharge_times"
ATTR_ACTUAL_DISCHARGE_PRICES: Final = "actual_discharge_prices"
ATTR_COMPLETED_CHARGE_WINDOWS: Final = "completed_charge_windows"
ATTR_COMPLETED_DISCHARGE_WINDOWS: Final = "completed_discharge_windows"
ATTR_COMPLETED_CHARGE_COST: Final = "completed_charge_cost"
ATTR_COMPLETED_DISCHARGE_REVENUE: Final = "completed_discharge_revenue"
ATTR_COMPLETED_SOLAR_EXPORT_REVENUE: Final = "completed_solar_export_revenue"
ATTR_COMPLETED_BASE_USAGE_COST: Final = "completed_base_usage_cost"
ATTR_COMPLETED_BASE_USAGE_BATTERY: Final = "completed_base_usage_battery"
ATTR_COMPLETED_CHARGE_KWH: Final = "completed_charge_kwh"
ATTR_COMPLETED_DISCHARGE_KWH: Final = "completed_discharge_kwh"
ATTR_COMPLETED_BASE_GRID_KWH: Final = "completed_base_grid_kwh"
ATTR_COMPLETED_NET_GRID_KWH: Final = "completed_net_grid_kwh"
ATTR_TOTAL_COST: Final = "total_cost"
ATTR_PLANNED_TOTAL_COST: Final = "planned_total_cost"
ATTR_PLANNED_CHARGE_COST: Final = "planned_charge_cost"
ATTR_NUM_WINDOWS: Final = "num_windows"
# Profit-based attributes (v1.2.0+)
ATTR_CHARGE_PROFIT_PCT: Final = "charge_profit_pct"
ATTR_DISCHARGE_PROFIT_PCT: Final = "discharge_profit_pct"
ATTR_CHARGE_PROFIT_MET: Final = "charge_profit_met"
ATTR_DISCHARGE_PROFIT_MET: Final = "discharge_profit_met"

ATTR_AVG_CHEAP_PRICE: Final = "avg_cheap_price"
ATTR_AVG_EXPENSIVE_PRICE: Final = "avg_expensive_price"
ATTR_CURRENT_PRICE: Final = "current_price"
ATTR_PRICE_OVERRIDE_ACTIVE: Final = "price_override_active"
ATTR_TIME_OVERRIDE_ACTIVE: Final = "time_override_active"
ATTR_CURRENT_SELL_PRICE: Final = "current_sell_price"
ATTR_SELL_PRICE_COUNTRY: Final = "sell_price_country"

# Grid and battery state tracking attributes (v1.2.0+)
ATTR_GRID_KWH_CURRENT_WINDOW: Final = "grid_kwh_current_window"
ATTR_GRID_KWH_ESTIMATED_TODAY: Final = "grid_kwh_estimated_today"
ATTR_GRID_KWH_ESTIMATED_TOMORROW: Final = "grid_kwh_estimated_tomorrow"
ATTR_BATTERY_STATE_CURRENT: Final = "battery_state_current"
ATTR_BATTERY_STATE_END_OF_DAY: Final = "battery_state_end_of_day"
ATTR_BATTERY_STATE_END_OF_DAY_VALUE: Final = "battery_state_end_of_day_value"
ATTR_BATTERY_STATE_END_OF_TOMORROW: Final = "battery_state_end_of_tomorrow"

# Service names
SERVICE_ROTATE_SETTINGS: Final = "rotate_tomorrow_settings"

# Events
EVENT_SETTINGS_ROTATED: Final = f"{DOMAIN}_settings_rotated"

# Logger
LOGGER_NAME: Final = f"custom_components.{DOMAIN}"

# Configuration keys that affect calculation and require coordinator refresh
# These keys, when changed, will trigger recalculation of energy windows
CALCULATION_AFFECTING_KEYS: Final = {
    # Basic calculation settings
    "automation_enabled",
    "charging_windows",
    "expensive_windows",
    "percentile_threshold",
    "min_price_difference",
    "min_price_diff_enabled",

    # Profit thresholds (v1.2.0+)
    "min_profit_charge",
    "min_profit_discharge",

    # Unified price country
    "price_country",

    # Buy/Sell formula parameters
    "buy_formula_param_a",
    "buy_formula_param_b",
    "sell_formula_param_a",
    "sell_formula_param_b",

    # Price adjustments (apply to both buy and sell)
    "vat",
    "tax",
    "additional_cost",

    # Base usage
    "base_usage",
    "base_usage_charge_strategy",
    "base_usage_idle_strategy",
    "base_usage_discharge_strategy",

    # Battery settings
    "battery_rte",
    "charge_power",
    "discharge_power",

    # RTE-aware discharge (global battery setting)
    "rte_aware_discharge",
    "rte_discharge_margin",
    "rte_aware_include_solar",

    # Price overrides
    "price_override_enabled",
    "price_override_threshold",

    # Time overrides
    "time_override_enabled",
    "time_override_start",
    "time_override_end",
    "time_override_mode",

    # Calculation window (restrict price analysis to time range)
    "calculation_window_enabled",
    "calculation_window_start",
    "calculation_window_end",

    # Buffer/chronological calculation settings
    "battery_buffer_kwh",
    "battery_buffer_kwh_tomorrow",
    "battery_available_energy_sensor",
    "use_battery_buffer_sensor",
    "battery_capacity",
    "limit_discharge_to_buffer",
    "discharge_buffer_limit_kwh",
    "use_projected_buffer_tomorrow",

    # Solar production settings
    "solar_window_start",
    "solar_window_end",
    "solar_priority_strategy",
    "expected_solar_kwh",
    "expected_solar_kwh_tomorrow",
    "use_solar_forecast",
    "use_solar_forecast_sensor",
    "solar_forecast_sensor",
    "solar_forecast_sensor_tomorrow",

    # Tomorrow settings
    "tomorrow_settings_enabled",
    "charging_windows_tomorrow",
    "expensive_windows_tomorrow",
    "percentile_threshold_tomorrow",
    "price_override_enabled_tomorrow",
    "price_override_threshold_tomorrow",
    "time_override_enabled_tomorrow",
    "time_override_start_tomorrow",
    "time_override_end_tomorrow",
    "time_override_mode_tomorrow",

    # Window duration
    "pricing_window_duration",

    # Sell price settings
    "min_sell_price",
    "use_min_sell_price",
    "min_sell_price_bypass_spread",

    # Tomorrow profit thresholds (v1.2.0+)
    "min_profit_charge_tomorrow",
    "min_profit_discharge_tomorrow",

}

# Configuration keys that DON'T affect calculation (UI/notification settings)
# These keys can be changed without triggering coordinator refresh
NON_CALCULATION_KEYS: Final = {
    # Notification settings
    "notifications_enabled",
    "quiet_hours_enabled",
    "quiet_hours_start",
    "quiet_hours_end",
    "midnight_rotation_notifications",
    "notify_automation_disabled",
    "notify_charging",
    "notify_discharge",
    "notify_idle",
    "notify_off",

    # Battery system tracking (display only)
    "battery_system_name",
    "battery_soc_sensor",
    "battery_energy_sensor",
    "battery_charge_sensor",
    "battery_discharge_sensor",
    "battery_power_sensor",

    # Price sensor (handled separately as it requires reload)
    "price_sensor_entity",
}