"""Constants for Cheapest Energy Windows."""
from datetime import timedelta
from typing import Final

# Domain
DOMAIN: Final = "cheapest_energy_windows"
PREFIX: Final = "cew_"
VERSION: Final = "1.0.9"

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
CONF_BASE_USAGE_AGGRESSIVE_STRATEGY: Final = "base_usage_aggressive_strategy"

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

# Default values
DEFAULT_PRICE_SENSOR: Final = ""
DEFAULT_VAT_RATE: Final = 21
DEFAULT_TAX: Final = 0.12286
DEFAULT_ADDITIONAL_COST: Final = 0.02398
DEFAULT_CHARGING_WINDOWS: Final = 6
DEFAULT_EXPENSIVE_WINDOWS: Final = 3
DEFAULT_PERCENTILE_THRESHOLD: Final = 25
DEFAULT_MIN_SPREAD: Final = 30
DEFAULT_MIN_SPREAD_DISCHARGE: Final = 30
DEFAULT_AGGRESSIVE_DISCHARGE_SPREAD: Final = 60
DEFAULT_MIN_PRICE_DIFFERENCE: Final = 0.05
DEFAULT_PRICE_OVERRIDE_THRESHOLD: Final = 0.15
DEFAULT_BATTERY_RTE: Final = 85
DEFAULT_CHARGE_POWER: Final = 800
DEFAULT_DISCHARGE_POWER: Final = 800
DEFAULT_BATTERY_SYSTEM_NAME: Final = "My Battery System"
DEFAULT_BATTERY_MIN_SOC_DISCHARGE: Final = 20
DEFAULT_BATTERY_MIN_SOC_AGGRESSIVE_DISCHARGE: Final = 30
DEFAULT_QUIET_START: Final = "22:00:00"
DEFAULT_QUIET_END: Final = "07:00:00"
DEFAULT_TIME_OVERRIDE_START: Final = "00:00:00"
DEFAULT_TIME_OVERRIDE_END: Final = "00:00:00"
DEFAULT_CALCULATION_WINDOW_START: Final = "00:00:00"
DEFAULT_CALCULATION_WINDOW_END: Final = "23:59:59"
DEFAULT_BASE_USAGE: Final = 0
DEFAULT_BASE_USAGE_CHARGE_STRATEGY: Final = "grid_covers_both"
DEFAULT_BASE_USAGE_IDLE_STRATEGY: Final = "battery_covers"
DEFAULT_BASE_USAGE_DISCHARGE_STRATEGY: Final = "subtract_base"
DEFAULT_BASE_USAGE_AGGRESSIVE_STRATEGY: Final = "same_as_discharge"

# Unified price country constants - internal values for backward compatibility
PRICE_COUNTRY_NETHERLANDS: Final = "netherlands"
PRICE_COUNTRY_BELGIUM_ENGIE: Final = "belgium_engie"
PRICE_COUNTRY_OTHER: Final = "other"

# Friendly display names - these are now the primary values
PRICE_COUNTRY_NETHERLANDS_DISPLAY: Final = "Netherlands"
PRICE_COUNTRY_BELGIUM_ENGIE_DISPLAY: Final = "Belgium (ENGIE)"
PRICE_COUNTRY_OTHER_DISPLAY: Final = "Other / Custom"

# Mapping from display names to internal values (for calculation engine)
PRICE_COUNTRY_TO_INTERNAL: Final = {
    PRICE_COUNTRY_NETHERLANDS_DISPLAY: PRICE_COUNTRY_NETHERLANDS,
    PRICE_COUNTRY_BELGIUM_ENGIE_DISPLAY: PRICE_COUNTRY_BELGIUM_ENGIE,
    PRICE_COUNTRY_OTHER_DISPLAY: PRICE_COUNTRY_OTHER,
    # Also support old internal values for backward compatibility
    PRICE_COUNTRY_NETHERLANDS: PRICE_COUNTRY_NETHERLANDS,
    PRICE_COUNTRY_BELGIUM_ENGIE: PRICE_COUNTRY_BELGIUM_ENGIE,
    PRICE_COUNTRY_OTHER: PRICE_COUNTRY_OTHER,
}

# Display names for select entity
PRICE_COUNTRY_DISPLAY_NAMES: Final = {
    PRICE_COUNTRY_NETHERLANDS: PRICE_COUNTRY_NETHERLANDS_DISPLAY,
    PRICE_COUNTRY_BELGIUM_ENGIE: PRICE_COUNTRY_BELGIUM_ENGIE_DISPLAY,
    PRICE_COUNTRY_OTHER: PRICE_COUNTRY_OTHER_DISPLAY,
}

PRICE_COUNTRY_OPTIONS: Final = [
    PRICE_COUNTRY_NETHERLANDS_DISPLAY,
    PRICE_COUNTRY_BELGIUM_ENGIE_DISPLAY,
]

DEFAULT_PRICE_COUNTRY: Final = PRICE_COUNTRY_NETHERLANDS

# Formula descriptions per country - used by dashboard to display formula info
PRICE_FORMULA_INFO: Final = {
    PRICE_COUNTRY_NETHERLANDS: {
        "buy_formula": "(raw × (1+VAT)) + tax + cost",
        "sell_formula": "= buy",
        "fields": ["vat", "tax", "additional_cost"],
    },
    PRICE_COUNTRY_BELGIUM_ENGIE: {
        "buy_formula": "(B × spot + A) × (1+VAT)",
        "sell_formula": "(B × spot − A)",
        "fields": ["buy_formula_param_a", "buy_formula_param_b", "vat"],
    },
    PRICE_COUNTRY_OTHER: {
        "buy_formula": "(B × raw + A) × (1+VAT)",
        "sell_formula": "(B × raw − A)",
        "fields": ["buy_formula_param_a", "buy_formula_param_b", "vat"],
    },
}

# Sell price defaults
DEFAULT_MIN_SELL_PRICE: Final = 0.0
DEFAULT_USE_MIN_SELL_PRICE: Final = False
DEFAULT_MIN_SELL_PRICE_BYPASS_SPREAD: Final = False

# Belgium/Other formula defaults
# Formula: BUY = (B × spot + A) × (1+VAT), SELL = (B × spot − A)
# A = ENGIE cost component in EUR/kWh (0.009 = 0.9 c€/kWh)
# B = Multiplier (1.0 since sensor already provides EUR/kWh)
DEFAULT_BUY_FORMULA_PARAM_A: Final = 0.009  # Cost (A) in EUR/kWh
DEFAULT_BUY_FORMULA_PARAM_B: Final = 1.0    # Multiplier (B)
DEFAULT_SELL_FORMULA_PARAM_A: Final = 0.009  # Cost (A) in EUR/kWh (same as buy for ENGIE)
DEFAULT_SELL_FORMULA_PARAM_B: Final = 1.0    # Multiplier (B)

# Belgium VAT rate (6% since April 2023)
DEFAULT_BELGIUM_VAT: Final = 6

# Base usage strategy options
BASE_USAGE_CHARGE_OPTIONS: Final = ["grid_covers_both", "battery_covers_base"]
BASE_USAGE_IDLE_OPTIONS: Final = ["grid_covers", "battery_covers"]
BASE_USAGE_DISCHARGE_OPTIONS: Final = ["already_included", "subtract_base"]
BASE_USAGE_AGGRESSIVE_OPTIONS: Final = ["same_as_discharge", "already_included", "subtract_base"]

# Update intervals
UPDATE_INTERVAL: Final = timedelta(seconds=10)

# Sensor states
STATE_CHARGE: Final = "charge"
STATE_DISCHARGE: Final = "discharge"
STATE_DISCHARGE_AGGRESSIVE: Final = "discharge_aggressive"
STATE_IDLE: Final = "idle"
STATE_OFF: Final = "off"
STATE_AVAILABLE: Final = "available"
STATE_UNAVAILABLE: Final = "unavailable"

# Battery modes for time overrides
MODE_IDLE: Final = "idle"
MODE_CHARGE: Final = "charge"
MODE_DISCHARGE: Final = "discharge"
MODE_DISCHARGE_AGGRESSIVE: Final = "discharge_aggressive"
MODE_OFF: Final = "off"

# Time override modes list
TIME_OVERRIDE_MODES: Final = [MODE_IDLE, MODE_CHARGE, MODE_DISCHARGE, MODE_DISCHARGE_AGGRESSIVE, MODE_OFF]

# Pricing window duration options
PRICING_15_MINUTES: Final = "15_minutes"
PRICING_1_HOUR: Final = "1_hour"
PRICING_WINDOW_OPTIONS: Final = [PRICING_15_MINUTES, PRICING_1_HOUR]

# Attribute names for sensors
ATTR_CHEAPEST_TIMES: Final = "cheapest_times"
ATTR_CHEAPEST_PRICES: Final = "cheapest_prices"
ATTR_EXPENSIVE_TIMES: Final = "expensive_times"
ATTR_EXPENSIVE_PRICES: Final = "expensive_prices"
ATTR_EXPENSIVE_TIMES_AGGRESSIVE: Final = "expensive_times_aggressive"
ATTR_EXPENSIVE_PRICES_AGGRESSIVE: Final = "expensive_prices_aggressive"
ATTR_ACTUAL_CHARGE_TIMES: Final = "actual_charge_times"
ATTR_ACTUAL_CHARGE_PRICES: Final = "actual_charge_prices"
ATTR_ACTUAL_DISCHARGE_TIMES: Final = "actual_discharge_times"
ATTR_ACTUAL_DISCHARGE_PRICES: Final = "actual_discharge_prices"
ATTR_COMPLETED_CHARGE_WINDOWS: Final = "completed_charge_windows"
ATTR_COMPLETED_DISCHARGE_WINDOWS: Final = "completed_discharge_windows"
ATTR_COMPLETED_CHARGE_COST: Final = "completed_charge_cost"
ATTR_COMPLETED_DISCHARGE_REVENUE: Final = "completed_discharge_revenue"
ATTR_COMPLETED_BASE_USAGE_COST: Final = "completed_base_usage_cost"
ATTR_COMPLETED_BASE_USAGE_BATTERY: Final = "completed_base_usage_battery"
ATTR_TOTAL_COST: Final = "total_cost"
ATTR_PLANNED_TOTAL_COST: Final = "planned_total_cost"
ATTR_NUM_WINDOWS: Final = "num_windows"
ATTR_MIN_SPREAD_REQUIRED: Final = "min_spread_required"
ATTR_SPREAD_PERCENTAGE: Final = "spread_percentage"
ATTR_SPREAD_MET: Final = "spread_met"
ATTR_SPREAD_AVG: Final = "spread_avg"
ATTR_ARBITRAGE_AVG: Final = "arbitrage_avg"
ATTR_ACTUAL_SPREAD_AVG: Final = "actual_spread_avg"
ATTR_DISCHARGE_SPREAD_MET: Final = "discharge_spread_met"
ATTR_AGGRESSIVE_DISCHARGE_SPREAD_MET: Final = "aggressive_discharge_spread_met"
ATTR_AVG_CHEAP_PRICE: Final = "avg_cheap_price"
ATTR_AVG_EXPENSIVE_PRICE: Final = "avg_expensive_price"
ATTR_CURRENT_PRICE: Final = "current_price"
ATTR_PRICE_OVERRIDE_ACTIVE: Final = "price_override_active"
ATTR_TIME_OVERRIDE_ACTIVE: Final = "time_override_active"
ATTR_CURRENT_SELL_PRICE: Final = "current_sell_price"
ATTR_SELL_PRICE_COUNTRY: Final = "sell_price_country"

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
    "min_spread",
    "min_spread_discharge",
    "aggressive_discharge_spread",
    "min_price_difference",

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
    "base_usage_aggressive_strategy",

    # Battery settings
    "battery_rte",
    "charge_power",
    "discharge_power",
    "battery_min_soc_discharge",
    "battery_min_soc_aggressive_discharge",

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

    # Tomorrow settings
    "tomorrow_settings_enabled",
    "charging_windows_tomorrow",
    "expensive_windows_tomorrow",
    "percentile_threshold_tomorrow",
    "min_spread_tomorrow",
    "min_spread_discharge_tomorrow",
    "aggressive_discharge_spread_tomorrow",
    "min_price_difference_tomorrow",
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
    "notify_discharge_aggressive",
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