"""Config flow for Cheapest Energy Windows integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    CONF_PRICE_SENSOR,
    CONF_BASE_USAGE,
    CONF_BASE_USAGE_CHARGE_STRATEGY,
    CONF_BASE_USAGE_NORMAL_STRATEGY,
    CONF_BASE_USAGE_DISCHARGE_STRATEGY,
    CONF_PRICE_COUNTRY,
    CONF_BUY_FORMULA_PARAM_A,
    CONF_BUY_FORMULA_PARAM_B,
    CONF_MIN_SELL_PRICE,
    CONF_USE_MIN_SELL_PRICE,
    CONF_MIN_SELL_PRICE_BYPASS_SPREAD,
    CONF_SELL_FORMULA_PARAM_A,
    CONF_SELL_FORMULA_PARAM_B,
    CONF_BATTERY_SYSTEM_NAME,
    CONF_BATTERY_SOC_SENSOR,
    CONF_BATTERY_ENERGY_SENSOR,
    CONF_BATTERY_CHARGE_SENSOR,
    CONF_BATTERY_DISCHARGE_SENSOR,
    CONF_BATTERY_POWER_SENSOR,
    CONF_VAT_RATE,
    CONF_TAX,
    CONF_ADDITIONAL_COST,
    DEFAULT_PRICE_SENSOR,
    DEFAULT_BASE_USAGE,
    DEFAULT_BASE_USAGE_CHARGE_STRATEGY,
    DEFAULT_BASE_USAGE_NORMAL_STRATEGY,
    DEFAULT_BASE_USAGE_DISCHARGE_STRATEGY,
    DEFAULT_PRICE_COUNTRY,
    DEFAULT_BUY_FORMULA_PARAM_A,
    DEFAULT_BUY_FORMULA_PARAM_B,
    DEFAULT_MIN_SELL_PRICE,
    DEFAULT_USE_MIN_SELL_PRICE,
    DEFAULT_MIN_SELL_PRICE_BYPASS_SPREAD,
    DEFAULT_SELL_FORMULA_PARAM_A,
    DEFAULT_SELL_FORMULA_PARAM_B,
    DEFAULT_VAT_RATE,
    DEFAULT_TAX,
    DEFAULT_ADDITIONAL_COST,
    BASE_USAGE_CHARGE_OPTIONS,
    BASE_USAGE_NORMAL_OPTIONS,
    BASE_USAGE_DISCHARGE_OPTIONS,
    DEFAULT_CHARGE_POWER,
    DEFAULT_DISCHARGE_POWER,
    DEFAULT_BATTERY_RTE,
    DEFAULT_CHARGING_WINDOWS,
    DEFAULT_EXPENSIVE_WINDOWS,
    DEFAULT_PERCENTILE_THRESHOLD,
    DEFAULT_MIN_PROFIT_CHARGE,
    DEFAULT_MIN_PROFIT_DISCHARGE,
    DEFAULT_MIN_PRICE_DIFFERENCE,
    DEFAULT_PRICE_OVERRIDE_THRESHOLD,
    PRICING_15_MINUTES,
    PRICING_1_HOUR,
    LOGGER_NAME,
    PREFIX,
)
from .formulas import get_country_options, get_formula

_LOGGER = logging.getLogger(LOGGER_NAME)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect."""

    # Check if price sensor exists
    price_sensor = data.get(CONF_PRICE_SENSOR)
    if price_sensor:
        sensor_state = hass.states.get(price_sensor)
        if not sensor_state:
            raise ValueError(f"Price sensor {price_sensor} not found")

        # Check if it has the required attributes
        if not hasattr(sensor_state, 'attributes'):
            raise ValueError(f"Price sensor {price_sensor} has no attributes")

        attrs = sensor_state.attributes

        # Check for either Nord Pool or ENTSO-E format
        has_nordpool = 'raw_today' in attrs and 'raw_tomorrow' in attrs
        has_entsoe = 'prices_today' in attrs or 'prices_tomorrow' in attrs

        if not has_nordpool and not has_entsoe:
            raise ValueError(f"Price sensor {price_sensor} missing required attributes. Need either 'raw_today'/'raw_tomorrow' (Nord Pool) or 'prices_today'/'prices_tomorrow' (ENTSO-E)")

        # ENTSO-E sensors don't have price_in_cents, only check for Nord Pool
        if has_nordpool and attrs.get('price_in_cents') is True:
            raise ValueError(f"Price sensor {price_sensor} uses cents/kWh. Only EUR/kWh sensors are supported.")

    return {"title": "Cheapest Energy Windows"}


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Cheapest Energy Windows."""

    VERSION = 1

    def __init__(self):
        """Initialize the config flow."""
        self.data = {}
        self.options = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        # Directly start guided setup
        return await self.async_step_dependencies()

    async def async_step_dependencies(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Check for required dependencies and auto-continue."""
        # Auto-continue to price sensor (dependencies info shown in price sensor step)
        return await self.async_step_price_sensor()

    async def async_step_price_sensor(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure the price sensor."""
        errors = {}

        if user_input is not None:
            # Validate the price sensor
            try:
                await validate_input(self.hass, user_input)
                self.data.update(user_input)
                return await self.async_step_price_formulas()
            except ValueError as e:
                errors["base"] = "invalid_price_sensor"
                _LOGGER.error(f"Price sensor validation failed: {e}")

        # Try to auto-detect price sensors (both Nord Pool and ENTSO-E formats)
        price_sensors = []
        entsoe_sensors = []
        nordpool_sensors = []

        for state in self.hass.states.async_all("sensor"):
            attrs = state.attributes

            # Check for Nord Pool format
            if attrs.get("raw_today") is not None:
                # Exclude sensors with price_in_cents
                if attrs.get("price_in_cents") is True:
                    continue
                nordpool_sensors.append(state.entity_id)
                price_sensors.append(state.entity_id)

            # Check for ENTSO-E format
            elif attrs.get("prices_today") is not None:
                entsoe_sensors.append(state.entity_id)
                price_sensors.append(state.entity_id)

        # Show error if no sensors found
        if not price_sensors:
            return self.async_show_form(
                step_id="price_sensor",
                data_schema=vol.Schema({}),
                errors={"base": "no_price_sensors"},
                description_placeholders={
                    "info": "⚠️ No compatible price sensors found!\n\nPlease install a compatible price sensor first. Supported sensors:\n• Nordpool (HACS)\n• ENTSO-E (HACS)\n\nVisit our GitHub page for setup instructions.\n\nThe sensor must have a 'raw_today' (Nordpool) or 'prices_today' (ENTSO-E) attribute with hourly or 15-minute price data."
                },
            )

        # Build sensor list with format indicators
        sensor_list = []
        for sensor in price_sensors[:5]:
            if sensor in entsoe_sensors:
                sensor_list.append(f"- {sensor} (ENTSO-E)")
            else:
                sensor_list.append(f"- {sensor} (Nord Pool)")

        # Add sensor format notes
        sensor_note = ""
        if nordpool_sensors or entsoe_sensors:
            sensor_note = "\n\n📝 **Sensor Requirements:**\n• **15-minute interval sensor required** - The integration needs 15-minute price data for optimal window calculation\n• If you have hourly pricing contracts, the system will automatically aggregate 15-minute data into hourly windows\n• Both Nord Pool and ENTSO-E 15-minute sensors are supported"

        # Show available sensors for selection (no default)
        return self.async_show_form(
            step_id="price_sensor",
            data_schema=vol.Schema({
                vol.Required(CONF_PRICE_SENSOR): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="sensor",
                        multiple=False,
                    )
                ),
            }),
            errors=errors,
            description_placeholders={
                "info": f"✅ Detected {len(price_sensors)} compatible price sensor(s)\n\n⚠️ **IMPORTANT - Price Unit Requirement:**\nYour price sensor MUST use EUR/kWh (e.g., 0.25), NOT cents (e.g., 25).\nSensors configured for cents/kWh are currently not supported and will cause incorrect calculations.\n\nPlease select your price sensor:\n{chr(10).join(sensor_list)}\n\nSupported sensor formats:\n• Nord Pool: 'raw_today'/'raw_tomorrow' attributes\n• ENTSO-E: 'prices_today'/'prices_tomorrow' attributes{sensor_note}"
            },
        )

    async def async_step_price_formulas(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step 1: Select price country/formula type."""
        if user_input is not None:
            self.data.update(user_input)
            # Route to appropriate formula step based on country selection
            country = user_input.get(CONF_PRICE_COUNTRY, DEFAULT_PRICE_COUNTRY)
            if country == "netherlands":
                return await self.async_step_netherlands_formulas()
            else:
                return await self.async_step_custom_formulas()

        # Get available countries from the formula registry
        country_options = get_country_options()

        return self.async_show_form(
            step_id="price_formulas",
            data_schema=vol.Schema({
                vol.Required(CONF_PRICE_COUNTRY, default=DEFAULT_PRICE_COUNTRY): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"label": name, "value": country_id}
                            for country_id, name in country_options
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }),
            description_placeholders={
                "info": "🌍 **Select Your Electricity Contract Type**\n\n"
                       "Each country has different ways of calculating your final electricity price from the spot price. "
                       "This includes things like VAT, energy taxes, and supplier fees.\n\n"
                       "Select your country/contract type from the dropdown. The next step will show the specific formula and let you configure the parameters.\n\n"
                       "🌐 **Your country not listed?**\n"
                       "Request it on our GitHub: github.com/cheapest-energy-windows/cheapest_energy_windows\n\n"
                       "💡 All parameters can be adjusted after installation via the dashboard."
            },
        )

    async def async_step_netherlands_formulas(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure Netherlands-specific pricing parameters."""
        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_sell_settings()

        return self.async_show_form(
            step_id="netherlands_formulas",
            data_schema=vol.Schema({
                vol.Required(CONF_VAT_RATE, default=DEFAULT_VAT_RATE): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=100,
                        step=1,
                        unit_of_measurement="%",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(CONF_TAX, default=DEFAULT_TAX): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=1.0,
                        step=0.001,
                        unit_of_measurement="EUR/kWh",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(CONF_ADDITIONAL_COST, default=DEFAULT_ADDITIONAL_COST): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=1.0,
                        step=0.001,
                        unit_of_measurement="EUR/kWh",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            }),
            description_placeholders={
                "info": "🇳🇱 **Netherlands Pricing Configuration**\n\n"
                       "Formula: `(spot_price × (1 + VAT/100)) + energy_tax + additional_cost`\n\n"
                       "**VAT Rate (BTW)**\n"
                       "Current standard rate is 21%. Reduced rate 9% for some energy.\n\n"
                       "**Energy Tax (Energiebelasting)**\n"
                       "Government tax per kWh. Check your invoice for exact amount.\n"
                       "2024 typical: €0.12286/kWh\n\n"
                       "**Additional Cost (Opslag)**\n"
                       "Supplier markup, transport costs, etc.\n"
                       "Typical: €0.02-0.05/kWh\n\n"
                       "💡 Check your energy invoice for exact values."
            },
        )

    async def async_step_custom_formulas(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure custom/Belgium pricing parameters."""
        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_sell_settings()

        country = self.data.get(CONF_PRICE_COUNTRY, DEFAULT_PRICE_COUNTRY)
        formula = get_formula(country)

        # Get default VAT from the formula registry
        default_vat = DEFAULT_VAT_RATE
        default_param_a = DEFAULT_BUY_FORMULA_PARAM_A
        default_param_b = DEFAULT_BUY_FORMULA_PARAM_B
        if formula:
            for param in formula.params:
                if param.key == "vat":
                    default_vat = param.default
                elif param.key == "buy_formula_param_a":
                    default_param_a = param.default
                elif param.key == "buy_formula_param_b":
                    default_param_b = param.default

        title = f"**{formula.name} Pricing**" if formula else "🔧 **Custom Formula Configuration**"
        if formula:
            formula_info = (
                f"**Formula:**\n"
                f"• BUY: `{formula.buy_formula_description}`\n"
                f"• SELL: `{formula.sell_formula_description}`\n\n"
                "**Parameters:**\n"
                "• **Multiplier (B)**: Spot price multiplier\n"
                "• **Cost (A)**: Supplier cost in EUR/kWh\n"
                f"• **VAT**: Default {default_vat}%\n\n"
                "💡 These values are from your energy contract/tariff card."
            )
        else:
            formula_info = (
                "**Custom Formula:**\n"
                "• BUY: `(B × spot + A) × (1 + VAT)`\n"
                "• SELL: `(B × spot − A)`\n\n"
                "**Parameters:**\n"
                "• **Multiplier (B)**: Spot price multiplier (usually 1.0)\n"
                "• **Cost (A)**: Supplier cost in EUR/kWh\n"
                "• **VAT**: Your country's VAT rate %\n"
            )

        return self.async_show_form(
            step_id="custom_formulas",
            data_schema=vol.Schema({
                vol.Required(CONF_BUY_FORMULA_PARAM_B, default=default_param_b): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=2.0,
                        step=0.01,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(CONF_BUY_FORMULA_PARAM_A, default=default_param_a): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=-0.1,
                        max=0.5,
                        step=0.001,
                        unit_of_measurement="EUR/kWh",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(CONF_VAT_RATE, default=default_vat): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=50,
                        step=1,
                        unit_of_measurement="%",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            }),
            description_placeholders={
                "info": f"{title}\n\n{formula_info}\n"
                       "💡 These parameters are for the BUY formula.\n"
                       "💡 Sell formula parameters can be configured via the dashboard after setup.\n"
                       "💡 VAT only applies to BUY price, not SELL."
            },
        )

    async def async_step_sell_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure minimum sell price settings."""
        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_base_usage()

        return self.async_show_form(
            step_id="sell_settings",
            data_schema=vol.Schema({
                vol.Required(CONF_USE_MIN_SELL_PRICE, default=DEFAULT_USE_MIN_SELL_PRICE): selector.BooleanSelector(),
                vol.Optional(CONF_MIN_SELL_PRICE, default=DEFAULT_MIN_SELL_PRICE): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=-0.5,
                        max=1.0,
                        step=0.01,
                        unit_of_measurement="EUR/kWh",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(CONF_MIN_SELL_PRICE_BYPASS_SPREAD, default=DEFAULT_MIN_SELL_PRICE_BYPASS_SPREAD): selector.BooleanSelector(),
            }),
            description_placeholders={
                "info": "💰 **Minimum Sell Price Settings**\n\n"
                       "**Use Minimum Sell Price**\n"
                       "Only discharge/export when the sell price exceeds a threshold.\n"
                       "Useful to avoid selling at very low or negative prices.\n\n"
                       "**Minimum Sell Price**\n"
                       "The threshold in EUR/kWh. Set to 0 to only block negative prices.\n\n"
                       "**Bypass Spread Check**\n"
                       "When enabled, the minimum sell price check bypasses the spread requirement.\n"
                       "This means: if price > min_sell_price, allow discharge even if spread isn't met.\n\n"
                       "💡 Leave disabled if you're not sure what to configure."
            },
        )

    async def async_step_base_usage(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure base usage tracking."""
        if user_input is not None:
            self.options.update(user_input)
            return await self.async_step_power()

        return self.async_show_form(
            step_id="base_usage",
            data_schema=vol.Schema({
                vol.Optional(CONF_BASE_USAGE, default=DEFAULT_BASE_USAGE): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=5000,
                        step=50,
                        unit_of_measurement="W",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(CONF_BASE_USAGE_CHARGE_STRATEGY, default=DEFAULT_BASE_USAGE_CHARGE_STRATEGY): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"label": "Grid powers house + charging", "value": "grid_covers_both"},
                            {"label": "Battery powers house during charging", "value": "battery_covers_base"},
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        translation_key="base_usage_charge_strategy",
                    )
                ),
                vol.Optional(CONF_BASE_USAGE_NORMAL_STRATEGY, default=DEFAULT_BASE_USAGE_NORMAL_STRATEGY): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"label": "Battery powers house", "value": "battery_covers"},
                            {"label": "Grid powers house", "value": "grid_covers"},
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        translation_key="base_usage_normal_strategy",
                    )
                ),
                vol.Optional(CONF_BASE_USAGE_DISCHARGE_STRATEGY, default=DEFAULT_BASE_USAGE_DISCHARGE_STRATEGY): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"label": "House first, export remainder", "value": "subtract_base"},
                            {"label": "Export full discharge power", "value": "already_included"},
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        translation_key="base_usage_discharge_strategy",
                    )
                ),
            }),
            description_placeholders={
                "info": "📊 **What is Base Usage?**\n"
                       "Track your household's constant power consumption (base load) - the power used by always-on appliances, lights, standby devices, etc.\n\n"
                       "📈 **Why configure it?**\n"
                       "Accurate cost tracking. Without this, only battery charge/discharge costs are calculated. With base usage, you get complete daily energy costs.\n\n"
                       "⚙️ **Strategy Configuration:**\n"
                       "Configure how your base load is handled during each battery state:\n\n"
                       "**During Charging** - Who powers the house while battery charges?\n"
                       "• Grid powers house + charging: Grid provides all power (default)\n"
                       "• Battery powers house: Battery covers household, grid only charges\n\n"
                       "**During Normal** - Who powers the house when battery is in normal mode?\n"
                       "• Grid powers house: Normal grid consumption (default)\n"
                       "• Battery powers house: Battery covers base load (zero-meter strategy)\n\n"
                       "**During Discharge** - How is export calculated?\n"
                       "• House first, export remainder: Battery powers house, exports what's left (default)\n"
                       "• Export full discharge power: Full discharge goes to grid\n\n"
                       "💡 **Default: 0W** - Leave at zero to disable base usage tracking."
            },
        )

    async def async_step_power(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure battery power parameters."""
        if user_input is not None:
            # Store power values in options (convert to int for consistency)
            self.options["charge_power"] = int(user_input.get("charge_power", DEFAULT_CHARGE_POWER))
            self.options["discharge_power"] = int(user_input.get("discharge_power", DEFAULT_DISCHARGE_POWER))
            self.options["battery_rte"] = int(user_input.get("battery_rte", DEFAULT_BATTERY_RTE))
            return await self.async_step_pricing_windows()

        return self.async_show_form(
            step_id="power",
            data_schema=vol.Schema({
                vol.Required("charge_power", default=DEFAULT_CHARGE_POWER): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=100,
                        max=10000,
                        step=100,
                        unit_of_measurement="W",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required("discharge_power", default=DEFAULT_DISCHARGE_POWER): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=100,
                        max=10000,
                        step=100,
                        unit_of_measurement="W",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required("battery_rte", default=DEFAULT_BATTERY_RTE): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=50,
                        max=100,
                        step=1,
                        unit_of_measurement="%",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            }),
            description_placeholders={
                "info": "🔋 **Battery Power & Efficiency**\n\n"
                       "**Charge Power**\n"
                       "Maximum power when charging your battery.\n"
                       "Typical: 800W for single home battery, 1600W+ for dual\n\n"
                       "**Discharge Power**\n"
                       "Maximum power when discharging/exporting.\n"
                       "Often same as charge power.\n\n"
                       "**Round-Trip Efficiency (RTE)**\n"
                       "Energy retained after charge+discharge cycle.\n"
                       "Typical: 85% (100 kWh in → 85 kWh out)\n"
                       "This affects profitability calculations.\n\n"
                       "💡 Check your battery specs for exact values."
            },
        )

    async def async_step_pricing_windows(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure pricing window duration and spread settings."""
        if user_input is not None:
            # Store all pricing window settings in options
            self.options.update(user_input)
            return await self.async_step_battery()

        return self.async_show_form(
            step_id="pricing_windows",
            data_schema=vol.Schema({
                vol.Required("pricing_window_duration", default=PRICING_1_HOUR): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"label": "15 Minutes (96 windows per day)", "value": PRICING_15_MINUTES},
                            {"label": "1 Hour (24 windows per day)", "value": PRICING_1_HOUR},
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required("charging_windows", default=DEFAULT_CHARGING_WINDOWS): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=96,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required("expensive_windows", default=DEFAULT_EXPENSIVE_WINDOWS): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=96,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required("percentile_threshold", default=DEFAULT_PERCENTILE_THRESHOLD): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1,
                        max=50,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required("min_profit_charge", default=DEFAULT_MIN_PROFIT_CHARGE): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=-100,
                        max=200,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required("min_profit_discharge", default=DEFAULT_MIN_PROFIT_DISCHARGE): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=-100,
                        max=200,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional("min_price_difference", default=DEFAULT_MIN_PRICE_DIFFERENCE): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=0.5,
                        step=0.01,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required("price_override_enabled", default=False): selector.BooleanSelector(),
                vol.Optional("price_override_threshold", default=DEFAULT_PRICE_OVERRIDE_THRESHOLD): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=0.5,
                        step=0.01,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            }),
            description_placeholders={
                "info": "⚡ **Pricing Windows Configuration**\n\n"
                       "**Charging Windows**\n"
                       "Number of cheapest time slots to use for charging.\n\n"
                       "**Discharge Windows**\n"
                       "Number of most expensive slots for discharging.\n\n"
                       "**Percentile Threshold**\n"
                       "Only use windows in the cheapest/most expensive X%.\n\n"
                       "**Profit Settings**\n"
                       "Profit = Spread - RTE Loss. Your battery loses energy during charge/discharge cycles.\n"
                       "Example: 25% price spread - 15% RTE loss = 10% actual profit.\n"
                       "Set the minimum profit % you want before the system acts.\n\n"
                       "**Price Override**\n"
                       "Always charge when price drops below threshold, ignoring other rules.\n\n"
                       "💡 These can be adjusted later via the dashboard."
            },
        )

    async def async_step_battery(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure battery system (optional)."""
        if user_input is not None:
            # Check if user actually entered any battery data
            battery_data = {
                k: v for k, v in user_input.items()
                if v is not None and v != "" and v != "not_configured"
            }

            # Save to both data and options so entities can access it
            if battery_data:
                self.data.update(battery_data)
                self.options.update(battery_data)

            return await self.async_step_battery_operations()

        return self.async_show_form(
            step_id="battery",
            data_schema=vol.Schema({
                vol.Optional(CONF_BATTERY_SYSTEM_NAME): cv.string,
                vol.Optional(CONF_BATTERY_SOC_SENSOR): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="sensor",
                        multiple=False,
                    )
                ),
                vol.Optional(CONF_BATTERY_ENERGY_SENSOR): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="sensor",
                        multiple=False,
                    )
                ),
                vol.Optional(CONF_BATTERY_CHARGE_SENSOR): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="sensor",
                        multiple=False,
                    )
                ),
                vol.Optional(CONF_BATTERY_DISCHARGE_SENSOR): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="sensor",
                        multiple=False,
                    )
                ),
                vol.Optional(CONF_BATTERY_POWER_SENSOR): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="sensor",
                        multiple=False,
                    )
                ),
            }),
            description_placeholders={
                "info": "Optional: Configure battery system sensors for monitoring and automation.\n\nLeave fields empty to skip battery configuration.\n\nYou can configure these later through the integration settings."
            },
        )

    async def async_step_battery_operations(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure battery operation automations."""
        if user_input is not None:
            # Set default "not_configured" for any empty/missing fields
            battery_ops = {
                "battery_normal_action": user_input.get("battery_normal_action", "not_configured"),
                "battery_charge_action": user_input.get("battery_charge_action", "not_configured"),
                "battery_discharge_action": user_input.get("battery_discharge_action", "not_configured"),
                "battery_off_action": user_input.get("battery_off_action", "not_configured"),
            }
            self.data.update(battery_ops)
            return await self.async_step_automation()

        return self.async_show_form(
            step_id="battery_operations",
            data_schema=vol.Schema({
                vol.Optional("battery_normal_action"): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["automation", "script", "scene"],
                        multiple=False,
                    )
                ),
                vol.Optional("battery_charge_action"): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["automation", "script", "scene"],
                        multiple=False,
                    )
                ),
                vol.Optional("battery_discharge_action"): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["automation", "script", "scene"],
                        multiple=False,
                    )
                ),
                vol.Optional("battery_off_action"): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["automation", "script", "scene"],
                        multiple=False,
                    )
                ),
            }),
            description_placeholders={
                "info": "⚙️ **Battery Operations (Optional)**\n\nLink existing automations, scripts, or scenes to battery modes. They'll be triggered automatically when modes change.\n\n**How it works:**\n- Create your battery control automations/scripts first\n- Select them from the dropdowns below\n- CEW will automatically trigger them when entering each mode\n\nLeave blank to configure later in Settings → Battery Operations."
            },
        )

    async def async_step_automation(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Create notification automation."""
        errors = {}

        if user_input is not None:
            # Import the function here to avoid circular imports
            from .services import async_create_notification_automation

            # Create the automation
            success, message = await async_create_notification_automation(self.hass)

            if success:
                _LOGGER.info(f"Automation creation: {message}")
                # Store success message to show in confirmation step
                self.options["_automation_created"] = True
                self.options["_automation_message"] = message
                return await self.async_step_dashboard()
            else:
                _LOGGER.error(f"Automation creation failed: {message}")
                errors["base"] = "automation_creation_failed"
                # Store error message for display
                self.options["_automation_error"] = message

        return self.async_show_form(
            step_id="automation",
            data_schema=vol.Schema({}),
            errors=errors,
            description_placeholders={
                "info": "🤖 **Create Battery Control Automation**\n\n"
                       "A battery control automation will be created automatically for you.\n\n"
                       "**What it provides:**\n"
                       "- Triggers on CEW state changes (charge, discharge, normal, off)\n"
                       "- Automatically calls YOUR linked automations/scripts/scenes\n"
                       "- Handles notifications (configured via dashboard switches)\n\n"
                       "**What it does NOT provide:**\n"
                       "- ❌ Battery device actions (you link those via Battery Operations in the Dashboard)\n\n"
                       "**After setup:**\n"
                       "1. Your linked automations/scripts will be called automatically based on CEW state\n"
                       "2. Configure notification preferences via the dashboard switches\n"
                       "3. The automation is auto-managed - updates automatically with new CEW releases\n\n"
                       "**Important switches to configure:**\n"
                       "- switch.cew_automation_enabled (master switch - MUST be ON)\n"
                       "- switch.cew_notifications_enabled (enables notifications)\n"
                       "- Individual notification toggles (notify_charging, notify_discharge, etc.)\n\n"
                       "**Note:** This automation is managed by CEW. Manual edits will be overwritten on updates.\n\n"
                       "Click **Submit** to create the automation."
            },
        )

    async def async_step_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Confirm configuration and complete setup."""
        if user_input is not None:
            # Complete setup
            return self.async_create_entry(
                title="Cheapest Energy Windows",
                data=self.data,
                options=self.options,
            )

        # Show summary of what will be created
        charge_power = self.options.get("charge_power", DEFAULT_CHARGE_POWER)
        discharge_power = self.options.get("discharge_power", DEFAULT_DISCHARGE_POWER)
        pricing_duration = self.options.get("pricing_window_duration", PRICING_15_MINUTES)
        charging_windows = self.options.get("charging_windows", DEFAULT_CHARGING_WINDOWS)
        expensive_windows = self.options.get("expensive_windows", DEFAULT_EXPENSIVE_WINDOWS)

        battery_configured = self.data.get(CONF_BATTERY_SYSTEM_NAME) is not None
        automation_created = self.options.get("_automation_created", False)
        automation_message = self.options.get("_automation_message", "")

        price_country = self.data.get(CONF_PRICE_COUNTRY, DEFAULT_PRICE_COUNTRY)

        summary = f"""
Configuration Summary:
- Price Sensor: {self.data.get(CONF_PRICE_SENSOR, DEFAULT_PRICE_SENSOR)}
- Price Country: {price_country}
- Charge Power: {charge_power}W
- Discharge Power: {discharge_power}W
- Pricing Duration: {pricing_duration.replace('_', ' ').title()}
- Charging Windows: {charging_windows}
- Discharge Windows: {expensive_windows}
- Battery Configured: {'Yes' if battery_configured else 'No'}

This will create:
- 2 sensors (CEW Today, CEW Tomorrow)
- 26 number entities (pricing, power, battery config)
- 26 switch entities (automation toggles, battery display)
- 8 select entities (modes, duration, price formulas)
- 6 time entities (schedules, overrides)
- 6 text entities (sensor entity IDs, battery config)

Total: 74 entities
"""

        if automation_created:
            summary += f"\n✅ Automation Status: {automation_message}\n"
            summary += "Find it in Settings → Automations & Scenes\n"

        summary += "\nClick Submit to complete setup!"

        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders={
                "summary": summary,
            },
        )

    async def async_step_dashboard(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show dashboard installation instructions."""
        if user_input is not None:
            # Move to confirmation step
            return await self.async_step_confirm()

        return self.async_show_form(
            step_id="dashboard",
            data_schema=vol.Schema({}),
            description_placeholders={
                "info": "📊 **Dashboard Available via HACS**\n\n"
                       "A beautiful, pre-configured dashboard is available as a separate HACS plugin.\n\n"
                       "**Why install from HACS?**\n"
                       "✅ Automatic updates when improvements are released\n"
                       "✅ One-click installation\n"
                       "✅ Always stays in sync with integration features\n\n"
                       "**To install the dashboard:**\n\n"
                       "1. Go to **HACS** → **Frontend**\n"
                       "2. Click **Explore & Download Repositories**\n"
                       "3. Search for **\"Cheapest Energy Windows Dashboard\"**\n"
                       "4. Click **Download**\n"
                       "5. Follow the HACS installation instructions\n"
                       "6. The dashboard will appear in your sidebar automatically\n\n"
                       "**Dashboard Features:**\n"
                       "- Real-time energy price monitoring with ApexCharts visualizations\n"
                       "- Visual charge/discharge windows display\n"
                       "- Battery system status and metrics\n"
                       "- Quick access to all settings in collapsible sections\n"
                       "- Responsive mobile-friendly design\n\n"
                       "Click **Submit** to complete the setup."
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Get the options flow for this handler."""
        return CEWOptionsFlow(config_entry)


class CEWOptionsFlow(config_entries.OptionsFlow):
    """Handle options for Cheapest Energy Windows."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        options = self.config_entry.options

        # Get available countries from the formula registry
        country_options = get_country_options()

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Optional(
                    CONF_PRICE_SENSOR,
                    default=options.get(
                        CONF_PRICE_SENSOR,
                        self.config_entry.data.get(CONF_PRICE_SENSOR, DEFAULT_PRICE_SENSOR)
                    ),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="sensor",
                        multiple=False,
                    )
                ),
                vol.Optional(
                    CONF_PRICE_COUNTRY,
                    default=options.get(
                        CONF_PRICE_COUNTRY,
                        self.config_entry.data.get(CONF_PRICE_COUNTRY, DEFAULT_PRICE_COUNTRY)
                    ),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"label": name, "value": country_id}
                            for country_id, name in country_options
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        translation_key="price_country",
                    )
                ),
            }),
        )