"""Netherlands pricing formula.

Formula structure:
- Buy: (spot x (1 + VAT)) + tax + additional_cost
- Sell: Same as buy price (net metering until 2027)

Parameters:
- vat: VAT rate as percentage (default: 21%)
- tax: Energy tax in EUR/kWh (default: 0.12286)
- additional_cost: Additional supplier costs in EUR/kWh (default: 0.02398)
"""
from __future__ import annotations

from typing import Any, Dict

from .base import CountryFormula, FormulaParam, ParamType


def buy_formula(spot_kwh: float, params: Dict[str, Any]) -> float:
    """Calculate buy price for Netherlands.

    Formula: (spot x (1 + VAT)) + tax + additional_cost

    Args:
        spot_kwh: Raw spot price in EUR/kWh
        params: Dictionary containing vat, tax, additional_cost

    Returns:
        Calculated buy price in EUR/kWh
    """
    vat = params.get("vat", 21) / 100  # Convert percentage to decimal
    tax = params.get("tax", 0.12286)
    additional_cost = params.get("additional_cost", 0.02398)

    buy_price = (spot_kwh * (1 + vat)) + tax + additional_cost
    return max(0, buy_price)


def sell_formula(spot_kwh: float, buy_price: float, params: Dict[str, Any]) -> float:
    """Calculate sell price for Netherlands.

    Netherlands uses net metering (until 2027), so sell price equals buy price.

    Args:
        spot_kwh: Raw spot price in EUR/kWh
        buy_price: Calculated buy price in EUR/kWh
        params: Dictionary containing formula parameters

    Returns:
        Calculated sell price in EUR/kWh (same as buy price)
    """
    return buy_price


FORMULA = CountryFormula(
    id="netherlands",
    name="Netherlands",
    params=[
        FormulaParam(
            key="vat",
            name="VAT Rate",
            param_type=ParamType.PERCENTAGE,
            default=21,
            min_value=0,
            max_value=50,
            step=0.1,
            icon="mdi:percent",
            description="Value Added Tax percentage"
        ),
        FormulaParam(
            key="tax",
            name="Energy Tax",
            param_type=ParamType.CURRENCY_KWH,
            default=0.12286,
            min_value=0,
            max_value=0.5,
            step=0.00001,
            icon="mdi:cash-plus",
            description="Energy tax per kWh"
        ),
        FormulaParam(
            key="additional_cost",
            name="Additional Cost",
            param_type=ParamType.CURRENCY_KWH,
            default=0.02398,
            min_value=0,
            max_value=0.5,
            step=0.00001,
            icon="mdi:cash-plus",
            description="Additional supplier costs per kWh"
        ),
    ],
    buy_formula=buy_formula,
    sell_formula=sell_formula,
    buy_formula_description="(spot x (1+VAT)) + tax + additional",
    sell_formula_description="Same as buy price"
)
