"""Belgium (ENGIE) pricing formula.

Formula structure:
- Buy: (B x spot + A) x (1 + VAT)
- Sell: (B x spot - A) (no VAT on injection)

Parameters:
- param_b: Multiplier B (default: 1.0)
- param_a: Cost component A in EUR/kWh (default: 0.009 = ENGIE cost)
- vat: VAT rate as percentage (default: 6% since April 2023)

Note: Belgium has reduced VAT on electricity (6%) since April 2023.
Injection (selling) has no VAT applied.
"""
from __future__ import annotations

from typing import Any, Dict

from .base import CountryFormula, FormulaParam, ParamType


def buy_formula(spot_kwh: float, params: Dict[str, Any]) -> float:
    """Calculate buy price for Belgium (ENGIE).

    Formula: (B x spot + A) x (1 + VAT)

    Args:
        spot_kwh: Raw spot price in EUR/kWh
        params: Dictionary containing buy_formula_param_a, buy_formula_param_b, vat

    Returns:
        Calculated buy price in EUR/kWh
    """
    # Support both old and new param names for flexibility
    param_a = params.get("buy_formula_param_a", params.get("param_a", 0.009))
    param_b = params.get("buy_formula_param_b", params.get("param_b", 1.0))
    vat = params.get("vat", 6) / 100  # Convert percentage to decimal

    buy_price = (param_b * spot_kwh + param_a) * (1 + vat)
    return max(0, buy_price)


def sell_formula(spot_kwh: float, _buy_price: float, params: Dict[str, Any]) -> float:
    """Calculate sell price for Belgium (ENGIE).

    Formula: (B x spot - A) - No VAT on injection

    Args:
        spot_kwh: Raw spot price in EUR/kWh
        buy_price: Calculated buy price in EUR/kWh (not used)
        params: Dictionary containing sell_formula_param_a, sell_formula_param_b

    Returns:
        Calculated sell price in EUR/kWh
    """
    # Support both old and new param names for flexibility
    param_a = params.get("sell_formula_param_a", params.get("param_a", 0.009))
    param_b = params.get("sell_formula_param_b", params.get("param_b", 1.0))

    sell_price = (param_b * spot_kwh) - param_a
    return max(0, sell_price)


FORMULA = CountryFormula(
    id="belgium_engie",
    name="Belgium (ENGIE)",
    params=[
        # Buy formula parameters
        FormulaParam(
            key="buy_formula_param_b",
            name="Buy Multiplier (B)",
            param_type=ParamType.MULTIPLIER,
            default=1.0,
            min_value=0,
            max_value=2,
            step=0.01,
            icon="mdi:alpha-b-circle",
            description="Spot price multiplier for buy formula"
        ),
        FormulaParam(
            key="buy_formula_param_a",
            name="Buy Cost (A)",
            param_type=ParamType.CURRENCY_KWH,
            default=0.009,
            min_value=0,
            max_value=0.1,
            step=0.001,
            icon="mdi:alpha-a-circle",
            description="ENGIE cost component per kWh for buy formula"
        ),
        # Sell formula parameters
        FormulaParam(
            key="sell_formula_param_b",
            name="Sell Multiplier (B)",
            param_type=ParamType.MULTIPLIER,
            default=1.0,
            min_value=0,
            max_value=2,
            step=0.01,
            icon="mdi:alpha-b-circle",
            description="Spot price multiplier for sell formula"
        ),
        FormulaParam(
            key="sell_formula_param_a",
            name="Sell Cost (A)",
            param_type=ParamType.CURRENCY_KWH,
            default=0.009,
            min_value=0,
            max_value=0.1,
            step=0.001,
            icon="mdi:alpha-a-circle",
            description="ENGIE cost component per kWh for sell formula"
        ),
        # VAT (applied to buy formula only)
        FormulaParam(
            key="vat",
            name="VAT Rate",
            param_type=ParamType.PERCENTAGE,
            default=6,
            min_value=0,
            max_value=50,
            step=0.1,
            icon="mdi:percent",
            description="Belgian VAT (6% since April 2023)"
        ),
    ],
    buy_formula=buy_formula,
    sell_formula=sell_formula,
    buy_formula_description="(B x spot + A) x (1+VAT)",
    sell_formula_description="(B x spot - A)"
)
