"""Base classes for country formula definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Any, List, Optional


class ParamType(Enum):
    """Parameter types for formula parameters."""

    PERCENTAGE = "percentage"      # 0-100, displayed as %
    CURRENCY_KWH = "currency_kwh"  # EUR/kWh
    MULTIPLIER = "multiplier"      # Decimal multiplier


@dataclass
class FormulaParam:
    """Definition of a formula parameter.

    Attributes:
        key: Internal key used in config (e.g., "vat")
        name: Display name shown in UI (e.g., "VAT Rate")
        param_type: Type of parameter for UI hints
        default: Default value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        step: Step increment for UI controls
        icon: MDI icon name for UI
        description: Optional description for tooltips
    """

    key: str
    name: str
    param_type: ParamType
    default: float
    min_value: float = 0
    max_value: float = 100
    step: float = 0.01
    icon: str = "mdi:numeric"
    description: str = ""

    def get_unit(self) -> str:
        """Get the unit string for this parameter type."""
        if self.param_type == ParamType.PERCENTAGE:
            return "%"
        elif self.param_type == ParamType.CURRENCY_KWH:
            return "EUR/kWh"
        elif self.param_type == ParamType.MULTIPLIER:
            return ""
        return ""


@dataclass
class CountryFormula:
    """Definition of a country-specific pricing formula.

    Each country formula file should define a FORMULA constant of this type.
    The formula will be auto-discovered by the registry.

    Attributes:
        id: Internal ID (e.g., "netherlands") - must be unique
        name: Display name (e.g., "Netherlands") - shown in UI
        params: List of parameters required by this formula
        buy_formula: Function to calculate buy price: (spot_kwh, params) -> price
        sell_formula: Function to calculate sell price: (spot_kwh, buy_price, params) -> price
        buy_formula_description: Human-readable formula description for buy
        sell_formula_description: Human-readable formula description for sell
    """

    id: str
    name: str
    params: List[FormulaParam]
    buy_formula: Callable[[float, Dict[str, Any]], float]
    sell_formula: Callable[[float, float, Dict[str, Any]], float]
    buy_formula_description: str
    sell_formula_description: str

    def get_param(self, key: str) -> Optional[FormulaParam]:
        """Get a parameter by key."""
        for param in self.params:
            if param.key == key:
                return param
        return None

    def get_param_keys(self) -> List[str]:
        """Get list of parameter keys."""
        return [p.key for p in self.params]

    def get_defaults(self) -> Dict[str, float]:
        """Get default values for all parameters."""
        return {p.key: p.default for p in self.params}
