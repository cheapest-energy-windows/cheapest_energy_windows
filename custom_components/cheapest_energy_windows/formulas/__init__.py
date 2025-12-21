"""Country formula registry with auto-discovery.

This module provides auto-discovery of country formula definitions.
To add a new country, simply create a new .py file in this directory
with a FORMULA constant of type CountryFormula.

Example:
    # formulas/france.py
    from .base import CountryFormula, FormulaParam, ParamType

    def buy_formula(spot_kwh: float, params: dict) -> float:
        return (spot_kwh + params["turpe"]) * (1 + params["vat"]/100)

    def sell_formula(spot_kwh: float, buy_price: float, params: dict) -> float:
        return spot_kwh * 0.95

    FORMULA = CountryFormula(
        id="france",
        name="France",
        params=[...],
        buy_formula=buy_formula,
        sell_formula=sell_formula,
        buy_formula_description="(spot + TURPE) x (1+TVA)",
        sell_formula_description="spot x 0.95"
    )
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .base import CountryFormula, FormulaParam, ParamType

_LOGGER = logging.getLogger(__name__)

# Registry of discovered formulas
_registry: Dict[str, CountryFormula] = {}
_discovered: bool = False


def discover_formulas() -> Dict[str, CountryFormula]:
    """Auto-discover all country formula modules.

    Scans this package directory for Python modules containing a FORMULA
    constant of type CountryFormula.

    Returns:
        Dictionary mapping country ID to CountryFormula instance.
    """
    global _registry, _discovered

    if _discovered:
        return _registry

    package_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        # Skip __init__ and base modules
        if module_info.name in ("__init__", "base"):
            continue

        try:
            module = importlib.import_module(f".{module_info.name}", package=__name__)

            if hasattr(module, "FORMULA"):
                formula = module.FORMULA
                if isinstance(formula, CountryFormula):
                    _registry[formula.id] = formula
                    _LOGGER.debug(f"Discovered country formula: {formula.name} ({formula.id})")
                else:
                    _LOGGER.warning(
                        f"Module {module_info.name} has FORMULA but it's not a CountryFormula"
                    )
        except Exception as e:
            _LOGGER.error(f"Error loading formula module {module_info.name}: {e}")

    _discovered = True
    _LOGGER.info(f"Discovered {len(_registry)} country formulas: {list(_registry.keys())}")
    return _registry


def clear_registry() -> None:
    """Clear the formula registry cache for reload support."""
    global _registry, _discovered
    _registry = {}
    _discovered = False


def get_formula(country_id: str) -> Optional[CountryFormula]:
    """Get a formula by country ID.

    Args:
        country_id: The internal ID of the country (e.g., "netherlands")

    Returns:
        CountryFormula instance or None if not found.
    """
    formulas = discover_formulas()
    return formulas.get(country_id)


def get_all_formulas() -> Dict[str, CountryFormula]:
    """Get all discovered formulas.

    Returns:
        Dictionary mapping country ID to CountryFormula instance.
    """
    return discover_formulas()


def get_country_options() -> List[Tuple[str, str]]:
    """Get list of country options for UI selectors.

    Returns:
        List of tuples: [(id, display_name), ...]
    """
    formulas = discover_formulas()
    return [(f.id, f.name) for f in sorted(formulas.values(), key=lambda x: x.name)]


def get_all_param_keys() -> Set[str]:
    """Get all unique parameter keys across all formulas.

    This is useful for creating entities for all possible parameters.

    Returns:
        Set of all parameter keys.
    """
    formulas = discover_formulas()
    all_keys: Set[str] = set()
    for formula in formulas.values():
        all_keys.update(formula.get_param_keys())
    return all_keys


def get_all_params() -> Dict[str, FormulaParam]:
    """Get all unique parameters across all formulas.

    If the same key appears in multiple formulas, the first one found is used.

    Returns:
        Dictionary mapping param key to FormulaParam instance.
    """
    formulas = discover_formulas()
    all_params: Dict[str, FormulaParam] = {}
    for formula in formulas.values():
        for param in formula.params:
            if param.key not in all_params:
                all_params[param.key] = param
    return all_params


def is_param_active(country_id: str, param_key: str) -> bool:
    """Check if a parameter is active for the given country.

    Args:
        country_id: The country ID
        param_key: The parameter key

    Returns:
        True if the parameter is used by this country's formula.
    """
    formula = get_formula(country_id)
    if formula:
        return param_key in formula.get_param_keys()
    return False


# Re-export base classes for convenience
__all__ = [
    "CountryFormula",
    "FormulaParam",
    "ParamType",
    "discover_formulas",
    "clear_registry",
    "get_formula",
    "get_all_formulas",
    "get_country_options",
    "get_all_param_keys",
    "get_all_params",
    "is_param_active",
]
