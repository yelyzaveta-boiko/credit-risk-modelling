"""Smoke tests to confirm the environment and core dependencies are installed correctly."""

from importlib import import_module

import pytest

CORE_DEPENDENCIES = ("numpy", "pandas", "sklearn", "xgboost", "shap")


@pytest.mark.parametrize("module_name", CORE_DEPENDENCIES)
def test_core_imports(module_name):
    """Fails fast if core dependencies are missing or broken."""
    try:
        module = import_module(module_name)
    except Exception as exc:
        hint = ""
        if module_name == "xgboost" and "libomp" in str(exc):
            hint = " On macOS, install the OpenMP runtime with `brew install libomp`."
        pytest.fail(f"{module_name} could not be imported.{hint}\n{exc}", pytrace=False)

    assert module.__version__


def test_placeholder():
    """Placeholder to confirm pytest itself is wired up"""
    assert 1 + 1 == 2
