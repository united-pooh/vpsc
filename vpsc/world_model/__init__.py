"""Streaming language/world-model research components for E2/E3.

PyTorch, Gym, TextWorld, and HomeGrid are intentionally optional at package
import time.  Public PyTorch conveniences below are resolved lazily so the
task-adapter contracts remain importable in a lightweight environment.
"""

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "CausalLanguageModel": (".lm", "CausalLanguageModel"),
    "CausalTransformerCore": (".cores", "CausalTransformerCore"),
    "E2SignedCore": (".cores", "E2SignedCore"),
    "E3CumulativeScanCore": (".cores", "E3CumulativeScanCore"),
    "E3FixedPointScanCore": (".cores", "E3FixedPointScanCore"),
    "E3OscillatoryScanCore": (".cores", "E3OscillatoryScanCore"),
    "FairLMConfig": (".factory", "FairLMConfig"),
    "FrozenE2Config": (".factory", "FrozenE2Config"),
    "StatefulLSTMCore": (".cores", "StatefulLSTMCore"),
    "assert_parameter_budget": (".factory", "assert_parameter_budget"),
    "build_model_suite": (".factory", "build_model_suite"),
    "count_parameters": (".cores", "count_parameters"),
    "state_nbytes": (".cores", "state_nbytes"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "CausalLanguageModel",
    "CausalTransformerCore",
    "E2SignedCore",
    "E3CumulativeScanCore",
    "E3FixedPointScanCore",
    "E3OscillatoryScanCore",
    "FairLMConfig",
    "FrozenE2Config",
    "StatefulLSTMCore",
    "assert_parameter_budget",
    "build_model_suite",
    "count_parameters",
    "state_nbytes",
]
