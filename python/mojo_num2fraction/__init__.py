"""mojo-num2fraction: the num2fraction algorithm with its numeric core in Mojo.

Importable alongside the real thing; nothing here shadows a stdlib module.
"""

from . import core
from ._lib import lib
from .core import (
    Num2FractionError,
    convert,
    convert_array,
    fraction,
    gcd,
    js_remainder,
    js_round,
    num2fraction,
    num2fraction_batch,
    parse_float,
    precision,
    precision_array,
)

__version__ = "0.1.0"

__all__ = [
    "Num2FractionError",
    "convert",
    "convert_array",
    "fraction",
    "gcd",
    "js_remainder",
    "js_round",
    "core",
    "lib",
    "num2fraction",
    "num2fraction_batch",
    "parse_float",
    "precision",
    "precision_array",
]
