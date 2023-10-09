"""
This script contains a basic set of enumerations for curve parameters
and configurations and the class for Knot-Value pairs used by curves to interpolate.
"""


from enum import Enum
from typing import NamedTuple
from datetime import date

class InterpolationMethod(Enum):
    PREVIOUS = 'previous'
    LINEAR = 'linear'
    QUADRATIC_SPLINE = 'quadratic'
    CUBIC_SPLINE = 'cubic'

class InterpolationSpace(Enum):
    DISCOUNT_FACTOR = 0
    FORWARD_RATES = 1
    YIELD = 2
    CONTINUOUSLY_COMPOUNDED_YIELD = 3
    YIELD_TO_MATURITY = 4
    CONTINUOUSLY_COMPOUNDED_YIELD_TO_MATURITY = 5

class CurveIndex(Enum):
    NONE = -1
    US_TREASURY = 0
    SOFR = 1
    TERM_SOFR_1M = 2
    TERM_SOFR_3M = 3
    TERM_SOFR_6M = 4
    TERM_SOFR_12M = 5
    FED_FUND = 6
    LIBOR_3M = 7

class EndBehavior(Enum):
    ERROR = 0
    CONSTANT = 1


class KnotValuePair(NamedTuple):
    knot: date
    value: float
