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
    NONE = 'None'
    US_TREASURY = 'US TREASURY YIELD'
    SOFR = 'US SOFR'
    TERM_SOFR_1M = 'TERM SOFR 1M'
    TERM_SOFR_3M = 'TERM SOFR 3M'
    TERM_SOFR_6M = 'TERM SOFR 6M'
    TERM_SOFR_12M = 'TERM SOFR 12M'
    FED_FUND = 'FEDERAL FUNDS'
    LIBOR_3M = 'LIBOR 3M'

class EndBehavior(Enum):
    ERROR = 0
    CONSTANT = 1

