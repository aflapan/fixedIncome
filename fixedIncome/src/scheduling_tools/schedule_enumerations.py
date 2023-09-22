from enum import Enum

class PaymentFrequency(Enum):
    ZERO_COUPON = 0
    QUARTERLY = 1
    SEMI_ANNUAL = 2
    ANNUAL = 3


class BusinessDayAdjustment(Enum):
    FOLLOWING = 0
    MODIFIED_FOLLOWING = 1

class DayCountConvention(Enum):
    ACTUAL_OVER_360 = 0
    ACTUAL_OVER_365 = 1
    ACTUAL_OVER_365_POINT_25 = 2
    ACTUAL_OVER_ACTUAL = 3
    THIRTY_OVER_THREESIXTY = 4

