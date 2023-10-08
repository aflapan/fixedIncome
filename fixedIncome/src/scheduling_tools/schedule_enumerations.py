from enum import Enum


class PaymentFrequency(Enum):  # values indicate number of payments per year
    ZERO_COUPON = 0
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4


class BusinessDayAdjustment(Enum):
    FOLLOWING = 0
    MODIFIED_FOLLOWING = 1


class DayCountConvention(Enum):
    ACTUAL_OVER_360 = 0
    ACTUAL_OVER_365 = 1
    ACTUAL_OVER_365_POINT_25 = 2
    ACTUAL_OVER_ACTUAL = 3
    THIRTY_OVER_THREESIXTY = 4


class SettlementConvention(Enum):
    T_PLUS_ZERO_BUSINESS = 0
    T_PLUS_ONE_BUSINESS = 1
    T_PLUS_TWO_BUSINESS = 2
    T_PLUS_THREE_BUSINESS = 3
    T_PLUS_ZERO_CALENDAR = 4
    T_PLUS_ONE_CALENDAR = 5
    T_PLUS_TWO_CALENDAR = 6
    T_PLUS_THREE_CALENDAR = 7
