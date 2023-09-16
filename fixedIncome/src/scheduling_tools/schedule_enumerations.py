from enum import Enum

class PaymentFrequency(Enum):
    ZERO_COUPON = 0
    QUARTERLY = 1
    SEMI_ANNUAL = 2
    ANNUAL = 3


class BusinessDayAdjustment(Enum):
    FOLLOWING = 0
    MODIFIED_FOLLOWING = 1

