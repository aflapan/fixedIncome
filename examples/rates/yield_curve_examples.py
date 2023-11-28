import math
from datetime import date
from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention, PaymentFrequency
from fixedIncome.src.curves.curve_enumerations import InterpolationMethod
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments import UsTreasuryBond
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.curves.yield_curves.yield_curve import YieldCurve, YieldCurveFactory

FIXED_YIELD = 0.05  # yield is 5%
purchase_date = date(2023, 2, 27)

one_yr_price = 100 * math.exp(-FIXED_YIELD * DayCountCalculator.compute_accrual_length(purchase_date,
                                                                                       date(2024, 2, 28),
                                                                                       DayCountConvention.ACTUAL_OVER_ACTUAL))
one_yr_zc = UsTreasuryBond(price=one_yr_price,
                           coupon_rate=0.00,
                           principal=100,
                           tenor='1Y',
                           payment_frequency=PaymentFrequency.ZERO_COUPON,
                           purchase_date=purchase_date,
                           maturity_date=date(2024, 2, 28))

two_yr_price = 100 * math.exp(-FIXED_YIELD * DayCountCalculator.compute_accrual_length(purchase_date,
                                                                                       date(2025, 2, 28),
                                                                                       DayCountConvention.ACTUAL_OVER_ACTUAL))
two_yr_zc = UsTreasuryBond(price=two_yr_price-1.0,
                           coupon_rate=0.00,
                           principal=100,
                           tenor='2Y',
                           payment_frequency=PaymentFrequency.ZERO_COUPON,
                           purchase_date=purchase_date,
                           maturity_date=date(2025, 2, 28))

three_yr_price = 100 * math.exp(-FIXED_YIELD * DayCountCalculator.compute_accrual_length(purchase_date,
                                                                                         date(2026, 3, 2),
                                                                                         DayCountConvention.ACTUAL_OVER_ACTUAL))
three_yr_zc = UsTreasuryBond(price=three_yr_price-2.0,
                             coupon_rate=0.00,
                             principal=100,
                             tenor='3Y',
                             payment_frequency=PaymentFrequency.ZERO_COUPON,
                             purchase_date=purchase_date,
                             maturity_date=date(2026, 2, 28))

four_yr_price = 100 * math.exp(-FIXED_YIELD * DayCountCalculator.compute_accrual_length(purchase_date,
                                                                                        date(2027, 3, 1),
                                                                                        DayCountConvention.ACTUAL_OVER_ACTUAL))
four_yr_zc = UsTreasuryBond(price=four_yr_price-3.0,
                            coupon_rate=0.00,
                            principal=100,
                            tenor='4Y',
                            payment_frequency=PaymentFrequency.ZERO_COUPON,
                            purchase_date=purchase_date,
                            maturity_date=date(2027, 2, 28))

five_yr_price = 100 * math.exp(-FIXED_YIELD * DayCountCalculator.compute_accrual_length(purchase_date,
                                                                                        date(2028, 2, 28),
                                                                                        DayCountConvention.ACTUAL_OVER_ACTUAL))
five_yr_zc = UsTreasuryBond(price=five_yr_price-5.0,
                            coupon_rate=0.00,
                            principal=100,
                            tenor='5Y',
                            payment_frequency=PaymentFrequency.ZERO_COUPON,
                            purchase_date=purchase_date,
                            maturity_date=date(2028, 2, 28))

test_zc_collection = [one_yr_zc, two_yr_zc, three_yr_zc, four_yr_zc, five_yr_zc]

curve_factory_obj = YieldCurveFactory()
yield_curve = curve_factory_obj.construct_yield_curve(test_zc_collection,
                                                      interpolation_method=InterpolationMethod.CUBIC_SPLINE,
                                                      reference_date=purchase_date)

yield_curve.plot()

bond_a = UsTreasuryBond(price=96.73164773103913,
                        coupon_rate=5,
                        principal=100,
                        tenor='4Y',
                        purchase_date=purchase_date,
                        maturity_date=date(2027, 2, 15))

bond_b = UsTreasuryBond(price=89.6613284260062,
                        coupon_rate=3,
                        principal=100,
                        tenor='4Y',
                        purchase_date=purchase_date,
                        maturity_date=date(2027, 2, 15))



print('Flat Yield Curve Environment ' + '-'*20)
print('Price of Bond A:', yield_curve.present_value(bond_a))
print('Price of Bond B:', yield_curve.present_value(bond_b))
print('\n')
print('Semi-annual compounding YTM for Bond A:', bond_a.yield_to_maturity())
print('Semi-annual compounding YTM for Bond b:', bond_b.yield_to_maturity())
