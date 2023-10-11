from datetime import date

from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention, PaymentFrequency
from fixedIncome.src.curves.curve_enumerations import InterpolationMethod
from fixedIncome.src.curves.key_rate import KeyRate, KeyRateCollection
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments import (ONE_BASIS_POINT,
                                                                                    UsTreasuryBond)
from fixedIncome.src.curves.yield_curves.yield_curve import YieldCurve, YieldCurveFactory



def main(bond_collection, curve_factory) -> None:

    #---------------------------------------------------------------------
    # Yield Curve

    yield_curve = curve_factory.construct_yield_curve(bond_collection,
                                                      interpolation_method=InterpolationMethod.LINEAR,
                                                      reference_date=purchase_date)
    # Trial Key Rate to test bumping Yield Curve
    four_wk_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                         key_rate_date=date(2023, 3, 28),
                         prior_date=None,
                         next_date=date(2024, 2, 22))

    one_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                        key_rate_date=date(2024, 2, 22),
                        prior_date=date(2023, 3, 28),
                        next_date=date(2025, 2, 28))

    two_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                        key_rate_date=date(2025, 2, 28),
                        prior_date=date(2024, 2, 22),
                        next_date=date(2026, 2, 15))

    three_year_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                            key_rate_date=date(2026, 2, 15),
                            prior_date=date(2025, 2, 28),
                            next_date=date(2030, 2, 28))

    seven_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                          key_rate_date=date(2030, 2, 28),
                          prior_date=date(2026, 2, 15),
                          next_date=date(2033, 2, 15))

    ten_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                        key_rate_date=date(2033, 2, 15),
                        prior_date=date(2030, 2, 28),
                        next_date=date(2043, 2, 15))

    twenty_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                           key_rate_date=date(2043, 2, 15),
                           prior_date=date(2033, 2, 15),
                           next_date=date(2053, 2, 15))

    thirty_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                           key_rate_date=date(2053, 2, 15),
                           prior_date=date(2043, 2, 15),
                           next_date=None)

    key_rate_list = [four_wk_kr, one_yr_kr, two_yr_kr, three_year_kr, seven_yr_kr, ten_yr_kr, twenty_yr_kr,
                     thirty_yr_kr]
    #kr_collection = KeyRateCollection(key_rate_list)
    #kr_collection._set_dates_in_collection()

    yield_curve.plot(adjustment=two_yr_kr)
    yield_curve.present_value(two_yr)  # schedule isn't exactly correct


    #yield_curve.plot_price_curve(thirty_yr)

    # DV01 and convexity calculations

    #durations = [yield_curve.duration(bond) for bond in bond_collection]
    #print("Duration values are...")
    #print(durations)

    #convexities = [yield_curve.convexity(bond) for bond in bond_collection]
    #print("Convexity values are...")
    #print(convexities)

    #----------------------------------------------------------------
    # Key Rate DV01s

    #dv01s = yield_curve.calculate_dv01s(ten_yr, kr_collection)
    #print("Key rate DV01s...")
    #print(dv01s)
    #print(format(kr_collection))




if __name__ == '__main__':
    curve_factory_obj = YieldCurveFactory()

    # Construct Bond Objects from U.S. Treasury Bonds found on
    # https://www.treasurydirect.gov/auctions/announcements-data-results/

    purchase_date = date(2023, 2, 27)

    # Four Week
    four_wk = UsTreasuryBond(price=99.648833,
                             coupon=0.00,
                             principal=100,
                             tenor='1M',
                             payment_frequency=PaymentFrequency.ZERO_COUPON,
                             purchase_date=purchase_date,
                             maturity_date=date(2023, 3, 28))

    three_month = UsTreasuryBond(price=98.63,
                   coupon=0.00,
                   principal=100,
                   tenor='3M',
                   payment_frequency=PaymentFrequency.ZERO_COUPON,
                   purchase_date=purchase_date,
                   maturity_date=date(2023, 5, 28))

    six_month = UsTreasuryBond(price=97.20,
                   coupon=0.00,
                   principal=100,
                   tenor='6M',
                   payment_frequency=PaymentFrequency.ZERO_COUPON,
                   purchase_date=purchase_date,
                   maturity_date=date(2023, 8, 28))

    one_yr = UsTreasuryBond(price=94.724,
                  coupon=0.00,
                  principal=100,
                  tenor='1Y',
                  payment_frequency=PaymentFrequency.ZERO_COUPON,
                  purchase_date=purchase_date,
                  maturity_date=date(2024, 2, 22))

    # Two Year
    two_yr = UsTreasuryBond(price=99 + 9/32,
                  coupon=5.00,
                  principal=100,
                  tenor='2Y',
                  purchase_date=purchase_date,
                  maturity_date=date(2025, 2, 28))

    # Three Year
    three_yr = UsTreasuryBond(price=99 + 3/32,
                    coupon=4.625,
                    principal=100,
                    tenor='3Y',
                    purchase_date=purchase_date,
                    maturity_date=date(2026, 2, 15))

    # Five Year
    five_yr = UsTreasuryBond(price=99 + 4/32,
                   coupon=4.625,
                   principal=100,
                   tenor='5Y',
                   purchase_date=purchase_date,
                   maturity_date=date(2028, 2, 28))

    # Seven Year
    seven_yr = UsTreasuryBond(price=98 + 9/32,
                    coupon=4.625,
                    principal=100,
                    tenor='7Y',
                    purchase_date=purchase_date,
                    maturity_date=date(2030, 2, 28))

    # Ten Year
    ten_yr = UsTreasuryBond(price=92 + 8/32,
                  coupon=3.875,
                  principal=100,
                  tenor='10Y',
                  purchase_date=purchase_date,
                  maturity_date=date(2033, 2, 15))

    # Twenty Year
    twenty_yr = UsTreasuryBond(price=90.052,
                     coupon=4.375,
                     principal=100,
                     tenor='20Y',
                     purchase_date=purchase_date,
                     maturity_date=date(2043, 2, 15))

    # Thirty Year
    thirty_yr = UsTreasuryBond(price=86 + 9/32,
                     coupon=4.125,
                     principal=100,
                     tenor='30Y',
                     purchase_date=purchase_date,
                     maturity_date=date(2053, 2, 15))

    bond_list = [
        four_wk,
        three_month,
        six_month,
        one_yr,
        two_yr,
        three_yr,
        five_yr,
        seven_yr,
        ten_yr,
        twenty_yr,
        thirty_yr
    ]

    main(bond_list, curve_factory_obj)
