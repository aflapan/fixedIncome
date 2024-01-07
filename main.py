from datetime import date
import scipy.optimize

from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention, PaymentFrequency
from fixedIncome.src.curves.curve_enumerations import InterpolationMethod
from fixedIncome.src.risk.key_rate import KeyRate, KeyRateCollection
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments import (UsTreasuryBond)
from fixedIncome.src.curves.yield_curves.yield_curve import YieldCurveFactory
from fixedIncome.src.portfolio.base_portfolio import Portfolio, PortfolioEntry


def main(bond_collection, curve_factory) -> None:
    #---------------------------------------------------------------------
    # Yield Curve

    yield_curve = curve_factory.bootstrap_yield_curve(bond_collection,
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

    portfolio = Portfolio([PortfolioEntry(asset=bond, quantity=1) for bond in bond_list])
    key_rate_list = [four_wk_kr, one_yr_kr, two_yr_kr, three_year_kr, seven_yr_kr, ten_yr_kr, twenty_yr_kr,
                     thirty_yr_kr]

    kr_collection = portfolio.to_key_rate_collection(DayCountConvention.ACTUAL_OVER_ACTUAL)
    #yield_curve.plot(adjustment=kr_collection)


    bond_a_portfolio = PortfolioEntry(2.0, bond_a)
    bond_b_portfolio = PortfolioEntry(1.0, bond_b)
    long_short_bond_portfolio = Portfolio((bond_a_portfolio, bond_b_portfolio))

    #----------------------------------------------------------------
    # Key Rate Derivatives
    risk_ladder = yield_curve.calculate_pv01_risk_ladder(long_short_bond_portfolio, kr_collection)
    print("Risk ladder is...")
    print(risk_ladder)



if __name__ == '__main__':

    curve_factory_obj = YieldCurveFactory()
    purchase_date = date(2023, 2, 27)

    # Construct Bond Objects from U.S. Treasury Bonds found on
    # https://www.treasurydirect.gov/auctions/announcements-data-results/

    purchase_date = date(2023, 2, 27)

    # Four Week
    four_wk = UsTreasuryBond(price=99.648833,
                             coupon_rate=0.00,
                             principal=100,
                             tenor='1M',
                             payment_frequency=PaymentFrequency.ZERO_COUPON,
                             purchase_date=purchase_date,
                             maturity_date=date(2023, 3, 28))

    three_month = UsTreasuryBond(price=98.63,
                   coupon_rate=0.00,
                   principal=100,
                   tenor='3M',
                   payment_frequency=PaymentFrequency.ZERO_COUPON,
                   purchase_date=purchase_date,
                   maturity_date=date(2023, 5, 28))

    six_month = UsTreasuryBond(price=97+1/32,
                   coupon_rate=0.00,
                   principal=100,
                   tenor='6M',
                   payment_frequency=PaymentFrequency.ZERO_COUPON,
                   purchase_date=purchase_date,
                   maturity_date=date(2023, 8, 28))

    one_yr = UsTreasuryBond(price=94,
                  coupon_rate=0.00,
                  principal=100,
                  tenor='1Y',
                  payment_frequency=PaymentFrequency.ZERO_COUPON,
                  purchase_date=purchase_date,
                  maturity_date=date(2024, 2, 22))

    # Two Year
    two_yr = UsTreasuryBond(price=96+27/32,
                  coupon_rate=5.00,
                  principal=100,
                  tenor='2Y',
                  purchase_date=purchase_date,
                  maturity_date=date(2025, 2, 28))

    # Three Year
    three_yr = UsTreasuryBond(price=94+15/32,
                    coupon_rate=4.625,
                    principal=100,
                    tenor='3Y',
                    purchase_date=purchase_date,
                    maturity_date=date(2026, 2, 15))

    # Five Year
    five_yr = UsTreasuryBond(price=91 + 17/32,
                   coupon_rate=4.625,
                   principal=100,
                   tenor='5Y',
                   purchase_date=purchase_date,
                   maturity_date=date(2028, 2, 28))

    # Seven Year
    seven_yr = UsTreasuryBond(price=89+9/32,
                    coupon_rate=4.625,
                    principal=100,
                    tenor='7Y',
                    purchase_date=purchase_date,
                    maturity_date=date(2030, 2, 28))

    # Ten Year
    ten_yr = UsTreasuryBond(price=86+8/32,
                  coupon_rate=4.5,
                  principal=100,
                  tenor='10Y',
                  purchase_date=purchase_date,
                  maturity_date=date(2033, 2, 15))

    # Twenty Year
    twenty_yr = UsTreasuryBond(price=87+17/32,
                     coupon_rate=5.275,
                     principal=100,
                     tenor='20Y',
                     purchase_date=purchase_date,
                     maturity_date=date(2043, 2, 15))

    # Thirty Year
    thirty_yr = UsTreasuryBond(price=83 + 9/32,
                     coupon_rate=5.125,
                     principal=100,
                     tenor='30Y',
                     purchase_date=purchase_date,
                     maturity_date=date(2053, 2, 15))


    bond_a = UsTreasuryBond(price=96.73164773103913,
                                coupon_rate=5,
                                principal=100_000,
                                tenor='4Y',
                                purchase_date=purchase_date,
                                maturity_date=date(2027, 2, 15))

    bond_b = UsTreasuryBond(price=89.6613284260062,
                                coupon_rate=3,
                                principal=100_000,
                                tenor='20Y',
                                purchase_date=purchase_date,
                                maturity_date=date(2043, 2, 15))


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
