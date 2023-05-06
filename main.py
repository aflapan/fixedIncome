from datetime import date

from fixedIncome.src.curves.yield_curve import *
from fixedIncome.src.curves.key_rate import *
from fixedIncome.src.web_scraper.web_scraper import *

def main(bond_collection, curve_factory) -> None:

    #---------------------------------------------------------------------
    # Yield Curve

    yield_curve = curve_factory.construct_yield_curve(bond_collection,
                                                      interpolation_method='cubic',
                                                      reference_date=purchase_date)
    # Trial Key Rate to test bumping Yield Curve

    four_wk_kr = KeyRate(day_count_convention='act/act',
                         key_rate_date=date(2023, 3, 28),
                         prior_date=None,
                         next_date=date(2024, 2, 22))

    one_yr_kr = KeyRate(day_count_convention='act/act',
                        key_rate_date=date(2024, 2, 22),
                        prior_date=date(2023, 3, 28),
                        next_date=date(2025, 2, 28))

    two_yr_kr = KeyRate(day_count_convention='act/act',
                        key_rate_date=date(2025, 2, 28),
                        prior_date=date(2024, 2, 22),
                        next_date=date(2026, 2, 15))

    three_year_kr = KeyRate(day_count_convention='act/act',
                            key_rate_date=date(2026, 2, 15),
                            prior_date=date(2025, 2, 28),
                            next_date=date(2030, 2, 28))

    seven_yr_kr = KeyRate(day_count_convention='act/act',
                          key_rate_date=date(2030, 2, 28),
                          prior_date=date(2026, 2, 15),
                          next_date=date(2033, 2, 15))

    ten_yr_kr = KeyRate(day_count_convention='act/act',
                        key_rate_date=date(2033, 2, 15),
                        prior_date=date(2030, 2, 28),
                        next_date=date(2043, 2, 15))

    twenty_yr_kr = KeyRate(day_count_convention='act/act',
                           key_rate_date=date(2043, 2, 15),
                           prior_date=date(2033, 2, 15),
                           next_date=date(2053, 2, 15))

    thirty_yr_kr = KeyRate(day_count_convention='act/act',
                           key_rate_date=date(2053, 2, 15),
                           prior_date=date(2043, 2, 15),
                           next_date=None)

    key_rate_list = [four_wk_kr, one_yr_kr, two_yr_kr, three_year_kr, seven_yr_kr, ten_yr_kr, twenty_yr_kr,
                     thirty_yr_kr]
    kr_collection = KeyRateCollection(key_rate_list)

    kr_collection._set_dates_in_collection()

    yield_curve.plot(adjustment=kr_collection)

    yield_curve.plot_price_curve(thirty_yr)

    # DV01 calculations

    durations = [yield_curve.duration(bond) for bond in bond_collection]

    print("Duration values are...")
    print(durations)

    convexities = [yield_curve.convexity(bond) for bond in bond_collection]
    print("Convexity values are...")
    print(convexities)

    #----------------------------------------------------------------
    # Key Rate DV01s

    dv01s = yield_curve.calculate_dv01s(ten_yr, kr_collection)
    print("Key rate DV01s...")
    print(dv01s)

    print(format(kr_collection))






if __name__ == '__main__':
    curve_factory_obj = YieldCurveFactory()

    # Construct Bond Objects from U.S. Treasury Bonds found on
    # https://www.treasurydirect.gov/auctions/announcements-data-results/

    purchase_date = date(2023, 2, 27)

    # Four Week
    four_wk = Bond(price=99.648833,
                   coupon=0.00,
                   principal=100,
                   tenor='1M',
                   payment_frequency='zero-coupon',
                   purchase_date=purchase_date,
                   maturity_date=date(2023, 3, 28))

    one_yr = Bond(price=95.151722,
                  coupon=0.00,
                  principal=100,
                  tenor='1Y',
                  payment_frequency='zero-coupon',
                  purchase_date=purchase_date,
                  maturity_date=date(2024, 2, 22))

    # Two Year
    two_yr = Bond(price=99.909356,
                  coupon=4.625,
                  principal=100,
                  tenor='2Y',
                  purchase_date=purchase_date,
                  maturity_date=date(2025, 2, 28))

    # Three Year
    three_yr = Bond(price=99.795799,
                    coupon=4.0000,
                    principal=100,
                    tenor='3Y',
                    purchase_date=purchase_date,
                    maturity_date=date(2026, 2, 15))

    # Five Year
    five_yr = Bond(price=99.511842,
                   coupon=4.000,
                   principal=100,
                   tenor='5Y',
                   purchase_date=purchase_date,
                   maturity_date=date(2028, 2, 28))

    # Seven Year
    seven_yr = Bond(price=99.625524,
                    coupon=4.000,
                    principal=100,
                    tenor='7Y',
                    purchase_date=purchase_date,
                    maturity_date=date(2030, 2, 28))

    # Ten Year
    ten_yr = Bond(price=99.058658,
                  coupon=3.5000,
                  principal=100,
                  tenor='10Y',
                  purchase_date=purchase_date,
                  maturity_date=date(2033, 2, 15))

    # Twenty Year
    twenty_yr = Bond(price=98.601167,
                     coupon=3.875,
                     principal=100,
                     tenor='20Y',
                     purchase_date=purchase_date,
                     maturity_date=date(2043, 2, 15))

    # Thirty Year
    thirty_yr = Bond(price=98.898317,
                     coupon=3.625,
                     principal=100,
                     tenor='30Y',
                     purchase_date=purchase_date,
                     maturity_date=date(2053, 2, 15))

    bond_list = [
        four_wk, one_yr, two_yr, three_yr, five_yr, seven_yr, ten_yr, twenty_yr, thirty_yr
    ]

    main(bond_list, curve_factory_obj)
