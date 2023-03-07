import datetime
from fixedIncome.curves.yield_curve import *
from fixedIncome.curves.key_rate import *
from fixedIncome.utils.day_count_calculator import *



curve_factory_obj = YieldCurveFactory()

# Construct Bond Objects from U.S. Treasury Bonds


purchase_date = datetime.date(2023, 2, 27)

# Four Week
four_wk = Bond(price=99.648833,
                coupon=0.00,
                principal=100,
                tenor='1M',
                payment_frequency='zero-coupon',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2023, 3, 28))


one_yr = Bond(price=95.151722,
                coupon=0.00,
                principal=100,
                tenor='1Y',
                payment_frequency='zero-coupon',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2024, 2, 22))

# Two Year
two_yr = Bond(price=99.909356,
                coupon=4.625,
                principal=100,
                tenor='2Y',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2025, 2, 28))

# Three Year
three_yr = Bond(price=99.795799,
                  coupon=4.0000,
                  principal=100,
                  tenor='3Y',
                  purchase_date=purchase_date,
                  maturity_date=datetime.date(2026, 2, 15))


# Five Year
five_yr = Bond(price=99.511842,
                coupon=4.000,
                principal=100,
                tenor='5Y',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2028, 2, 28))


# Seven Year
seven_yr = Bond(price=99.625524,
                coupon=4.000,
                principal=100,
                tenor='7Y',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2030, 2, 28))

# Ten Year
ten_yr = Bond(price=99.058658,
                coupon=3.5000,
                principal=100,
                tenor='10Y',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2033, 2, 15))

# Twenty Year
twenty_yr = Bond(price=98.601167,
                coupon=3.875,
                principal=100,
                tenor='20Y',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2043, 2, 15))


# Thirty Year
thirty_yr = Bond(price=98.898317,
                   coupon=3.625,
                   principal=100,
                   tenor='30Y',
                   purchase_date=purchase_date,
                   maturity_date=datetime.date(2053, 2, 15))



bond_collection = [
    four_wk, one_yr, two_yr, three_yr, five_yr, seven_yr, ten_yr, twenty_yr, thirty_yr
]

#---------------------------------------------------------------------
# Yield Curve

yield_curve = curve_factory_obj.construct_yield_curve(bond_collection,
                                                      interpolation_method='cubic',
                                                      reference_date=purchase_date)

key_rate_obj = KeyRate('act/act',
                       key_rate_date=datetime.date(2030, 2, 28),
                       prior_key_rate_date=datetime.date(2029, 2, 28),
                       next_key_rate_date=datetime.date(2051, 5, 15))

key_rate_obj.set_adjustment_level(0.05)

yield_curve.plot(adjustment=key_rate_obj)

yield_curve.plot_price_curve(thirty_yr)

#------------------------------------------------------------------------
# compute present values

present_values = [yield_curve.calculate_present_value(bond) for bond in bond_collection]
full_bond_prices = [bond.full_price for bond in bond_collection]

print([abs(pv - full_price) < 1e-5 for (pv, full_price) in zip(present_values, full_bond_prices)])
