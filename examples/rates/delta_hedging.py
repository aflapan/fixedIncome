"""
This script is an exploration in the effectiveness of delta hedging.
"""
import math
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
import multiprocessing
import itertools

from fixedIncome.src.assets.base_cashflow import CashflowKeys
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments import UsTreasuryBond, ONE_BASIS_POINT
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator, DayCountConvention
from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.scheduling_tools.schedule_enumerations import PaymentFrequency
from fixedIncome.src.portfolio.base_portfolio import Portfolio, PortfolioEntry
from fixedIncome.src.risk.risk_metrics import Risk, RiskLadder
from fixedIncome.src.risk.key_rate import KeyRate, KeyRateCollection
from fixedIncome.src.curves.yield_curves.yield_curve import YieldCurve, YieldCurveFactory
from fixedIncome.src.curves.curve_enumerations import InterpolationMethod

#------------------------------------------------------
# load historical yield curve data

data = pd.read_csv('../../fixedIncome/data/rates/feds200628.csv', skiprows=9)
data.set_index('Date', inplace=True)

zero_coupon_yields = [
        'SVENY01', 'SVENY02', 'SVENY03', 'SVENY04', 'SVENY05',
        'SVENY06', 'SVENY07', 'SVENY08', 'SVENY09', 'SVENY10',
        'SVENY11', 'SVENY12', 'SVENY13', 'SVENY14', 'SVENY15',
        'SVENY16', 'SVENY17', 'SVENY18', 'SVENY19', 'SVENY20',
        'SVENY21', 'SVENY22', 'SVENY23', 'SVENY24', 'SVENY25',
        'SVENY26', 'SVENY27', 'SVENY28', 'SVENY29', 'SVENY30'
]


def yields_to_bond_instruments(zc_yields: pd.Series, start_date: date) -> list[UsTreasuryBond]:
    """
    Takes a row of zero-coupon bond yields and produces a set of
    US Treasury bonds.
    """
    prefix = 'SVENY'
    maturity_years = [int(col.removeprefix(prefix)) for col in zero_coupon_yields]
    maturities = [start_date + relativedelta(years=year) for year in maturity_years]
    accruals = [DayCountCalculator.compute_accrual_length(start_date, maturity,DayCountConvention.ACTUAL_OVER_ACTUAL)
                    for maturity in maturities]  # might need to re-work this based on adjusted maturity dates

    zc_prices = [math.exp(-zc_yield / 100 * accrual) * 100 for zc_yield, accrual in zip(zc_yields, accruals)]
    zc_bonds = []

    for maturity, price, year in zip(maturities, zc_prices, maturity_years):

        bond = UsTreasuryBond(price=price,
                                  coupon_rate=0.00,
                                  principal=100,
                                  tenor=f'{year}Y',
                                  payment_frequency=PaymentFrequency.ZERO_COUPON,
                                  purchase_date=start_date,
                                  maturity_date=maturity)
        zc_bonds.append(bond)

    return zc_bonds



#------------------------------------------------------
# Setup

curve_factory = YieldCurveFactory()

start_date = date(1993, 9, 3)

three_yr = UsTreasuryBond(price=95 + 3 / 32,
                            coupon_rate=6.25,
                            principal=100,
                            tenor='3Y',
                            purchase_date=date(1993, 8, 15),
                            maturity_date=date(1996, 8, 15))

seven_yr = UsTreasuryBond(price=81 + 3 / 32,
                            coupon_rate=4.25,
                            principal=100,
                            tenor='7Y',
                            purchase_date=date(1993, 8, 15),
                            maturity_date=date(2000, 8, 15))

ten_yr = UsTreasuryBond(price=86 + 8 / 32,
                            coupon_rate=4.5,
                            principal=100,
                            tenor='10Y',
                            purchase_date=date(1993, 8, 15),
                            maturity_date=date(2003, 8, 15))


zc_bond_list = yields_to_bond_instruments(data[zero_coupon_yields].loc['1993-09-03'], start_date)
yield_curve = curve_factory.bootstrap_yield_curve(zc_bond_list,
                                                  interpolation_method=InterpolationMethod.LINEAR,
                                                  reference_date=start_date)

zc_portfolio = Portfolio([PortfolioEntry(asset=bond, quantity=1) for bond in zc_bond_list])
treasury_kr_collection = zc_portfolio.to_key_rate_collection(DayCountConvention.ACTUAL_OVER_ACTUAL)


bond_portfolio = Portfolio(
    (PortfolioEntry(2.0, ten_yr),
     PortfolioEntry(5.0, three_yr),
     PortfolioEntry(-3.0, seven_yr)
     )
)

# ----------------------------------------------------------------

risk_ladder = yield_curve.calculate_pv01_risk_ladder(bond_portfolio, treasury_kr_collection)
portfolio_rate_exposure = np.array([risk.pv01 for risk in risk_ladder])

hedge_risk_ladders = [yield_curve.calculate_pv01_risk_ladder(bond, treasury_kr_collection) for bond in zc_bond_list]
hedge_portfolio_risk = np.array([[risk.pv01 for risk in risk_ladder] for risk_ladder in hedge_risk_ladders]).T  # should be a diagonal matrix of risks
                                                                                                                    # because we are computing risks on a series
                                                                                                                    # of zero-coupon bonds. So longer-dated bonds
                                                                                                                    # have no short-term exposure.
hedge_amounts = np.linalg.solve(a=hedge_portfolio_risk, b=portfolio_rate_exposure)

hedging_portfolio = Portfolio([PortfolioEntry(asset=zc_bond, quantity=hedge_amount)
                               for zc_bond, hedge_amount in zip(zc_bond_list, hedge_amounts)])


#--------------------------------------------------------------------------------
#

def compute_protoflio_pvs(new_date, data) -> tuple[float | None, float | None ]:
    try:
        new_zc_bond_list = yields_to_bond_instruments(data[zero_coupon_yields].loc[str(new_date)], new_date)
    except KeyError:
        return None, None

    new_yield_curve = curve_factory.bootstrap_yield_curve(new_zc_bond_list,
                                                          interpolation_method=InterpolationMethod.LINEAR,
                                                          reference_date=new_date)

    # collect portfolio cash position
    portfolio_cash_position = 0.0
    for portfolio_entry in bond_portfolio:
        for cashflow in portfolio_entry.asset.cashflow_list:
            for payment in cashflow:
                if payment.payment_date <= new_date:
                    portfolio_cash_position += portfolio_entry.quantity * payment.payment

    bond_portfolio.cash_amount = portfolio_cash_position

    # collect hedging portfolio cash position
    hedging_cash_position = 0.0
    for portfolio_entry in hedging_portfolio:
        for cashflow in portfolio_entry.asset.cashflow_list:
            for payment in cashflow:
                if payment.payment_date <= new_date:
                    hedging_cash_position += portfolio_entry.quantity * payment.payment

    hedging_portfolio.cash_amount = hedging_cash_position

    portfolio_pv = new_yield_curve.present_value(bond_portfolio)
    hedge_pv = new_yield_curve.present_value(hedging_portfolio)
    return portfolio_pv, hedge_pv


frzn_pv = functools.partial(compute_protoflio_pvs, data=data)
date_range = Scheduler.generate_dates_by_increments(start_date=date(1993, 9, 3),
                                                    end_date=date(2000, 9, 3),
                                                    increment=timedelta(1))

if __name__ == '__main__':

    NUM_CORES = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=int(NUM_CORES/2)) as mp_pool:
        present_values = mp_pool.map(frzn_pv, date_range)

    plt.figure(figsize=(12, 5))
    plt.title('Bond Portfolio and Static Key Rate Hedging Portfolio Present Values Through Time')
    plt.plot(date_range, [bond_pv for bond_pv, hedge_pv in present_values])
    plt.plot(date_range, [hedge_pv for bond_pv, hedge_pv in present_values], color='mediumaquamarine')
    plt.legend(['Bond Portfolio', 'Hedge Portfolio'], frameon=False)
    plt.xlabel('Date')
    plt.ylabel('Present Value ($)')
    plt.grid(alpha=0.25)
    plt.legend(['Bond Portfolio', 'Hedge Portfolio'], frameon=False)
    plt.savefig('../../docs/images/rates/portfolio_and_hedge_present_value_across_time_with_cash.png')
    plt.show()