"""
Unit tests for fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.stochastic_market_price_of_risk.py
"""

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.vasicek_model import VasicekModel
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.stochastic_market_price_of_risk import StochasticMarketPriceOfRisk
from fixedIncome.src.scheduling_tools.scheduler import Scheduler

short_rate_reversion = 0.1
medium_reversion = 0.25
market_price_of_risk_reversion_speed = 0.5

short_rate_vol = 50 / 10_000  # 10 bps
medium_vol = 100 / 10_000
market_price_of_risk_vol = 500 / 10_000

short_rate_medium_rate_correlation = 0.5
short_rate_market_price_of_risk_correlation = 0.01
medium_rate_market_price_of_risk_correlation = 0.2

medium_rate_mean = 200 / 10_000
market_price_of_risk_mean = 100 / 10_000

start_time = datetime(2024, 1, 1, 0)
end_time = datetime(2053, 12, 31, 23, 59)

dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                               end_date=end_time,
                                               increment=timedelta(1),
                                               max_dates=100_000)

smpr_model = StochasticMarketPriceOfRisk(
    short_rate_reversion_speed=short_rate_reversion,
    short_rate_vol=short_rate_vol,
    medium_rate_reversion_speed=medium_reversion,
    medium_rate_vol=medium_vol,
    medium_rate_mean=medium_rate_mean,
    market_price_of_risk_reversion_speed=market_price_of_risk_reversion_speed,
    market_price_of_risk_vol=market_price_of_risk_vol,
    market_price_of_risk_mean=market_price_of_risk_mean,
    short_rate_medium_rate_correlation=short_rate_medium_rate_correlation,
    short_rate_market_price_of_risk_correlation=short_rate_market_price_of_risk_correlation,
    medium_rate_market_price_of_risk_correlation=medium_rate_market_price_of_risk_correlation,
    start_datetime=start_time,
    end_datetime=end_time,
    dt=relativedelta(hours=1)
)

starting_state_space_vals = np.array([0.05, 0.01, 0.03])


def test_long_term_process_is_single_variable_vasicek() -> None:
    """
    Tests that the sample path for the long term rate matches exactly what would be given in a single-variable Vasicek
    model with the same parameters.
    """
    PASS_THRESH = 1E-10

    smpr_model.generate_path(starting_state_space_values=starting_state_space_vals,
                             set_path=True,
                             seed=1)

    test_bm = BrownianMotion(start_date_time=start_time, end_date_time=end_time)

    vm = VasicekModel(
        reversion_level=market_price_of_risk_mean,
        reversion_speed=market_price_of_risk_reversion_speed,
        volatility=market_price_of_risk_vol,
        brownian_motion=test_bm
    )

    vm.generate_path(starting_state_space_values=starting_state_space_vals[0], seed=1, set_path=True)

    assert all(abs(vm(date_obj) - smpr_model.state_variables_diffusion_process(date_obj)[0]) < PASS_THRESH
               for date_obj in dates)
