"""
Unit tests for fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.trebly_mean_reverting_vasicek.py
"""

from datetime import datetime, timedelta
import numpy as np

from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.vasicek_model import VasicekModel
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.doubly_mean_reverting_vasicek import DoublyMeanRevertingVasicek
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.trebly_mean_reverting_vasicek import TreblyMeanRevertingVasicek
from fixedIncome.src.scheduling_tools.scheduler import Scheduler


short_rate_reversion = 0.5
medium_rate_reversion = 0.25
long_reversion = 0.1

short_rate_vol = 100 / 10_000
medium_rate_vol = 75 / 10_000
long_vol = 50 / 10_000
short_long_corr = 0.5
short_medium_corr = 0.5
medium_long_corr = 0.5

long_term_mean = 0.05

start_time = datetime(2024, 1, 1, 0)
end_time = datetime(2053, 12, 31, 23, 59)


starting_state_space_vals = np.array([0.075, 0.05, 0.025])


dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                               end_date=end_time,
                                               increment=timedelta(1),
                                               max_dates=1_000_000)

tmr_vm = TreblyMeanRevertingVasicek(
    short_rate_reversion_speed=short_rate_reversion,
    short_rate_volatility=short_rate_vol,
    medium_rate_reversion_speed=medium_rate_reversion,
    medium_rate_volatility=medium_rate_vol,
    long_term_mean=long_term_mean,
    long_term_reversion_speed=long_reversion,
    long_term_volatility=long_vol,
    short_rate_medium_rate_correlation=short_medium_corr,
    short_rate_long_term_correlation=short_long_corr,
    medium_rate_long_term_correlation_correlation=medium_long_corr,
    start_datetime=start_time,
    end_datetime=end_time
)

def test_long_term_process_is_single_variable_vasicek() -> None:
    """
    Tests that the sample path for the long term rate matches exactly what would be given in a single-variable Vasicek
    model with the same parameters.
    """
    PASS_THRESH = 1E-10

    tmr_vm.generate_path(starting_state_space_values=starting_state_space_vals,
                         set_path=True,
                         seed=1)

    test_bm = BrownianMotion(start_date_time=start_time, end_date_time=end_time)

    vm = VasicekModel(
        reversion_level=long_term_mean,
        reversion_speed=long_reversion,
        volatility=long_vol,
        brownian_motion=test_bm
    )

    vm.generate_path(starting_state_space_values=starting_state_space_vals[0], seed=1, set_path=True)

    assert all(abs(vm(date_obj) - tmr_vm.state_variables_diffusion_process(date_obj)[0]) < PASS_THRESH
               for date_obj in dates)


def test_medium_term_process_is_doubly_mean_reverting_vasicek() -> None:
    """
    Tests that the sample path for the medium term rate matches exactly what would be given in a doubly-mean reverting
    Vasicek model with the same parameters.
    """
    PASS_THRESH = 1E-10

    tmr_vm.generate_path(starting_state_space_values=starting_state_space_vals,
                         set_path=True,
                         seed=1)


    dmr_vm = DoublyMeanRevertingVasicek(
        short_rate_reversion_speed=medium_rate_reversion,
        short_rate_volatility=medium_rate_vol,
        long_term_mean=long_term_mean,
        long_term_reversion_speed=long_reversion,
        long_term_volatility=long_vol,
        short_rate_long_term_correlation=medium_long_corr,
        start_datetime=start_time,
        end_datetime=end_time
    )

    dmr_vm.generate_path(starting_state_space_values=starting_state_space_vals[:2], seed=1, set_path=True)

    assert all(abs(dmr_vm.state_variables_diffusion_process(date_obj)[1] -
                   tmr_vm.state_variables_diffusion_process(date_obj)[1]) < PASS_THRESH
               for date_obj in dates)
