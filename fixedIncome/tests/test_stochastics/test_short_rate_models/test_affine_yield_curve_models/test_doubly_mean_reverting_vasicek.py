"""
Unit tests for fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.doubly_mean_reverting_vasicek.py
"""
from datetime import date, datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.vasicek_model import VasicekModel
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.doubly_mean_reverting_vasicek import DoublyMeanRevertingVasicek
from fixedIncome.src.scheduling_tools.scheduler import Scheduler

short_rate_reversion = 0.5
long_reversion = 0.1

short_rate_vol = 100 / 10_000
long_vol = 50 / 10_000
short_long_corr = 0.0

long_term_mean = 0.05

start_time = datetime(2024, 1, 1, 0)
end_time = datetime(2053, 12, 31, 23, 59)

dmr_vm = DoublyMeanRevertingVasicek(
    short_rate_reversion_speed=short_rate_reversion,
    short_rate_volatility=short_rate_vol,
    long_term_mean=long_term_mean,
    long_term_reversion_speed=long_reversion,
    long_term_volatility=long_vol,
    short_rate_long_term_correlation=short_long_corr,
    start_datetime=start_time,
    end_datetime=end_time
)

starting_state_space_vals = np.array([0.075, 0.05])

dmr_vm.generate_path(starting_state_space_values=starting_state_space_vals,
                     set_path=True,
                     seed=1)

dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                               end_date=end_time,
                                               increment=timedelta(1),
                                               max_dates=1_000_000)

def test_long_term_process_is_single_variable_vasicek() -> None:
    """
    Tests that if the long rate and short rate are uncorrelated, then the sample path
    for the short rate matches exactly what would be given in a single-variable Vasicek
    model
    """
    PASS_THRESH = 1E-10

    test_dmr_vm = DoublyMeanRevertingVasicek(
        short_rate_reversion_speed=short_rate_reversion,
        short_rate_volatility=short_rate_vol,
        long_term_mean=long_term_mean,
        long_term_reversion_speed=long_reversion,
        long_term_volatility=long_vol,
        short_rate_long_term_correlation=0.0,
        start_datetime=start_time,
        end_datetime=end_time
    )

    test_dmr_vm.generate_path(starting_state_space_values=starting_state_space_vals,
                              set_path=True,
                              seed=1)

    test_bm = BrownianMotion(start_date_time=start_time, end_date_time=end_time)

    test_vm = VasicekModel(
        reversion_level=long_term_mean,
        reversion_speed=long_reversion,
        volatility=long_vol,
        brownian_motion=test_bm
    )

    test_vm.generate_path(starting_state_space_values=starting_state_space_vals[1], seed=1, set_path=True)



    plt.figure(figsize=(15, 5))
    #plt.plot(dates, [test_vm(date_obj) for date_obj in dates])
    plt.plot(dates, [test_dmr_vm.state_variables_diffusion_process(date_obj)[0] for date_obj in dates])  # 0 is the long-term process index
    plt.plot(dates, [test_dmr_vm.state_variables_diffusion_process(date_obj)[1] for date_obj in dates])  # 0 is the long-term process index
    plt.legend(['Single Variable Vasicek Model', 'Long Term Process in Double Vasicek Model'])
    plt.show()




