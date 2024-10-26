"""
Unittests for fixedIncome.src.stochastics.base_processes
"""
import itertools
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.base_processes import DiffusionProcess, DriftDiffusionPair
from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator, DayCountConvention

import matplotlib.pyplot as plt

def test_drift_diffusion_pair_from_dictionary() -> None:
    """
    This is to
    """

    def create_drift_difussion_pair(driftTerm, DiffusionTerm):
        return DriftDiffusionPair(
            drift=lambda time, state_variables: driftTerm,
            diffusion=lambda time, state_variables: DiffusionTerm
        )


    PASS_THRESH = 1E-5
    drifts = tuple(range(10))

    # closure problem within the lambdas, they all reference i in the dict comprehension, but only a reference.
    # Hence, all lambdas point to the last i when being used.
    # See https://stackoverflow.com/questions/7368522/weird-behavior-lambda-inside-list-comprehension
    drift_diffusion_dict = {}
    for i in drifts:
        drift_diffusion_dict[f'state_variable_{int(i)}'] = create_drift_difussion_pair(float(i), np.array([0.0 for _ in drifts]))

    for index in range(len(drifts)):
        val = drift_diffusion_dict[f'state_variable_{int(index)}'].drift(0.0, np.array([]))
        assert abs(val - float(index)) < PASS_THRESH


def test_base_drift_process_for_constant_drift_no_volatility() -> None:
    """
    Tests that a trivial base process with constant drift component and zero volatility term
    is just a straight line upwards.
    """
    PASS_THRESH = 1E-5

    test_drift_diffusion_fxcns = DriftDiffusionPair(drift=lambda time, state_variables: 1.0,
                                                    diffusion=lambda time, state_varaibles: np.array([0.0]))

    start_time = datetime(2024, 1, 1, 0)
    end_time = datetime(2053, 12, 31, 23, 59)

    test_bm = BrownianMotion(start_date_time=start_time, end_date_time=end_time)

    test_drift_diffusion_process = DiffusionProcess(
        drift_diffusion_collection={'state_variable_0':test_drift_diffusion_fxcns},
        brownian_motion=test_bm,
        dt=relativedelta(hours=1)
    )

    test_drift_diffusion_process.generate_path(starting_value=np.array([0.0]), set_path=True, seed=1)

    dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                   end_date=end_time,
                                                   increment=timedelta(1),
                                                   max_dates=1_000_000)

    final_value = test_drift_diffusion_process(dates[-1])

    total_accrued_time = sum(
        DayCountCalculator.compute_accrual_length(start_date=start_datetime,
                                                  end_date=end_datetime,
                                                  dcc=test_drift_diffusion_process.day_count_convention)
        for start_datetime, end_datetime in itertools.pairwise(dates)
    )

    assert abs(float(final_value) - total_accrued_time) < PASS_THRESH


