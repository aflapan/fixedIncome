"""
This module contains the unit tests for
fixedIncome.src.stochastics.short_rate_models.one_factor_models.vasicek_model.py
"""

from datetime import datetime
import numpy as np
import pandas as pd
from fixedIncome.src.stochastics.short_rate_models.one_factor_models.vasicek_model import VasicekModel


start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
end_time = datetime(2043, 10, 15, 0, 0, 0, 0)

vm = VasicekModel(long_term_mean=0.04,
                  reversion_scale=2.0,
                  volatility=0.02,
                  start_date_time=start_time,
                  end_date_time=end_time)



def test_long_term_mean() -> None:
    """ Tests that the mean of the sample path is within a tolerance. """
    path = vm.generate_path(dt=1 / 10, starting_value=0.04, set_path=True, seed=1)  # start at the mean
    PASS_THRESH = 1E-4
    sample_mean = np.mean(vm.path)
    assert abs(sample_mean - 0.04) < PASS_THRESH


def test_model_evaluates_to_path_on_interpoalting_dates() -> None:
    """
    Tests the callable feature of the model correctly gives the path values
    when the datetime object used as an argument is an interpolation date.
    """
    PASS_THRESH = 1E-10
    date_range = pd.date_range(start=start_time, end=end_time, periods=len(vm.path)).to_pydatetime()

    assert all(abs(float(vm(date_time_obj)) - vm.path[index]) < PASS_THRESH
               for index, date_time_obj in enumerate(date_range))


