"""
A suit of unit tests for fixedIncome.src.stochastics.brownian_motion.py
"""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from fixedIncome.src.stochastics.brownian_motion import BrownianMotion



def test_brownian_motion_evaluates_to_path_values_when_called_on_interpolating_dates() -> None:
    """
    Tests that the brownian motion object produces the numpy path values when called on
    datetimes which lie exactly on the 1/len(path) grid between the start and end datetimes
    for the path.
    """
    PASS_THRESH = 1E-10  # want to test close to exact

    start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
    end_time = datetime(2023, 10, 16, 0, 0, 0, 0)

    bm = BrownianMotion(start_date_time=start_time,
                        end_date_time=end_time,
                        dimension=1,
                        correlation_matrix=None)

    bm.generate_path(dt=timedelta(hours=1), seed=1)

    date_range = pd.date_range(start=start_time, end=end_time, periods=bm.path.shape[1]).to_pydatetime()

    assert all(abs(float(bm(date_time_obj)) - bm.path[0, index]) < PASS_THRESH
               for index, date_time_obj in enumerate(date_range))


def test_correlation_of_brownian_increments_is_within_error_of_provided_matrix() -> None:
    """
    Tests that the increments of the Brownian Motion sample path are
    correlated and that the estimated correlation matrix is within
    error of the true correlation matrix.
    """
    rho = 0.5
    correlation_matrix = np.array([[1.0, rho], [rho, 1.0]])
    start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
    end_time = datetime(2023, 10, 16, 0, 0, 0, 0)

    bm = BrownianMotion(start_date_time=start_time,
                        end_date_time=end_time,
                        dimension=2,
                        correlation_matrix=correlation_matrix)

    bm.generate_path(dt=timedelta(hours=1), seed=1)

    path_increments = np.diff(bm.path, axis=1)
    np.allclose(correlation_matrix, np.corrcoef(path_increments))