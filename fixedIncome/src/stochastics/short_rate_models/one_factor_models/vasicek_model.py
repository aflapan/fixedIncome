"""
This script contains the Vasicek Model for the short rate.

Unit tests are contained in
fixedIncome.tests.test_stochastics.test_short_rate_models.test_one_factor_models.test_vasicek_model.py
"""
from datetime import datetime
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.base_short_rate_model import ShortRateModel, DriftDiffusionPair


def vasicek_drift_diffusion(long_term_mean: float, reversion_scale: float, volatility: float) -> DriftDiffusionPair:
    """
    Function to auto-generate the drift and diffusion functions for the Vasicek interest rate
    model. The model has the SDE dr = a * (m - r) dt + sigma dWt where
    a is a positive float representing the mean reversion scaling,
    m is the long-term mean for the interest rate, and
    sigma is the volatility.

    Reference, *Fixed Income Securities, 4th Ed.* by Tuckman and Serrat, page 205.
    """
    def drift_fxcn(time: float, current_value: float) -> float:
        return reversion_scale * (long_term_mean - current_value)

    def diffusion_fxcn(time: float, current_value: float) -> float:
        return volatility

    return DriftDiffusionPair(drift=drift_fxcn, diffusion=diffusion_fxcn)




class VasicekModel(ShortRateModel):
    """
    A class for generating samples of the Vasicek short rate model.
    """

    def __init__(self, long_term_mean, reversion_scale, volatility, start_date_time, end_date_time) -> None:
        self.start_date_time = start_date_time
        self.end_date_time = end_date_time
        self.long_term_mean = long_term_mean
        self.reversion_scale = reversion_scale
        self.volatility = volatility
        self.drift_diffusion_pair = vasicek_drift_diffusion(long_term_mean=long_term_mean,
                                                            reversion_scale=reversion_scale,
                                                            volatility=volatility)

        bm = BrownianMotion(start_date_time=start_date_time,
                            end_date_time=end_date_time)

        self.keys = ('short_rate',)

        super().__init__(drift_diffusion_collection={self.keys[0]: self.drift_diffusion_pair},
                         brownian_motion=bm)


    def __call__(self, datetime_obj: datetime) -> float:
        """
        Allows the user to call the object directly using a datetime object directly.
        If the datetime is an interpolation datetime between the start and end datetimes,
        then the value will be the path value at the corresponding index. Otherwise, the return
        value will be linearly interpolated between the previous and next path values.
        """
        if self.path is None:
            raise ValueError('Brownian Motion called when path is None. '
                             'First call generate_path method with set_path variable set to True.')

        if datetime_obj < self.start_date_time or datetime_obj > self.end_date_time:
            raise ValueError(f'Provided datetime {str(datetime_obj)} is outside of'
                             f'the range {str(self.start_date_time)} to {str(self.end_date_time)}.')

        num_steps = len(self.path)
        time_diff = self.end_date_time - self.start_date_time
        time_spread = self.brownian_motion.to_fraction_in_days(time_diff)
        time_since_start = (datetime_obj - self.start_date_time)

        interpolation_float = (num_steps - 1) * self.brownian_motion.to_fraction_in_days(time_since_start) / time_spread
        interpolated_lower_index = math.floor(interpolation_float)
        interpolated_upper_index = math.ceil(interpolation_float)
        if interpolated_lower_index == interpolated_upper_index:
            return self.path[interpolated_lower_index]

        prev_value = self.path[interpolated_lower_index]
        next_value = self.path[interpolated_upper_index]

        time_since_prev_index = interpolation_float - interpolated_lower_index
        time_to_next_index = interpolated_upper_index - interpolation_float
        total_time = time_since_prev_index + time_to_next_index

        return (time_since_prev_index * next_value + prev_value * time_to_next_index) / total_time




    def show_drift_diffusion_collection_keys(self) -> tuple[str]:
        """
        Interface method which returns the tuple of keys
        """
        return self.keys

    def generate_path(
            self, dt: float, starting_value: np.ndarray | float, set_path: bool = True, seed: Optional[int] = None
    ) -> np.ndarray:
        """ Generates the Vasicek solution path through the Euler Discretization method. """
        drift_fxcn, diffusion_fxcn = self.drift_diffusion_pair
        brownian_increments = self.brownian_motion.generate_increments(dt=dt, seed=seed).flatten()
        solution = np.empty((len(brownian_increments)+1,))
        current_val = float(starting_value)
        time = 0
        for index, shock in enumerate(brownian_increments):
            solution[index] = current_val
            drift_increment = drift_fxcn(time, current_val) * dt
            diffusion_shock = diffusion_fxcn(time, current_val) * shock  # shock contains sqrt(dt) scaling
            current_val = current_val + drift_increment + diffusion_shock
            time += dt

        solution[len(brownian_increments)] = current_val  # solution has one more slot
        if set_path:
            self._path = solution

        return solution


    def long_term_variance(self) -> float:
        """
        Returns the long term variance of the Vasicek Model, equal to sigma**2 / 2*a.
        """
        return self.volatility**2 / (2 * self.reversion_scale)


    def plot(self) -> None:
        """ Produces a plot of  """

        title_str = f'Vasicek Model Sample Path with Parameters\n' \
                    f'Mean {self.long_term_mean}; Volatility {self.volatility}; Reversion Factor {self.reversion_scale}'
        plt.figure(figsize=(15, 6))
        plt.title(title_str)
        date_range = pd.date_range(start=self.start_date_time, end=self.end_date_time, periods=len(self.path))
        plt.plot(date_range, self.path.T * 100, linewidth=0.5, alpha=1)
        plt.axhline(self.long_term_mean * 100, linestyle="--", linewidth=0.75, color="grey")
        plt.ylabel('Short Rate (%)')
        plt.grid(alpha=0.25)
        plt.legend(['Sample Short Rate Path', 'Long-Term Mean'], frameon=False)
        plt.show()


if __name__ == '__main__':
    start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
    end_time = datetime(2023, 11, 15, 0, 0, 0, 0)

    vm = VasicekModel(long_term_mean=0.04,
                      reversion_scale=2.0,
                      volatility=0.02,
                      start_date_time=start_time,
                      end_date_time=end_time)

    path = vm.generate_path(dt=1/100, starting_value=0.08, set_path=True, seed=1)
    vm.plot()

    vm(end_time)

