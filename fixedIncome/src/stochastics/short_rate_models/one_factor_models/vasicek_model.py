"""
This script contains the Vasicek Model for the short rate.
Reference, *Fixed Income Securities, 4th Ed.* by Tuckman and Serrat, page 205.

Unit tests are contained in
fixedIncome.tests.test_stochastics.test_short_rate_models.test_one_factor_models.test_vasicek_model.py
"""
from datetime import datetime
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
    """
    def drift_fxcn(time: float, current_value: float) -> float:
        return reversion_scale * (long_term_mean - current_value)

    def diffusion_fxcn(time: float, current_value: float) -> float:
        return volatility

    return DriftDiffusionPair(drift=drift_fxcn, diffusion=diffusion_fxcn)


class VasicekModel(ShortRateModel):
    """
    A class for generating sample paths of the Vasicek short rate model.
    """

    def __init__(self,
                 long_term_mean,
                 reversion_scale,
                 volatility,
                 start_date_time,
                 end_date_time,
                 dt: float = 1/1_000) -> None:
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
                         brownian_motion=bm,
                         dt=dt)  # inherits __call__ from ShortRate class

    def show_drift_diffusion_collection_keys(self) -> tuple[str]:
        """
        Interface method which returns the tuple of keys
        """
        return self.keys

    def generate_path(
            self, starting_value: np.ndarray | float, set_path: bool = True, seed: Optional[int] = None
    ) -> np.ndarray:
        """ Generates the Vasicek solution path through the Euler Discretization method. """
        drift_fxcn, diffusion_fxcn = self.drift_diffusion_pair
        brownian_increments = self.brownian_motion.generate_increments(dt=self.dt, seed=seed).flatten()
        solution = np.empty((1,  len(brownian_increments)+1))
        current_val = float(starting_value)
        time = 0
        for index, shock in enumerate(brownian_increments):
            solution[0, index] = current_val
            drift_increment = drift_fxcn(time, current_val) * self.dt
            diffusion_shock = diffusion_fxcn(time, current_val) * shock  # shock contains sqrt(dt) scaling
            current_val = current_val + drift_increment + diffusion_shock
            time += self.dt

        solution[0, len(brownian_increments)] = current_val  # solution has one more slot
        if set_path:
            self._path = solution

        return solution


    def long_term_variance(self) -> float:
        """
        Returns the long term variance of the Vasicek Model, equal to sigma**2 / 2*a.
        """
        return self.volatility**2 / (2 * self.reversion_scale)


    def plot(self, show_fig: bool = False) -> None:
        """ Produces a plot of  """

        title_str = f'Vasicek Model Sample Path with Parameters\n' \
                    f'Mean {self.long_term_mean}; Volatility {self.volatility}; Reversion Factor {self.reversion_scale}'
        plt.figure(figsize=(15, 6))
        plt.title(title_str)
        date_range = pd.date_range(start=self.start_date_time, end=self.end_date_time, periods=len(self.path.flatten()))
        plt.plot(date_range, self.path.T * 100, linewidth=0.5, alpha=1)
        plt.axhline(self.long_term_mean * 100, linestyle="--", linewidth=0.75, color="grey")
        plt.ylabel('Short Rate (%)')
        plt.grid(alpha=0.25)
        plt.legend(['Sample Short Rate Path', 'Long-Term Mean'], frameon=False)
        if show_fig:
            plt.show()


if __name__ == '__main__':
    from datetime import timedelta
    from fixedIncome.src.scheduling_tools.scheduler import Scheduler

    start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
    end_time = datetime(2073, 10, 15, 0, 0, 0, 0)

    vm = VasicekModel(long_term_mean=0.04,
                      reversion_scale=2.0,
                      volatility=0.02,
                      start_date_time=start_time,
                      end_date_time=end_time)

    path = vm.generate_path(starting_value=0.08, set_path=True, seed=1)
    vm.plot()
    plt.savefig('../../../../../../fixedIncome/docs/images/Vasicek_Short_Rate.png')
    plt.show()

    NUM_CURVES = 1_000
    plt.figure(figsize=(13, 5))
    for seed in range(NUM_CURVES):
        vm.generate_path(starting_value=0.08, set_path=True, seed=seed)
        vm_df_curve = vm.discount_curve()

        dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                       end_date=vm_df_curve.interpolation_values[-1].knot,
                                                       increment=timedelta(1),
                                                       max_dates=1_000_000)

        discount_factors = [vm_df_curve(date_obj) for date_obj in dates]
        plt.plot(dates, discount_factors, color='tab:blue', alpha=0.01, linewidth=0.5)
        print(seed)


    plt.grid(alpha=0.25)
    plt.title(f'Discount Curves from Continuously-Compounding the Vasicek Model Short Rate with Model Parameters\n'
              f'Mean {vm.long_term_mean}; Volatility {vm.volatility}; Reversion Factor {vm.reversion_scale}')
    plt.ylabel('Discount Factor')
    plt.savefig('../../../../../../fixedIncome/docs/images/Vasicek_Discount_Curves.png')
    plt.show()

