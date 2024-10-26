"""
This script contains the implementation for the Doubly-Mean-Reverting Vasicek Model.

Unit tests are contained in
fixedIncome.tests.test_stochastics.test_short_rate_models.test_affine_yield_curve_models.test_doubly_mean_reverting_vasicek.py
"""

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.vasicek_model import MultivariateVasicekModel

class DoublyMeanRevertingVasicek(MultivariateVasicekModel):
    """
        d r_t      = kappa (theta_t - r_t) dt + sigma_r dW_r
        d theta_t  = alpha ( beta - theta_t) dt + sigma_theta dW_theta

         dW_r dW_theta  = rho dt

    """

    def __init__(self,
                 short_rate_reversion_speed: float,
                 short_rate_volatility: float,
                 long_term_mean: float,
                 long_term_reversion_speed: float,
                 long_term_volatility: float,
                 short_rate_long_term_correlation: float,
                 start_datetime: date | datetime,
                 end_datetime: date | datetime,
                 dt: timedelta | relativedelta = relativedelta(hours=1)
                 ) -> None:

        self.short_rate_reversion_speed = short_rate_reversion_speed
        self.short_rate_volatility = short_rate_volatility
        self.long_term_mean = long_term_mean
        self.long_term_reversion_speed = long_term_reversion_speed
        self.long_term_volatility = long_term_volatility
        self.short_rate_long_term_correlation = short_rate_long_term_correlation

        reversion_mat = np.array([[self.long_term_reversion_speed, 0.0],
                                  [-self.short_rate_reversion_speed, self.short_rate_reversion_speed]])

        across_term_volatility = self.short_rate_volatility * self.long_term_volatility * self.short_rate_long_term_correlation

        self.variance_mat = np.array([[self.long_term_volatility**2, across_term_volatility],
                                      [across_term_volatility, self.short_rate_volatility**2]])


        brownian_motion = BrownianMotion(start_date_time=start_datetime,
                                         end_date_time=end_datetime,
                                         dimension=2)

        super().__init__(
            short_rate_intercept=0.0,
            short_rate_coefficients=np.array([0.0, 1.0]),
            reversion_level=np.array([self.long_term_mean, self.long_term_mean]),
            reversion_matrix=reversion_mat,
            volatility_matrix=np.linalg.cholesky(self.variance_mat),
            brownian_motion=brownian_motion,
            dt=dt)


if __name__ == '__main__':
    """
    The following example is adapted from Tuckman and Serrat *Fixed Income Securities, 4th Ed.* page 216.
    """
    import matplotlib.pyplot as plt
    from fixedIncome.src.scheduling_tools.scheduler import Scheduler

    short_rate_reversion = 0.5
    long_reversion = 0.05

    short_rate_vol = 50/10_000
    long_vol = 25/10_000
    short_long_corr = 0.25

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

    NUM_PATHS = 1

    dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                   end_date=end_time,
                                                   increment=timedelta(1),
                                                   max_dates=1_000_000)

    starting_state_space_vals = np.array([0.075, 0.05])

    plt.figure(figsize=(13, 5))
    for seed in range(NUM_PATHS):

        dmr_vm.generate_path(starting_state_space_values=starting_state_space_vals,
                             set_path=True,
                             seed=seed)

        values = [dmr_vm(date_obj)*100 for date_obj in dates]
        plt.plot(dates, values, linewidth=0.5, color='tab:blue')

    plt.axhline(dmr_vm.long_term_mean * 100, linestyle="--", linewidth=0.75, color="grey")
    plt.grid(alpha=0.25)
    plt.show()



    yields = [dmr_vm.zero_coupon_bond_yield(date_obj)*100 for date_obj in dates[1:]]
    plt.figure(figsize=(13, 5))
    plt.title('Bond Yields for the Doubly-Mean Reverting Vasicek Model')
    plt.plot(dates[1:], yields)
    plt.grid(alpha=0.25)
    plt.show()

