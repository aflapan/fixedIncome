"""
The
See Rebonato *Bond Pricing and Yield Curve Modeling* Section 18.7.4.
"""

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.vasicek_model import MultivariateVasicekModel

class StochasticMarketPriceOfRisk(MultivariateVasicekModel):
    """
        d r_t      = kappa_r (theta_t - r_t) dt + sigma_r dW_r
        d theta_t  = kappa_theta (Theta_inf - theta_t) dt + lambda_theta sigma_theta dt + sigma_theta dW_theta
        d L_t      = Kappa_L ( L_inf - L_t) dt + sigma_L dW_L

        with
        dr_t d theta_t        = rho_rtheta dt
        dr_t d lambda_t       = rho_rlambda dt
        d theta_t d lambda_t  = rho_thetalambda dt
    """

    def __init__(self,
                 short_rate_reversion_speed: float,
                 short_rate_vol: float,
                 medium_rate_reversion_speed: float,
                 medium_rate_vol: float,
                 medium_rate_mean: float,
                 market_price_of_risk_reversion_speed: float,
                 market_price_of_risk_vol: float,
                 market_price_of_risk_mean: float,
                 short_rate_medium_rate_correlation: float,
                 short_rate_market_price_of_risk_correlation: float,
                 medium_rate_market_price_of_risk_correlation: float,
                 start_datetime: date | datetime,
                 end_datetime: date | datetime,
                 dt: timedelta | relativedelta = relativedelta(hours=1)
                 ) -> None:

        self.short_rate_reversion_speed = short_rate_reversion_speed
        self.short_rate_vol = short_rate_vol
        self.medium_rate_reversion_speed = medium_rate_reversion_speed
        self.medium_rate_vol = medium_rate_vol
        self.medium_rate_mean = medium_rate_mean
        self.market_price_of_risk_reversion_speed = market_price_of_risk_reversion_speed
        self.market_price_of_risk_vol = market_price_of_risk_vol
        self.market_price_of_risk_mean = market_price_of_risk_mean
        self.short_rate_medium_rate_correlation = short_rate_medium_rate_correlation
        self.short_rate_market_price_of_risk_correlation = short_rate_market_price_of_risk_correlation
        self.medium_rate_market_price_of_risk_correlation = medium_rate_market_price_of_risk_correlation

        brownian_motion = BrownianMotion(start_date_time=start_datetime,
                                         end_date_time=end_datetime,
                                         dimension=3)

        reversion_mat = np.array([
            [self.market_price_of_risk_reversion_speed, 0.0, 0.0],
            [-self.medium_rate_vol, self.medium_rate_reversion_speed, 0.0],
            [0.0, -self.short_rate_reversion_speed, self.short_rate_reversion_speed]
        ])
        avg_market_price_of_risk = (self.medium_rate_vol * self.market_price_of_risk_mean) / self.medium_rate_reversion_speed

        reversion_levels = np.array([self.market_price_of_risk_mean,
                                     self.medium_rate_mean + avg_market_price_of_risk,
                                     self.medium_rate_mean + avg_market_price_of_risk])

        self.variance_mat = np.array([
            [
                self.market_price_of_risk_vol ** 2,
                self.market_price_of_risk_vol * self.medium_rate_vol * self.medium_rate_market_price_of_risk_correlation,
                self.market_price_of_risk_vol * self.short_rate_vol * self.short_rate_market_price_of_risk_correlation
            ],
            [
                self.market_price_of_risk_vol * self.medium_rate_vol * self.medium_rate_market_price_of_risk_correlation,
                self.medium_rate_vol ** 2,
                self.medium_rate_vol * self.short_rate_vol * self.short_rate_medium_rate_correlation
            ],
            [
                self.market_price_of_risk_vol * self.short_rate_vol * self.short_rate_market_price_of_risk_correlation,
                self.medium_rate_vol * self.short_rate_vol * self.short_rate_medium_rate_correlation,
                self.short_rate_vol ** 2
            ]
        ])

        super().__init__(
            short_rate_intercept=0.0,
            short_rate_coefficients=np.array([0.0, 0.0, 1.0]),
            reversion_level=reversion_levels,
            reversion_matrix=reversion_mat,
            volatility_matrix=np.linalg.cholesky(self.variance_mat),
            brownian_motion=brownian_motion,
            dt=dt)



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from fixedIncome.src.scheduling_tools.scheduler import Scheduler

    short_rate_reversion = 0.1
    medium_reversion = 0.25
    market_price_of_risk_reversion_speed = 0.5

    short_rate_vol = 50/10_000  # 10 bps
    medium_vol = 100 / 10_000
    market_price_of_risk_vol = 500/10_000

    short_rate_medium_rate_correlation = 0.5
    short_rate_market_price_of_risk_correlation = 0.01
    medium_rate_market_price_of_risk_correlation = 0.2

    medium_rate_mean = 200/10_000
    market_price_of_risk_mean = 100/10_000

    start_time = datetime(2024, 1, 1, 0)
    end_time = datetime(2053, 12, 31, 23, 59)

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
        end_datetime=end_time ,
        dt = relativedelta(hours=1)
    )

    starting_state_space_vals = np.array([0.05, 0.01, 0.03])

    smpr_model.generate_path(starting_state_space_values=starting_state_space_vals,
                         set_path=True,
                         seed=1)

    dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                   end_date=end_time,
                                                   increment=timedelta(1),
                                                   max_dates=1_000_000)

    values = [smpr_model(date_obj)*100 for date_obj in dates]
    plt.figure(figsize=(13, 5))
    plt.title(f'Sample Path of the Stochastic Market Price of Risk Model')
    plt.plot(dates, values)
    plt.grid(alpha=0.25)
    plt.ylabel('Rate (%)')
    plt.show()


    # Yield plot
    yields = [smpr_model.zero_coupon_bond_yield(date_obj)*100 for date_obj in dates[1:]]
    plt.figure(figsize=(13, 5))
    plt.title(f'Zero-Coupon Bond Yields from the Stochastic Market Price of Risk Model')
    plt.plot(dates[1:], yields,  linewidth=0.75)
    plt.ylabel('Yield (%)')
    plt.grid(alpha=0.25)
    plt.show()
