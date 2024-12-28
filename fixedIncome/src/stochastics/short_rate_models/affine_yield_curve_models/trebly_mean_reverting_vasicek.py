"""
This script contains the implementation for the Trebly-Mean-Reverting Vasicek Model.
Reference: Rebonato *Bond Pricing and Yield Curve Modeling*

"""

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.vasicek_model import MultivariateVasicekModel

class TreblyMeanRevertingVasicek(MultivariateVasicekModel):
    """
        d r_t      = kappa_r (R_t - r_t) dt + sigma_r dW_r
        d R_t      = kappa_R (L_t - R_t) dt + sigma_R dW_R
        d L_t      = Kappa_L ( L_inf - L_t) dt + sigma_L dW_L

        with
        dr_t dL_t  = rho_rL dt
        dr_t dR_t  = rho_rR dt
        dR_t dL_t  = rho_RL dt
    """
    def __init__(self,
                 short_rate_reversion_speed: float,
                 short_rate_volatility: float,
                 medium_rate_reversion_speed: float,
                 medium_rate_volatility: float,
                 long_term_mean: float,
                 long_term_reversion_speed: float,
                 long_term_volatility: float,
                 short_rate_medium_rate_correlation: float,
                 short_rate_long_term_correlation: float,
                 medium_rate_long_term_correlation_correlation: float,
                 start_datetime: date | datetime,
                 end_datetime: date | datetime,
                 dt: timedelta | relativedelta = relativedelta(hours=1)) -> None:

        self.short_rate_reversion_speed = short_rate_reversion_speed
        self.short_rate_volatility = short_rate_volatility
        self.medium_rate_reversion_speed = medium_rate_reversion_speed
        self.medium_rate_volatility = medium_rate_volatility
        self.long_term_mean = long_term_mean
        self.long_term_reversion_speed = long_term_reversion_speed
        self.long_term_volatility = long_term_volatility
        self.short_rate_medium_rate_correlation = short_rate_medium_rate_correlation
        self.short_rate_long_term_correlation = short_rate_long_term_correlation
        self.medium_rate_long_term_correlation_correlation = medium_rate_long_term_correlation_correlation

        brownian_motion = BrownianMotion(start_date_time=start_datetime,
                                         end_date_time=end_datetime,
                                         dimension=3)

        reversion_mat = np.array([
            [self.long_term_reversion_speed, 0.0, 0.0],
            [-self.medium_rate_reversion_speed, self.medium_rate_reversion_speed, 0.0],
            [0.0, -self.short_rate_reversion_speed, self.short_rate_reversion_speed]
        ])

        self.variance_mat = np.array([
            [
                self.long_term_volatility**2,
                self.long_term_volatility * self.medium_rate_volatility * self.medium_rate_long_term_correlation_correlation,
                self.long_term_volatility * self.short_rate_volatility * self.short_rate_long_term_correlation
            ],
            [
                self.long_term_volatility * self.medium_rate_volatility * self.medium_rate_long_term_correlation_correlation,
                self.medium_rate_volatility**2,
                self.medium_rate_volatility * self.short_rate_volatility * self.short_rate_medium_rate_correlation
            ],
            [
                self.long_term_volatility * self.short_rate_volatility * self.short_rate_long_term_correlation,
                self.medium_rate_volatility * self.short_rate_volatility * self.short_rate_medium_rate_correlation,
                self.short_rate_volatility**2
            ]
        ])

        super().__init__(
            short_rate_intercept=0.0,
            short_rate_coefficients=np.array([0.0, 0.0,  1.0]),
            reversion_level=np.array([self.long_term_mean, self.long_term_mean, self.long_term_mean]),
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

    short_rate_reversion = 1.0547
    medium_reversion = 0.6358
    long_reversion = 0.0165

    short_rate_vol = 50/10_000  # 50 bps
    medium_vol = 109.2 / 10_000
    long_vol = 96.4/10_000

    short_medium_corr = 0.5
    short_long_corr = 0.01
    medium_long_cor = 0.212

    long_term_mean = 0.10555  # 10.555%

    start_time = datetime(2024, 1, 1, 0)
    end_time = datetime(2053, 12, 31, 23, 59)

    tmr_vm = TreblyMeanRevertingVasicek(
        short_rate_reversion_speed=short_rate_reversion,
        short_rate_volatility=short_rate_vol,
        medium_rate_reversion_speed=medium_reversion,
        medium_rate_volatility=medium_vol,
        long_term_mean=long_term_mean,
        long_term_reversion_speed=long_reversion,
        long_term_volatility=long_vol,
        short_rate_medium_rate_correlation=short_medium_corr,
        short_rate_long_term_correlation=short_long_corr,
        medium_rate_long_term_correlation_correlation=medium_long_cor,
        start_datetime=start_time,
        end_datetime=end_time
    )

    starting_state_space_vals = np.array([0.05, 0.01, 0.03])

    tmr_vm.generate_path(starting_state_space_values=starting_state_space_vals,
                         set_path=True,
                         seed=1)

    dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                   end_date=end_time,
                                                   increment=timedelta(1),
                                                   max_dates=1_000_000)

    values = [tmr_vm(date_obj)*100 for date_obj in dates]
    plt.figure(figsize=(13, 5))
    plt.title(f'Sample Path of the Trebly-Mean Reverting Vasicek Model')
    plt.axhline(tmr_vm.long_term_mean * 100, linestyle="--", linewidth=0.75, color="grey")
    plt.plot(dates, [tmr_vm.state_variables_diffusion_process(date_obj)[0]*100 for date_obj in dates], linewidth=0.75)
    plt.plot(dates, [tmr_vm.state_variables_diffusion_process(date_obj)[1] * 100 for date_obj in dates], linewidth=0.75)
    plt.plot(dates, [tmr_vm.state_variables_diffusion_process(date_obj)[2] * 100 for date_obj in dates], linewidth=0.75)
    plt.legend(['Long Term Mean', 'Long Term Process', 'Medium Term Process', 'Short Rate Process'], frameon=False)
    plt.grid(alpha=0.25)
    plt.ylabel('Rate (%)')
    plt.show()


    # Yield plot
    yields = [tmr_vm.zero_coupon_bond_yield(date_obj)*100 for date_obj in dates[1:]]
    plt.figure(figsize=(13, 5))
    plt.title(f'Zero-Coupon Bond Yields from the Trebly-Mean Reverting Vasicek Model\n'
              f'Model Parameters: Long-Term Mean {round(long_term_mean, 3)}, Long Term Volatiltiy {round(long_vol, 3)}\n')
    plt.plot(dates[1:], yields,  linewidth=0.75)
    plt.ylabel('Yield (%)')
    plt.grid(alpha=0.25)
    plt.show()


    # Yield Volatility plot
    yield_vols = [tmr_vm.yield_volatility(date_obj) for date_obj in dates[1:]]
    plt.figure(figsize=(13, 5))
    plt.title(f'Zero-Coupon Bond Yield Volatilites from the Trebly-Mean Reverting Vasicek Model\n'
              f'Model Parameters: Long-Term Mean {round(long_term_mean, 3)}, Long Term Volatiltiy {round(long_vol, 3)}\n')
    plt.plot(dates[1:], yield_vols,  linewidth=0.75)
    plt.ylabel('Yield Volatility')
    plt.grid(alpha=0.25)
    plt.show()

    # Convexity plot
    yields = [tmr_vm.zero_coupon_bond_yield(date_obj) * 100 for date_obj in dates[1:]]
    avg_expected_sr = [tmr_vm.average_expected_short_rate(date_obj) * 100 for date_obj in dates[1:]]
    yield_convexities = [tmr_vm.yield_convexity(date_obj) * 10_000 for date_obj in dates[1:]]

    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    plt.title(f'Convexity in the Trebly-Mean Reverting Model by Comparing Yields and Conditional Short Rate Averages')
    plt.plot(dates[1:], yields,  color='tab:blue')
    plt.plot(dates[1:], avg_expected_sr, color='darkred')
    plt.ylabel('Yield (%)')
    plt.grid(alpha=0.25)
    plt.legend([
        'Yield',
        'Avg. Expected Short Rate'],
        loc='center left', frameon=False)

    ax2 = ax.twinx()

    plt.plot(dates[1:], yield_convexities, color='grey', linestyle='dotted')
    plt.ylabel('Convexity Adjustment (basis points)')
    plt.legend(['Difference (Right)'],
               loc='center right', frameon=False)
    plt.show()





