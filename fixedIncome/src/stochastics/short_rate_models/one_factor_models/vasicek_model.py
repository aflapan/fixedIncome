"""
This script contains the Vasicek Model for the short rate.
Reference, *Fixed Income Securities, 4th Ed.* by Tuckman and Serrat, page 205.

Unit tests are contained in
fixedIncome.tests.test_stochastics.test_short_rate_models.test_one_factor_models.test_vasicek_model.py
"""
from datetime import datetime, date
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Optional
from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.base_short_rate_model import ShortRateModel, DriftDiffusionPair
from fixedIncome.src.stochastics.short_rate_models.affine_model_mixin import AffineModelMixin
from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.scheduling_tools.scheduler import Scheduler


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


class VasicekModel(AffineModelMixin, ShortRateModel):
    """
    A class for generating sample paths of the Vasicek short rate model.
    """

    def __init__(self,
                 long_term_mean,
                 reversion_speed,
                 volatility,
                 start_date_time,
                 end_date_time,
                 day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_OVER_ACTUAL,
                 dt: float = 1/1_000) -> None:
        self.start_date_time = start_date_time
        self.end_date_time = end_date_time
        self.long_term_mean = long_term_mean
        self.reversion_speed = reversion_speed
        self.volatility = volatility
        self.drift_diffusion_pair = vasicek_drift_diffusion(long_term_mean=long_term_mean,
                                                            reversion_scale=reversion_speed,
                                                            volatility=volatility)

        bm = BrownianMotion(start_date_time=start_date_time,
                            end_date_time=end_date_time)

        self.keys = ('short_rate',)


        super().__init__(drift_diffusion_collection={self.keys[0]: self.drift_diffusion_pair},
                         brownian_motion=bm,
                         day_count_convention=day_count_convention,
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

        self._reset_paths_and_curves()

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


    def conditional_variance(self, datetime_obj: datetime) -> float:
        """
        Returns the conditional variance of the Vasicek Model, equal to sigma**2 / 2*a * (1- exp(-2 k * (t-t_0))).
        """
        accrual = DayCountCalculator.compute_accrual_length(self.start_date_time,
                                                            datetime_obj,
                                                            self.day_count_convention)

        shrinkage_factor = 1 - math.exp(-2 * self.reversion_speed * accrual)
        return self.volatility**2 / (2 * self.reversion_speed) * shrinkage_factor

    def conditional_mean(self, initial_value_short_rate: float, datetime_obj: datetime) -> float:
        """
        Returns the conditional mean of the Vasicek short rate given the initial value of the short rate r_0.
        """
        accrual = DayCountCalculator.compute_accrual_length(self.start_date_time,
                                                            datetime_obj,
                                                            self.day_count_convention)
        weight = math.exp(-self.reversion_speed * accrual)
        return weight * initial_value_short_rate + (1 - weight) * self.long_term_mean

    # Affine model functions

    def _create_bond_price_coeffs(self, maturity_date: date) -> None:
        """
        Overrides the class method for setting the coefficients for calculating a zero-coupon bond price.
        """
        accrual = DayCountCalculator.compute_accrual_length(self.start_date_time,
                                                            maturity_date,
                                                            self.day_count_convention)
        coefficient = -(1 - math.exp(-self.reversion_speed * accrual)) / self.reversion_speed
        term1 = (self.long_term_mean - self.volatility**2/(2 * self.reversion_speed**2)) * (coefficient + accrual)
        term2 = self.volatility**2 / (4 * self.reversion_speed) * coefficient**2
        intercept = -term1 - term2
        self.bond_price_state_variable_coeffs = {'intercept': intercept, 'coefficient': coefficient}

    def _create_bond_yield_coeffs(self, maturity_date) -> None:
        """
        Overrides the class method for setting the coefficients for calculating a zero-coupon bond yield.
        Reference:  Pages 165-171 of Rebonato *Bond Pricing and Yield Curve Modeling*.
        """
        accrual = DayCountCalculator.compute_accrual_length(self.start_date_time,
                                                            maturity_date,
                                                            self.day_count_convention)
        coefficient = (1 - math.exp(-self.reversion_speed * accrual)) / (self.reversion_speed * accrual)
        term1 = (self.long_term_mean - self.volatility**2 / (2 * self.reversion_speed**2)) * (coefficient + accrual)
        term2 = self.volatility ** 2 / (4 * self.reversion_speed) * coefficient ** 2
        intercept = (term1+term2)/accrual
        self.yield_state_variable_coeffs = {'intercept': intercept, 'coefficient': coefficient}


    def zero_coupon_bond_price(self, initial_value_short_rate: float, maturity_date: date | datetime) -> float:
        """
        Overrides the abstract class method

        Reference:  Pages 165-171 of Rebonato *Bond Pricing and Yield Curve Modeling*.
        """
        self._create_bond_price_coeffs(maturity_date)
        return math.exp(self.bond_price_state_variable_coeffs['intercept']
                        + self.bond_price_state_variable_coeffs['coefficient'] * initial_value_short_rate)

    def zero_coupon_yield(self, initial_value_short_rate: float, maturity_date: date | datetime) -> float:
        """
        Overrides the abstract class method
        Reference:  Pages 165-171 of Rebonato *Bond Pricing and Yield Curve Modeling*.
        """
        self._create_bond_yield_coeffs(maturity_date)
        yield_value = self.yield_state_variable_coeffs['intercept'] \
                      + self.yield_state_variable_coeffs['coefficient'] * initial_value_short_rate

        return yield_value

    def yield_convexity(self, maturity_date: date | datetime) -> float:
        """
        Calculates the theoretical convexity of the Vasicek Model in yield terms. I.e., it calculates
        the difference between the actual yield for a time T zero-coupon bond and the pseudo-yield
        constructed by pricing a zero-coupon bond using the conditional expected value of the short rate path
        """
        accrual = DayCountCalculator.compute_accrual_length(self.start_date_time,
                                                            maturity_date,
                                                            self.day_count_convention)
        self._create_bond_price_coeffs(maturity_date)
        term1 = (-self.bond_price_state_variable_coeffs['coefficient'] - accrual) / (2 * self.reversion_speed**2)
        term2 = self.bond_price_state_variable_coeffs['coefficient']**2 / (4 * self.reversion_speed)
        return (self.volatility**2 / accrual) * (term1 + term2)

    def yield_volatility(self, maturity_date: date | datetime) -> float:
        """
        Returns the volatility for a zero-coupon bond's yield for time T-maturity bonds.
        """
        accrual = DayCountCalculator.compute_accrual_length(self.start_date_time,
                                                            maturity_date,
                                                            self.day_count_convention)
        self._create_bond_price_coeffs(maturity_date)
        return -(self.bond_price_state_variable_coeffs['coefficient'] / accrual) * self.volatility


    # Plotting

    def plot(self, show_fig: bool = False) -> None:
        """ Produces a plot of  """
        initial_value = float(self.path[0,0])
        title_str = f'Vasicek Model Sample Path with Parameters\n' \
                    f'Mean {self.long_term_mean}; Volatility {self.volatility}; Reversion Speed {self.reversion_speed}'
        plt.figure(figsize=(15, 6))
        plt.title(title_str)
        date_range = pd.date_range(start=self.start_date_time, end=self.end_date_time, periods=len(self.path.flatten())).to_pydatetime()
        plt.plot(date_range, self.path.T * 100, linewidth=0.5, alpha=1)
        plt.plot(date_range, [self.conditional_mean(initial_value, datetime_obj) * 100
                              for datetime_obj in date_range], linestyle='dotted', color='tab:red')
        plt.axhline(self.long_term_mean * 100, linestyle="--", linewidth=0.75, color="grey")
        plt.ylabel('Short Rate (%)')
        plt.grid(alpha=0.25)
        plt.legend(['Sample Short Rate Path', 'Conditional Mean', 'Long-Term Mean'], frameon=False)
        if show_fig:
            plt.show()


if __name__ == '__main__':
    from datetime import timedelta
    import itertools
    from fixedIncome.src.scheduling_tools.scheduler import Scheduler

    # ----------------------------------------------------------------
    start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
    end_time = datetime(2053, 10, 15, 0, 0, 0, 0)
    dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                   end_date=end_time,
                                                   increment=timedelta(1),
                                                   max_dates=1_000_000)

    vm = VasicekModel(long_term_mean=0.04,
                      reversion_speed=0.1,
                      volatility=0.02,
                      start_date_time=start_time,
                      end_date_time=end_time)

    path = vm.generate_path(starting_value=0.08, set_path=True, seed=1)
    admissible_dates = [date_obj for date_obj in dates if date_obj < vm.end_date_time]

    #vm.plot()
    #plt.savefig('../../../../../../fixedIncome/docs/images/Vasicek_Short_Rate.png')
    #plt.show()

    # DISCOUNT CURVES
    #NUM_CURVES = 20
    #plt.figure(figsize=(13, 5))
    #for seed in range(NUM_CURVES):
    #    vm.generate_path(starting_value=0.08, set_path=True, seed=seed)
    #    vm_df_curve = vm.discount_curve()
    #    discount_factors = [vm_df_curve(date_obj) for date_obj in admissible_dates]
    #    plt.plot(admissible_dates, discount_factors, color='tab:blue', alpha=1, linewidth=0.5)
    #    print(seed)

    #plt.grid(alpha=0.25)
    #plt.title(f'Discount Curves from Continuously-Compounding {NUM_CURVES} Vasicek Model Short Rate Paths\n'
    #          f'with Model Parameters Mean {vm.long_term_mean}; '
    #          f'Volatility {vm.volatility}; Reversion Speed {vm.reversion_speed}')
    #plt.ylabel('Discount Factor')
    #plt.savefig('../../../../../../fixedIncome/docs/images/Vasicek_Discount_Curves.png')
    #plt.show()

    # CONVEXITY
    NUM_CURVES = 10_000
    INITIAL_SHORT_RATE = 0.08
    vm_avg_short_rate = np.zeros((1, len(admissible_dates)))
    vm_avg_df = np.zeros((1, len(admissible_dates)))
    vm_avg_integrated_path = np.zeros((1, len(admissible_dates)))

    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    for seed in range(NUM_CURVES):
        vm.generate_path(starting_value=INITIAL_SHORT_RATE, set_path=True, seed=seed)
        vm.generate_integrated_path(datetimes=admissible_dates)
        vm_avg_integrated_path += vm._integrated_path
        vm_avg_df += np.exp(-vm._integrated_path)
        print(seed)

    accruals = [DayCountCalculator.compute_accrual_length(start_time, datetime_obj, DayCountConvention.ACTUAL_OVER_ACTUAL)
                for datetime_obj in admissible_dates]

    yields_for_prices = [-100 / acc * math.log(price) for acc, price in zip(accruals[1:], vm_avg_df.flatten()[1:] / NUM_CURVES)]
    plt.plot(admissible_dates[1:], yields_for_prices, color='tab:blue')

    affine_price_yields = [-100 / acc * math.log(vm.zero_coupon_bond_price(initial_value_short_rate=INITIAL_SHORT_RATE, maturity_date=date))
                          for acc, date in zip(accruals[1:], admissible_dates[1:])]

    plt.plot(admissible_dates[1:], affine_price_yields, color='tab:blue', linestyle='dotted')


    yields_for_avg_short_rate = [-100 / acc * math.log(price)
                                 for acc, price in zip(accruals[1:], np.exp(-vm_avg_integrated_path.flatten()[1:] / NUM_CURVES))]

    # Compute Yields for Conditional Mean
    running_integral = 0.0
    integral_vals = [running_integral]
    for start_dt, end_dt in itertools.pairwise(admissible_dates):
        start_rate = vm.conditional_mean(INITIAL_SHORT_RATE, start_dt)
        end_rate = vm.conditional_mean(INITIAL_SHORT_RATE, end_dt)
        min_rate, max_rate = min(start_rate, end_rate), max(start_rate, end_rate)
        accrual = DayCountCalculator.compute_accrual_length(start_dt, end_dt, vm.day_count_convention)

        if min_rate < 0:
            next_trapezoid_val = max_rate * accrual + (min_rate - max_rate) * accrual / 2
        else:
            next_trapezoid_val = min_rate * accrual + (max_rate - min_rate) * accrual / 2
        running_integral += next_trapezoid_val
        integral_vals.append(running_integral)

    price_for_conditional_mean = [math.exp(-integral_val) for integral_val in integral_vals]
    yield_for_conditional_mean_short_rate = [-100 / acc * math.log(price)
                                             for acc, price in zip(accruals[1:], price_for_conditional_mean[1:])]

    plt.plot(admissible_dates[1:], yields_for_avg_short_rate, color='darkred')
    plt.plot(admissible_dates[1:], yield_for_conditional_mean_short_rate, color='darkred', linestyle='dotted')

    plt.grid(alpha=0.25)
    plt.title(f'Yield Convexity in the Vasicek Model: Yields for Zero-Coupon Bonds Generates from {"{:,}".format(NUM_CURVES)} Short Rate Sample Paths\n'
              f'and Comparing it to the Yields Constructed by Pricing Bonds using the Average Short Rate Path;'
              f'\nModel Parameters: Mean {vm.long_term_mean}; Volatility {vm.volatility}; Reversion Speed {vm.reversion_speed}')
    plt.ylabel('Yield (%)')
    plt.legend(['Monte Carlo Yield',
                'Yield from Affine Model Coefficients',
                'Yield Constructed from the Monte Carlo Average Short Rate Path',
                'Yield Constructed on Conditional Short Rate Mean'],
               loc='lower left', frameon=False)

    ax2 = ax.twinx()
    monte_carlo_diff = [(yield_price - yield_avg_rate) * 100
                        for yield_price, yield_avg_rate in zip(yields_for_prices, yields_for_avg_short_rate)]

    affine_diff = [vm.yield_convexity(datetime_obj) * 10_000 for datetime_obj in admissible_dates[1:]]

    plt.plot(admissible_dates[1:], monte_carlo_diff, color='grey')
    plt.plot(admissible_dates[1:], affine_diff, color='grey', linestyle='dotted')

    plt.legend(['Monte Carlo Convexity Adjustment (Right)', 'Affine Convexity Adjustment (Right)'],
               loc='upper right', frameon=False)

    ax2.set_ylabel('Difference (bp)')
    plt.savefig('../../../../../../fixedIncome/docs/images/Vasicek_Convexity.png')
    plt.show()

