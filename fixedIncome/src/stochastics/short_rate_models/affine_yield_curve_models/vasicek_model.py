"""
This script contains the Vasicek Model for the short rate.
Reference, *Fixed Income Securities, 4th Ed.* by Tuckman and Serrat, page 205.

Unit tests are contained in
fixedIncome.tests.test_stochastics.test_short_rate_models.test_affine_yield_curve_models.test_vasicek_model.py
"""
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Optional
from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.base_processes import DiffusionProcess
from fixedIncome.src.stochastics.short_rate_models.base_short_rate_model import ShortRateModel
from fixedIncome.src.stochastics.base_processes import DriftDiffusionPair
from fixedIncome.src.stochastics.short_rate_models.affine_model_mixin import AffineModelMixin
from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator


def vasicek_drift_diffusion(reversion_level: float, reversion_scale: float, volatility: float) -> DriftDiffusionPair:
    """
    Function to auto-generate the drift and diffusion functions for the Vasicek interest rate
    model. The model has the SDE dr = k * (m - r) dt + sigma dWt where
    k is a positive float representing the reversion speed,
    m is the long-term mean for the interest rate, and
    sigma is the volatility.
    """
    def drift_fxcn(time: float, current_value: float) -> np.array:
        return np.array([reversion_scale * (reversion_level - current_value)])

    def diffusion_fxcn(time: float, current_value: float) -> np.array:
        return np.array([volatility])

    return DriftDiffusionPair(drift=drift_fxcn, diffusion=diffusion_fxcn)


class VasicekModel(ShortRateModel, AffineModelMixin):
    """
    A class for generating sample paths of the one-dimensional Vasicek short rate model with short rate dynamics
        dr_t = k(theta - r_t)dt + sigma dW_t
    """

    def __init__(self,
                 reversion_level: float,
                 reversion_speed: float,
                 volatility: float,
                 brownian_motion: BrownianMotion,
                 dt: timedelta | relativedelta = relativedelta(hours=1)) -> None:

        self.long_term_mean = reversion_level
        self.reversion_speed = reversion_speed
        self.volatility = volatility
        self.drift_diffusion_pair = vasicek_drift_diffusion(reversion_level=reversion_level,
                                                            reversion_scale=reversion_speed,
                                                            volatility=volatility)

        diffusion_process = DiffusionProcess(drift_diffusion_collection={'short rate': self.drift_diffusion_pair},
                                             brownian_motion=brownian_motion,
                                             dt=dt)

        super().__init__(short_rate_transformation=lambda short_rate: short_rate,
                         state_variables_diffusion_process=diffusion_process)

        self.convexity_limit = -self.volatility**2 / (2 * self.reversion_speed**2)


    def short_rate_variance(self, datetime_obj: datetime) -> float:
        """
        Returns the conditional variance of the Vasicek Model, equal to sigma**2 / 2*a * (1- exp(-2 k * (t-t_0))).
        """
        accrual = DayCountCalculator.compute_accrual_length(self.start_date_time,
                                                            datetime_obj,
                                                            self.day_count_convention)

        shrinkage_factor = 1 - math.exp(-2 * self.reversion_speed * accrual)
        return self.volatility**2 / (2 * self.reversion_speed) * shrinkage_factor

    def expected_short_rate(self,
                            maturity_date: date | datetime,
                            purchase_date: Optional[date | datetime] = None) -> float:
        """
        Returns the conditional mean of the Vasicek short rate given the initial value of the short rate r_0.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        accrual = DayCountCalculator.compute_accrual_length(purchase_date,
                                                            maturity_date,
                                                            self.day_count_convention)
        weight = math.exp(-self.reversion_speed * accrual)
        return weight * self(purchase_date) + (1 - weight) * self.long_term_mean

    # Affine model functions

    def _create_bond_price_coeffs(self, maturity_date: date, purchase_date: Optional[date | datetime] = None) -> None:
        """
        Overrides the class method for setting the coefficients for calculating a zero-coupon bond price.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        accrual = DayCountCalculator.compute_accrual_length(purchase_date,
                                                            maturity_date,
                                                            self.day_count_convention)
        coefficient = -(1 - math.exp(-self.reversion_speed * accrual)) / self.reversion_speed
        term1 = (self.long_term_mean + self.convexity_limit) * (coefficient + accrual)
        term2 = self.volatility**2 / (4 * self.reversion_speed) * coefficient**2
        intercept = -term1 - term2
        self.price_state_variable_coeffs = {'intercept': intercept, 'coefficient': coefficient}

    def _calculate_bond_price_coeff_derivatives_wrt_maturity(self, purchase_date, maturity_date) -> dict[str, float]:
        """
        Returns a dictionary with the same keys ('intercept' and 'coefficient') as
        the dictionary formed by self._create_bond_price_coeffs, but instead returns each bond price
        coefficient's derivative with respect to maturity time T.
        """
        accrual = DayCountCalculator.compute_accrual_length(start_date=purchase_date,
                                                            end_date=maturity_date,
                                                            dcc=self.day_count_convention)

        exp_factor = math.exp(-self.reversion_speed * accrual)
        one_minus_exp = (1-exp_factor)
        intercept_deriv = -one_minus_exp * (self.long_term_mean + one_minus_exp * self.convexity_limit)

        return {'intercept': intercept_deriv, 'coefficient': -exp_factor}

    def _create_bond_yield_coeffs(self, maturity_date) -> None:
        """
        Overrides the class method for setting the coefficients for calculating a zero-coupon bond yield.
        Reference:  Pages 165-171 of Rebonato *Bond Pricing and Yield Curve Modeling*.
        """
        accrual = DayCountCalculator.compute_accrual_length(self.start_date_time,
                                                            maturity_date,
                                                            self.day_count_convention)
        coefficient = -(1 - math.exp(-self.reversion_speed * accrual)) / self.reversion_speed
        term1 = (self.long_term_mean + self.convexity_limit) * (coefficient + accrual)
        term2 = self.volatility ** 2 / (4 * self.reversion_speed) * coefficient**2
        intercept = (term1+term2)/accrual
        self.yield_state_variable_coeffs = {'intercept': intercept, 'coefficient': -coefficient/accrual}


    def zero_coupon_bond_price(self, maturity_date: date | datetime, purchase_date: Optional[date | datetime] = None) -> float:
        """
        Overrides the abstract class method.

        Reference:  Pages 165-171 of Rebonato *Bond Pricing and Yield Curve Modeling*.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        self._create_bond_price_coeffs(maturity_date)
        return math.exp(self.price_state_variable_coeffs['intercept']
                        + self.price_state_variable_coeffs['coefficient'] * self(purchase_date))

    def zero_coupon_yield(self,
                          maturity_date: date | datetime,
                          purchase_date: Optional[date | datetime] = None) -> float:
        """
        Overrides the abstract class method
        Reference:  Pages 165-171 of Rebonato *Bond Pricing and Yield Curve Modeling*.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        self._create_bond_yield_coeffs(maturity_date)
        yield_value = self.yield_state_variable_coeffs['intercept'] \
                      + self.yield_state_variable_coeffs['coefficient'] * self(purchase_date)

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
        term1 = (-self.price_state_variable_coeffs['coefficient'] - accrual) / (2 * self.reversion_speed**2)
        term2 = self.price_state_variable_coeffs['coefficient']**2 / (4 * self.reversion_speed)
        return (self.volatility**2 / accrual) * (term1 + term2)

    def yield_volatility(self, maturity_date: date | datetime) -> float:
        """
        Returns the volatility for a zero-coupon bond's yield for time T-maturity bonds.
        """
        accrual = DayCountCalculator.compute_accrual_length(self.start_date_time,
                                                            maturity_date,
                                                            self.day_count_convention)
        self._create_bond_price_coeffs(maturity_date)
        return -(self.price_state_variable_coeffs['coefficient'] / accrual) * self.volatility

    def instantaneous_forward_rate(self,
                                   maturity_date: date | datetime,
                                   purchase_date: Optional[date | datetime] = None) -> float:
        """
        Returns the instantaneous forward rate of the Vasicek Model for a given maturity date T.
        Given the affine price coefficients where P^T = exp ( A^T + B^T r_t), the instantaneous short rate
        is defined to be:
            -d/dT log P^T = -d/dT [ A^T + B^T r_t ]
        """
        if self.path is None:
            raise ValueError('Path cannot both be None when calculating the instantaneous forward rate.')

        if purchase_date is None:
            purchase_date = self.start_date_time

        short_rate = self(purchase_date)
        derivatives = self._calculate_bond_price_coeff_derivatives_wrt_maturity(purchase_date = purchase_date,
                                                                                maturity_date = maturity_date)
        forward_rate = -derivatives['intercept'] - derivatives['coefficient'] * short_rate
        return forward_rate

    def average_expected_short_rate(self,
                                    maturity_date: date | datetime,
                                    purchase_date: Optional[date | datetime] = None) -> float:
        """
        Calculates the average from the purchase date t to the maturity
        date T of the conditional short rate mean in the vasicek model, conditioned on information up to time t.
        That this, this method calculates:

           1/ (T-t) *  int_{t}^{T} E[ r_s | F_t] ds
        where E[r_s | F_t] is the conditional expectation of the short rate at time s on information know at t, for s >= t.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        self._create_bond_price_coeffs(maturity_date=maturity_date, purchase_date=purchase_date)
        accrual = DayCountCalculator.compute_accrual_length(purchase_date,
                                                            maturity_date,
                                                            self.day_count_convention)

        mean_times_accrual = -self.long_term_mean * (self.price_state_variable_coeffs['coefficient'] + accrual) \
               + self.price_state_variable_coeffs['coefficient'] * self(purchase_date)

        return -mean_times_accrual/accrual

    # Plotting

    def plot(self, show_fig: bool = False) -> None:
        """ Produces a plot of the model short rate path. """
        title_str = f'Vasicek Model Sample Path with Parameters\n' \
                    f'Mean {self.long_term_mean}; Volatility {self.volatility}; Reversion Speed {self.reversion_speed}'
        plt.figure(figsize=(15, 6))
        plt.title(title_str)
        date_range = Scheduler.generate_dates_by_increments(start_date=self.start_date_time,
                                                            end_date=self.end_date_time,
                                                            increment=self.dt,
                                                            max_dates=1_000_000)

        plt.plot(date_range, [self(datetime_obj) * 100 for datetime_obj in date_range], linewidth=0.5, alpha=1)
        plt.plot(date_range, [self.expected_short_rate(datetime_obj) * 100 for datetime_obj in date_range],
                 linestyle='dotted', color='tab:red')

        plt.axhline(self.long_term_mean * 100, linestyle="--", linewidth=0.75, color="grey")
        plt.ylabel('Short Rate (%)')
        plt.grid(alpha=0.25)
        plt.legend(['Sample Short Rate Path', 'Expected Short Rate', 'Reversion Level'], frameon=False)
        if show_fig:
            plt.show()


class MultivariateVasicekModel(ShortRateModel, AffineModelMixin):
    """
    A class to generate sample paths and yield curves

        d X_t = K(theta - X_t)dt + S dW_t
        and r_t = mu + <g, X_t>.

    where K is
    -------------------------------------------------------------
    Reference, Chapter 18 of Rebonato's *Bond Pricing and Yield Curve Modelling*, pages 299 -328.
    """

    def __init__(self,
                 short_rate_intercept: float,
                 short_rate_coefficients: np.array,
                 reversion_level: np.array,
                 reversion_matrix: np.ndarray,
                 volatility_matrix: np.ndarray,
                 brownian_motion: BrownianMotion,
                 dt: timedelta | relativedelta = relativedelta(hours=1)
                 ) -> None:

        assert brownian_motion.dimension == len(volatility_matrix)
        assert brownian_motion.dimension == len(reversion_level)
        assert brownian_motion.dimension == len(reversion_matrix)


        self.short_rate_intercept = short_rate_intercept
        self.short_rate_coefficients = short_rate_coefficients
        self.reversion_level = reversion_level
        self.reversion_matrix = reversion_matrix
        self.volatility_matrix = volatility_matrix

        self.reversion_matrix_eigenvalues, self.reversion_matrix_eigenvectors = np.linalg.eig(self.reversion_matrix)

        diffusion_process = DiffusionProcess(
            drift_diffusion_collection={
                f'state space variable {i}': DriftDiffusionPair(
                    drift=lambda t, X: self.reversion_matrix[i, :] @ (self.reversion_level - X),
                    diffusion=lambda t, X: self.volatility_matrix[i, :]) for i in range(brownian_motion.dimension)
            },
            brownian_motion=brownian_motion,
            dt=dt)

        super().__init__(
            short_rate_transformation=lambda state_variables: self.short_rate_intercept + self.short_rate_coefficients @ state_variables,
            state_variables_diffusion_process=diffusion_process
        )

    def zero_coupon_bond_yield(self, maturity_date: date, purchase_date: Optional[date | datetime] = None) -> float:
        """
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        self._create_bond_price_coeffs(maturity_date, purchase_date)
        state_variables = self.state_variables_diffusion_process(purchase_date)
        accrual = DayCountCalculator.compute_accrual_length(purchase_date, maturity_date, self.day_count_convention)
        yield_to_maturity = -self.price_state_variable_coeffs['intercept']/accrual - self.price_state_variable_coeffs['coefficients'] @ state_variables/accrual
        return yield_to_maturity

    def _create_bond_price_coeffs(self, maturity_date: date, purchase_date: Optional[date | datetime] = None) -> None:
        """
        Reference: Equation (18.136) of Rebonato's *Bond Pricing and Yield Curve Modeling*
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        accrual = DayCountCalculator.compute_accrual_length(purchase_date, maturity_date, self.day_count_convention)
        reversion_inv = self.reversion_matrix_eigenvectors @ \
                        np.diag(1 / self.reversion_matrix_eigenvalues) @ \
                        self.reversion_matrix_eigenvectors.T

        B = self._create_bond_price_coefficient_term(accrual, reversion_inv)
        A = self._create_bond_price_intercept_term(accrual, reversion_inv)
        self.price_state_variable_coeffs = {'intercept': A, 'coefficients': B}

    def _create_bond_price_coefficient_term(self, accrual: float, reversion_inv: np.ndarray) -> np.array:
        """
        Helper method to calculate the bond price coefficient term
        """
        p = self.brownian_motion.dimension
        negative_exp_eigen_times_accrual = np.exp( -self.reversion_matrix_eigenvalues * accrual)
        exp_reversion = self.reversion_matrix_eigenvectors @ \
                        np.diag(negative_exp_eigen_times_accrual) @ \
                        self.reversion_matrix_eigenvectors.T


        B = (exp_reversion - np.eye(p)) @ reversion_inv @ self.short_rate_coefficients
        return B

    def _create_bond_price_intercept_term(self, accrual: float, reversion_inv: np.ndarray) -> float:
        """
        Reference Equations (18.142) to (18.189).
        """
        p = self.brownian_motion.dimension
        reversion_times_coeff = self.reversion_matrix @ self.short_rate_coefficients
        reversion_inv_times_coeff = reversion_inv @ self.short_rate_coefficients
        Dt_mat = np.diag((1 - np.exp(-self.reversion_matrix_eigenvalues * accrual)) / self.reversion_matrix_eigenvalues)  # Looks good
        transformed_Dt = self.reversion_matrix_eigenvectors @ Dt_mat @ self.reversion_matrix_eigenvectors.T
        C_mat = self.volatility_matrix @ self.volatility_matrix.T                                                         # looks good
        M_mat = self.reversion_matrix_eigenvectors.T @ C_mat @ self.reversion_matrix_eigenvectors
        eigen_plus_eigen = np.ones((p, p)) * self.reversion_matrix_eigenvalues \
                           + (np.ones((p, p)) * self.reversion_matrix_eigenvalues.T).T

        F_mat = M_mat * (1- np.exp(-eigen_plus_eigen*accrual)) / eigen_plus_eigen
        transformed_F_mat = self.reversion_matrix_eigenvectors @ F_mat @ self.reversion_matrix_eigenvectors.T

        int_1 = -self.short_rate_intercept * accrual
        int_2 = self.short_rate_coefficients.T @ (transformed_Dt @ self.reversion_level - self.reversion_level * accrual)

        int_3_d = 0.5 * reversion_inv_times_coeff.T @ C_mat @ reversion_inv_times_coeff * accrual

        int_3_c = -0.5 * reversion_inv_times_coeff.T @ transformed_Dt @ C_mat @ reversion_inv_times_coeff

        int_3_b = -0.5 * reversion_inv_times_coeff.T @ C_mat @ self.reversion_matrix_eigenvectors @ Dt_mat @ \
                  np.diag(1/self.reversion_matrix_eigenvalues) @ self.reversion_matrix_eigenvectors.T @ \
                  self.short_rate_coefficients

        int_3_a = 0.5 * reversion_inv_times_coeff.T @ transformed_F_mat @ reversion_inv_times_coeff

        int_3 = int_3_a + int_3_b + int_3_c + int_3_d
        return int_1 + int_2 + int_3



if __name__ == '__main__':
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta
    from fixedIncome.src.scheduling_tools.scheduler import Scheduler

    # ----------------------------------------------------------------


    start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
    end_time = datetime(2053, 10, 15, 0, 0, 0, 0)

    brownian_motion = BrownianMotion(start_date_time=start_time,
                                     end_date_time=end_time,
                                     dimension=1)

    dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                   end_date=end_time,
                                                   increment=timedelta(1),
                                                   max_dates=1_000_000)

    vm = VasicekModel(reversion_level=0.04,
                      reversion_speed=0.5,
                      volatility=0.02,
                      brownian_motion=brownian_motion)

    admissible_dates = [date_obj for date_obj in dates if date_obj <= vm.end_date_time]

    def plot_sample_path(vm):
        path = vm.generate_path(starting_state_space_values=np.array([0.08]), set_path=True, seed=1)
        vm.plot()
        plt.savefig('../../../../../../fixedIncome/docs/images/Vasicek_Short_Rate.png')
        plt.show()

    # DISCOUNT CURVES
    def plot_discount_curves(vm, admissible_dates):
        NUM_CURVES = 20
        plt.figure(figsize=(13, 5))
        for seed in range(NUM_CURVES):
            vm.generate_path(starting_state_space_values=0.08, set_path=True, seed=seed)
            vm_df_curve = vm.discount_curve()
            discount_factors = [vm_df_curve(date_obj) for date_obj in admissible_dates]
            plt.plot(admissible_dates, discount_factors, color='tab:blue', alpha=1, linewidth=0.5)
            print(seed)

        plt.grid(alpha=0.25)
        plt.title(f'Discount Curves from Continuously-Compounding {NUM_CURVES} Vasicek Model Short Rate Paths\n'
                  f'with Model Parameters Mean {vm.reversion_level}; '
                  f'Volatility {vm.volatility}; Reversion Speed {vm.reversion_speed}')
        plt.ylabel('Discount Factor')
        plt.savefig('../../../../../../fixedIncome/docs/images/Vasicek_Discount_Curves.png')
        plt.show()

    def plot_yield_convexity():
        # CONVEXITY

        vm = VasicekModel(reversion_level=0.04,
                          reversion_speed=0.1,
                          volatility=0.02,
                          brownian_motion=brownian_motion)

        INITIAL_SHORT_RATE = 0.08
        vm.generate_path(starting_state_space_values=INITIAL_SHORT_RATE, set_path=True, seed=1)
        vm_avg_short_rate = np.zeros((1, len(admissible_dates)))
        vm_avg_df = np.zeros((1, len(admissible_dates)))
        vm_avg_integrated_path = np.zeros((1, len(admissible_dates)))

        fig, ax = plt.subplots(1, 1, figsize=(13, 5))
        accruals = [DayCountCalculator.compute_accrual_length(start_time, datetime_obj, DayCountConvention.ACTUAL_OVER_ACTUAL)
                    for datetime_obj in admissible_dates]

        affine_yields = [100 * vm.zero_coupon_yield(maturity_date=date) for date in admissible_dates[1:]]

        plt.plot(admissible_dates[1:], affine_yields, color='tab:blue')

        plt.plot(admissible_dates[1:],
                 [vm.average_expected_short_rate(date_obj) * 100 for date_obj in admissible_dates[1:]],
                 color='darkred')

        plt.grid(alpha=0.25)
        plt.title(f'Convexity in the Vasicek Model by Comparing Yields and the Expected Short Rate Average Over Time;'
                  f'\nModel Parameters: Mean {vm.long_term_mean}; Volatility {vm.volatility}; Reversion Speed {vm.reversion_speed}')
        plt.ylabel('Yield (%)')
        plt.legend([
            'Yield',
            'Average Expected Short Rate'],
                   loc='lower left', frameon=False)

        ax2 = ax.twinx()

        affine_diff = [vm.yield_convexity(datetime_obj) * 10_000 for datetime_obj in admissible_dates[1:]]
        plt.plot(admissible_dates[1:], affine_diff, color='grey', linestyle='dotted')

        plt.legend(['Difference (Right)'],
                   loc='upper right', frameon=False)

        ax2.set_ylabel('Convexity Adjustment (bp)')
        plt.savefig('../../../../../../fixedIncome/docs/images/Vasicek_Convexity.png')
        plt.show()


    #----------------------------------------------------------------
    # Yield Volatilities
    def plot_yield_volatilities():
        SHORT_RATE_VOLATILITY = 0.008  # 80 bps
        reversion_speeds = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]

        start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
        end_time = datetime(2073, 10, 15, 0, 0, 0, 0)
        brownian_motion = BrownianMotion(start_date_time=start_time,
                                         end_date_time=end_time,
                                         dimension=1)

        dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                       end_date=end_time,
                                                       increment=timedelta(1),
                                                       max_dates=1_000_000)

        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        for speed in reversion_speeds:
            vm = VasicekModel(reversion_level=0.04,
                              reversion_speed=speed,
                              volatility=SHORT_RATE_VOLATILITY,
                              brownian_motion=brownian_motion)

            admissible_dates = [date_obj for date_obj in dates if date_obj < vm.end_date_time]
            ax.plot(admissible_dates[1:], [vm.yield_volatility(date_obj) for date_obj in admissible_dates[1:]])

        ax.grid(alpha=0.25)
        plt.title('Yield Volatilities Across Reversion Speeds and Maturity Dates for the Vasicek Model')
        plt.xlabel('Maturity Date')
        plt.ylabel('Volatility')
        ax.legend([f'\u03BA = {speed:0.2f}' for speed in reversion_speeds],
                  title='Reversion Speed',
                  bbox_to_anchor=(1.025, 0.6), frameon=False)
        plt.tight_layout()
        plt.show()

    #----------------------------------------------------------------------
    # Yield convexities for different reversion speeds
    def plot_yield_convexities():
        SHORT_RATE_VOLATILITY = 0.008  # 80 bps
        INITIAL_SHORT_RATE = 0.05
        reversion_speeds = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]

        start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
        end_time = datetime(2073, 10, 15, 0, 0, 0, 0)

        brownian_motion = BrownianMotion(start_date_time=start_time,
                                         end_date_time=end_time,
                                         dimension=1)

        dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                       end_date=end_time,
                                                       increment=timedelta(1),
                                                       max_dates=1_000_000)
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        for speed in reversion_speeds:
            vm = VasicekModel(reversion_level=0.035,
                              reversion_speed=speed,
                              volatility=SHORT_RATE_VOLATILITY,
                              brownian_motion=brownian_motion)

            admissible_dates = [date_obj for date_obj in dates if date_obj < vm.end_date_time]
            ax.plot(admissible_dates[1:], [vm.yield_convexity(date_obj) * 10_000
                                           for date_obj in admissible_dates[1:]])

        ax.grid(alpha=0.25)
        plt.title('Yield Convexities Across Reversion Speeds and Maturity Dates for the Vasicek Model')
        plt.xlabel('Maturity Date')
        plt.ylabel('Convexity (bp)')
        ax.legend([f'\u03BA = {speed:0.2f}' for speed in reversion_speeds],
                  title='Reversion Speed',
                  bbox_to_anchor=(1.025, 0.6), frameon=False)
        plt.tight_layout()
        plt.show()


    #-----------------------------------------------------------------------
    # Yield curves for different reversion speeds

    def plot_yield_curves_across_reversion_speed():
        SHORT_RATE_VOLATILITY = 0.008  # 80 bps
        INITIAL_SHORT_RATE = 0.05
        reversion_speeds = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]

        start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
        end_time = datetime(2073, 10, 15, 0, 0, 0, 0)

        brownian_motion = BrownianMotion(start_date_time=start_time,
                                         end_date_time=end_time,
                                         dimension=1)

        dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                       end_date=end_time,
                                                       increment=timedelta(1),
                                                       max_dates=1_000_000)

        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        for speed in reversion_speeds:
            vm = VasicekModel(reversion_level=0.035,
                              reversion_speed=speed,
                              volatility=SHORT_RATE_VOLATILITY,
                              brownian_motion=brownian_motion)

            vm.generate_path(starting_state_space_values=INITIAL_SHORT_RATE, set_path=True, seed=1)

            admissible_dates = [date_obj for date_obj in dates if date_obj < vm.end_date_time]
            ax.plot(admissible_dates[1:], [vm.zero_coupon_yield(date_obj) * 100
                                           for date_obj in admissible_dates[1:]])

        ax.grid(alpha=0.25)
        plt.title(f'Yields Across Reversion Speeds and Maturity Dates for the Vasicek Model'
                  f'\nModel Parameters: Mean {vm.long_term_mean}; Volatility {vm.volatility}; Reversion Speed Varies')
        plt.xlabel('Maturity Date')
        plt.ylabel('Yield (%)')
        ax.legend([f'\u03BA = {speed:0.2f}' for speed in reversion_speeds],
                  title='Reversion Speed',
                  bbox_to_anchor=(1.2, 0.65), frameon=False)
        plt.tight_layout()
        plt.show()

    # Instantaneous forward rates
    def plot_instantaneous_forward_rates():
        fig, ax = plt.subplots(1, 1, figsize=(13, 5))

        SHORT_RATE_VOLATILITY = 0.008  # 80 bps
        INITIAL_SHORT_RATE = 0.05
        reversion_speed = 0.2
        start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
        end_time = datetime(2073, 10, 15, 0, 0, 0, 0)

        brownian_motion = BrownianMotion(start_date_time=start_time,
                                         end_date_time=end_time,
                                         dimension=1)

        years = [1, 5, 10, 20, 30]

        dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                       end_date=end_time,
                                                       increment=timedelta(1),
                                                       max_dates=1_000_000)

        vm = VasicekModel(reversion_level=0.035,
                          reversion_speed=reversion_speed,
                          volatility=SHORT_RATE_VOLATILITY,
                          brownian_motion=brownian_motion)

        for year in years:
            vm.generate_path(starting_state_space_values=INITIAL_SHORT_RATE, set_path=True, seed=year)
            admissible_dates = [date_obj for date_obj in dates if date_obj < vm.end_date_time]
            ax.plot(admissible_dates[1:],
                    [vm.instantaneous_forward_rate(maturity_date=date_obj + relativedelta(years=year),
                                                   purchase_date=date_obj) * 100
                     for date_obj in admissible_dates[1:]],
                    linewidth=0.85)

        plt.axhline((vm.long_term_mean + vm.convexity_limit) * 100, linestyle="--", linewidth=0.75, color="grey")

        ax.legend([f'{years} Years' for years in years],
                  title='Forward Rate',
                  bbox_to_anchor=(1.0, 0.6), frameon=False)

        plt.grid(alpha=0.25)
        plt.title(f'Instantaneous Forward Rate Process Sample Paths in the Vasicek Model\n'
                   f'Model Parameters: Mean {vm.long_term_mean}; Volatility {vm.volatility}; Reversion Speed {reversion_speed}')
        plt.tight_layout()
        plt.savefig('../../../../../../fixedIncome/docs/images/Vasicek_Instantaneous_Forward_Rate_Processes.png')
        plt.show()


    #---------------------------------------------------------------------
    import math

    short_rate_intercept = 0.035
    short_rate_coefficients = np.array([0.02, 0.001])
    reversion_level = np.array([0.03, 0.05])

    rho = 0.0
    stand_dev_1 = 1
    stand_dev_2 = 1
    volatility_matrix = np.array([[stand_dev_1 ** 2, stand_dev_1 * stand_dev_2 * rho],
                                   [stand_dev_1 * stand_dev_2 * rho, stand_dev_2 ** 2]])

    brownian_motion = BrownianMotion(start_date_time=start_time,
                                     end_date_time=end_time,
                                     dimension=2)


    reversion_directions = np.array([[1.0/math.sqrt(2), 1.0/math.sqrt(2)],
                                     [1.0/math.sqrt(2), -1.0/math.sqrt(2)]])
    reversion_matrix = reversion_directions @ np.array([[0.02, 0.05], [0.05, 0.1]]) @ reversion_directions.T

    mvm = MultivariateVasicekModel(
        short_rate_intercept=short_rate_intercept,
        short_rate_coefficients=short_rate_coefficients,
        reversion_level=reversion_level,
        reversion_matrix=reversion_matrix,
        volatility_matrix=volatility_matrix,
        brownian_motion=brownian_motion)

    starting_state_variables = np.array([0.03, 0.075])
    mvm.generate_path(starting_state_variables, set_path=True, seed=1)

    admissible_dates = [date_obj for date_obj in dates if date_obj <= vm.end_date_time]
    plt.figure(figsize=(13, 5))
    plt.plot(admissible_dates[1:], [mvm(datetime_obj) for datetime_obj in admissible_dates[1:]], linewidth=0.3)
    plt.show()

    plt.figure(figsize=(13, 5))
    plt.plot(admissible_dates[1:], [mvm.zero_coupon_bond_yield(maturity_date=datetime_obj) for datetime_obj in admissible_dates[1:]])
    plt.show()