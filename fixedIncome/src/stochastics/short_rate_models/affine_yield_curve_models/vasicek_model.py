"""
This script contains the Vasicek Model for the short rate.
Reference, *Fixed Income Securities, 4th Ed.* by Tuckman and Serrat, page 205.

Unit tests are contained in
fixedIncome.tests.test_stochastics.test_short_rate_models.test_affine_yield_curve_models.test_vasicek_model.py
"""
import itertools
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Iterable, Callable
from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.base_processes import DiffusionProcess
from fixedIncome.src.stochastics.short_rate_models.base_short_rate_model import ShortRateModel
from fixedIncome.src.stochastics.base_processes import DriftDiffusionPair
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.affine_model_mixin import AffineModelMixin
from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator

def create_affine_function(short_rate_coefficients, short_rate_intercept) -> Callable[[np.array], float]:
    """
    Returns the affine function intercept + <coeff, state variables>
    """
    return lambda state_variables: short_rate_intercept + short_rate_coefficients @ state_variables


def create_constant_drift_diffusion_pair(reversion_level, reversion_speed, diffusion_term):
    """
    Alleviates the closure issue with anaonymous functions.
    """
    return DriftDiffusionPair(
        drift=lambda t, state_variables: reversion_speed @ (reversion_level - state_variables),
        diffusion=lambda time, state_variables: diffusion_term
    )


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


    def short_rate_variance(self,
                            maturity_date: date | datetime,
                            purchase_date: Optional[date | datetime] = None) -> float:
        """
        Returns the conditional variance of the Vasicek Model, equal to sigma**2 / 2*a * (1- exp(-2 k * (t-t_0))).
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        accrual = DayCountCalculator.compute_accrual_length(purchase_date,
                                                            maturity_date,
                                                            self.day_count_convention)

        shrinkage_factor = 1 - math.exp(-2 * self.reversion_speed * accrual)
        return self.volatility**2 / (2 * self.reversion_speed) * shrinkage_factor

    def expected_short_rate(self,
                            maturity_date: date | datetime,
                            purchase_date: Optional[date | datetime] = None) -> float:
        """
        Returns the conditional mean of the Vasicek Model short rate at a given maturity date T given
        the reference purchase date t_0.
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

    def _calculate_bond_price_coeff_derivatives_wrt_maturity(
            self, maturity_date: date | datetime, purchase_date: Optional[date | datetime] = None
    ) -> dict[str, float]:
        """
        Returns a dictionary with the same keys ('intercept' and 'coefficient') as
        the dictionary formed by self._create_bond_price_coeffs, but instead returns each bond price
        coefficient's derivative with respect to maturity time T.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

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

    def price_convexities(self,
                          maturity_dates: Iterable[datetime|date],
                          purchase_date: Optional[datetime] = None) -> np.array:
        """

        """
        maturity_dates = list(maturity_dates)
        if purchase_date is None:
            purchase_date = self.start_date_time

        if min(maturity_dates) <= purchase_date:
            raise ValueError(f'Minimum maturity datetime {min(maturity_dates)} can not be less than or equal to the purchase datetime {purchase_date}.')

        accruals = np.array([DayCountCalculator.compute_accrual_length(purchase_date, maturity_date, self.day_count_convention)
                            for maturity_date in maturity_dates])

        bond_prices = np.array([self.zero_coupon_bond_price(maturity_date=maturity_date, purchase_date=purchase_date)
                                for maturity_date in maturity_dates])

        volatilities = np.array([self.yield_volatility(maturity_date=maturity_date)
                        for maturity_date in maturity_dates])

        return -0.5 * accruals**2 * bond_prices * volatilities**2



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
        derivatives = self._calculate_bond_price_coeff_derivatives_wrt_maturity(purchase_date=purchase_date,
                                                                                maturity_date=maturity_date)
        forward_rate = -derivatives['intercept'] - derivatives['coefficient'] * short_rate
        return forward_rate

    def instantaneous_forward_rate_volatility(self,
                                              maturity_date: date | datetime,
                                              purchase_date: Optional[date | datetime] = None) -> float:
        """
        Calculates the volatility of the instantaneous forward rate at a maturity date given a reference purchase date.
        Reference Equation (16.68) in Rebonato *Bond Pricing and Yield Curve Modeling*, page 274.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        accrual = DayCountCalculator.compute_accrual_length(purchase_date,
                                                            maturity_date,
                                                            self.day_count_convention)

        return self.volatility * math.exp(-accrual * self.reversion_speed)


    def average_expected_short_rate(self,
                                    maturity_date: date | datetime,
                                    purchase_date: Optional[date | datetime] = None) -> float:
        """
        Calculates the average across time from the purchase date t to the maturity
        date T of the conditional short rate mean in the vasicek model, conditioned on information up to time t.
        That this, this method calculates:

           1/(T-t) *  int_{t}^{T} E[ r_s | F_t] ds
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

    where K is a diagonalizable reversion matrix, theta is the reversion level vector, X_t are the state variable
    process, and S is a volatility matrix.
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
        self.reversion_matrix_eigenvectors_inv = np.linalg.inv(self.reversion_matrix_eigenvectors)
        self.C_mat = self.volatility_matrix @ self.volatility_matrix.T

        drift_diffusion_collection = {}
        for i in range(brownian_motion.dimension):
            drift_diffusion_collection[f'state_variable_{i}'] = create_constant_drift_diffusion_pair(
                reversion_level=self.reversion_level,
                reversion_speed=self.reversion_matrix[i, :],
                diffusion_term=self.volatility_matrix[i, :]
            )

        diffusion_process = DiffusionProcess(
            drift_diffusion_collection=drift_diffusion_collection,
            brownian_motion=brownian_motion,
            dt=dt)

        super().__init__(
            short_rate_transformation=create_affine_function(short_rate_coefficients=short_rate_coefficients,
                                                             short_rate_intercept=short_rate_intercept),
            state_variables_diffusion_process=diffusion_process
        )

    def expected_state_variable_mean(self,
                                     maturity_date: date | datetime,
                                     purchase_date: Optional[date | datetime] = None) -> np.array:
        """
        Returns the conditional mean of the MultivariateVasicekModel state variables at a given maturity date T given
        the reference purchase date t_0.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        accrual = DayCountCalculator.compute_accrual_length(purchase_date,
                                                            maturity_date,
                                                            self.day_count_convention)
        mat_exponential = self.reversion_matrix_eigenvectors \
                          @ np.diag(np.exp(-self.reversion_matrix_eigenvalues * accrual)) \
                          @ self.reversion_matrix_eigenvectors_inv

        return mat_exponential @ self.state_variables_diffusion_process(purchase_date) \
               + (np.eye(self.dimension) - mat_exponential) @ self.reversion_level
    def expected_short_rate(self,
                            maturity_date: date | datetime,
                            purchase_date: Optional[date | datetime] = None) -> float:
        """
        Returns the conditional mean of the Multivariate Vasicek Model short rate at a given maturity date T given
        the reference purchase date t_0.
        """
        expected_state_variables = self.expected_state_variable_mean(maturity_date=maturity_date,
                                                                     purchase_date=purchase_date)

        return self.short_rate_transformation(expected_state_variables)

    def average_expected_short_rate(self,
                                    maturity_date: date | datetime,
                                    purchase_date: Optional[date | datetime] = None) -> float:
        """
        Calculates the average across time from the purchase date t to the maturity
        date T of the conditional short rate mean in the vasicek model, conditioned on information up to time t.
        That this, this method calculates:

           1/(T-t) *  int_{t}^{T} E[ r_s | F_t] ds
        where E[r_s | F_t] is the conditional expectation of the short rate at time s on information know at t, for s >= t.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        if maturity_date <= purchase_date:
            raise ValueError(f'Maturity datetime {maturity_date} can not be less than or equal to the purchase datetime {purchase_date}.')

        accrual = DayCountCalculator.compute_accrual_length(purchase_date, maturity_date, self.day_count_convention)
        diagonal_term = (1 - np.exp( -self.reversion_matrix_eigenvalues * accrual)) / (self.reversion_matrix_eigenvalues * accrual)
        avg_matrix_exponential = self.reversion_matrix_eigenvectors @ np.diag(diagonal_term) @ self.reversion_matrix_eigenvectors_inv

        term_1 = avg_matrix_exponential @ self.state_variables_diffusion_process(purchase_date)
        term_2 = (self.reversion_level - avg_matrix_exponential @ self.reversion_level)  # (I - exp) theta
        return self.short_rate_transformation(term_1 + term_2)

    def state_variable_covariance(self,
                                  maturity_date: date | datetime,
                                  purchase_date: Optional[date | datetime] = None) -> np.ndarray:
        """
        Calculates the covariance matrix of the state variables at a given maturity date conditioned
        on the values at the purchase date.
        Reference, pages 308 - 309 of Rebonato *Bond Pricing and Yield Curve Modeling*.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        accrual = DayCountCalculator.compute_accrual_length(purchase_date, maturity_date, self.day_count_convention)
        broadcast_mat = np.ones((self.dimension, self.dimension)) * self.reversion_matrix_eigenvalues
        sum_eigenvalues = broadcast_mat + broadcast_mat.T
        one_minus_exp_eigen_sum = 1 - np.exp(-sum_eigenvalues * accrual)
        H_mat = self.reversion_matrix_eigenvectors_inv @ self.C_mat @ self.reversion_matrix_eigenvectors
        covar = one_minus_exp_eigen_sum * H_mat / sum_eigenvalues
        return covar


    def short_rate_variance(self,
                            maturity_date: date | datetime,
                            purchase_date: Optional[date | datetime] = None) -> float:
        """
        Calculates the variance of the short rate at a given maturity date conditioned
        on the values of the state variances at the purchase date.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        state_variable_covar = self.state_variable_covariance(maturity_date, purchase_date)
        return self.short_rate_coefficients.T @ state_variable_covar @ self.short_rate_coefficients


    def zero_coupon_bond_price(self,
                               maturity_date: date | datetime,
                               purchase_date: Optional[date | datetime] = None) -> float:
        """
        Calculates the price of a zero coupon bond.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        self._create_bond_price_coeffs(maturity_date, purchase_date)
        state_variables = self.state_variables_diffusion_process(purchase_date)
        affine_price_expression = self.price_state_variable_coeffs['intercept'] + self.price_state_variable_coeffs['coefficients'] @ state_variables
        return math.exp(affine_price_expression)

    def zero_coupon_bond_price_deriv_wrt_state_variables(self,
                                                         maturity_date: date | datetime,
                                                         purchase_date: Optional[date | datetime] = None) -> np.array:
        """
        Computes the p-dimensional vector derivative dP / dx,
        where x = (x_1, ..., x_p) are the state variables at the purchase_date.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        bond_price = self.zero_coupon_bond_price(maturity_date=maturity_date, purchase_date=purchase_date)
        return bond_price * self.price_state_variable_coeffs['coefficients']

    def weights_for_riskless_portfolio(self,
                                       maturity_dates: Iterable[date|datetime],
                                       purchase_date: Optional[date|datetime] = None) -> np.array:
        """
        Get the weights (w_1, ..., w_{p+1}) for the p+1 zero-coupon bonds of maturities in the provided iterable such
        that
            sum_{i=1}^{p} w_i dP^{T_i} / dx = 0 and sum_{i=1}^{p+1} w_i = 1.0,
        where dP^{T_i} / dx is the vector derivative of zero-coupon bond with maturity T_i with respect to the state
        variables at time purchase_date.

        See Section 20.8 of Rebonato *Bond Pricing and Yield Curve Modeling*
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        maturity_dates = list(maturity_dates)
        assert len(maturity_dates) == (self.dimension+1)

        matirx_of_deriv_differences = np.zeros((self.dimension, self.dimension))
        deriv_for_last_bond = self.zero_coupon_bond_price_deriv_wrt_state_variables(purchase_date=purchase_date,
                                                                                    maturity_date=maturity_dates[self.dimension])

        for dimension in range(self.dimension):  # all but the last bond

            deriv_for_bond = self.zero_coupon_bond_price_deriv_wrt_state_variables(purchase_date=purchase_date,
                                                                                   maturity_date=maturity_dates[dimension])
            matirx_of_deriv_differences[:, dimension] = deriv_for_bond - deriv_for_last_bond

        weights = np.linalg.solve(matirx_of_deriv_differences, -deriv_for_last_bond)
        one_minus_sum_of_weights = np.array([1.0 - sum(weights)])
        return np.concatenate([weights, one_minus_sum_of_weights])

    def riskless_portfolio_convexity(self,
                                     maturity_dates: Iterable[date|datetime],
                                     purchase_date: Optional[date|datetime] = None) -> float:
        """
        Returns the convexity of the riskless portfolio of zero-coupon bonds of the provided maturity_dates.

        See equation (21.27) in Rebonato *Bond Pricing and Yield Curve Modeling*.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        maturity_dates = list(maturity_dates)
        assert len(maturity_dates) == (self.dimension+1)

        weights = self.weights_for_riskless_portfolio(maturity_dates=maturity_dates,
                                                      purchase_date=purchase_date)

        D_mat = np.zeros(shape=(self.dimension, self.dimension))

        for weight, maturity_date in zip(weights, maturity_dates):
            bond_price = self.zero_coupon_bond_price(maturity_date=maturity_date, purchase_date=purchase_date)
            price_coefficients = np.expand_dims(self.price_state_variable_coeffs['coefficients'], axis=1)
            B_t_times_B_t_transpose = price_coefficients @ price_coefficients.T
            D_mat += bond_price * weight * B_t_times_B_t_transpose

        return 0.5 * sum(np.diag(self.volatility_matrix.T @ D_mat @ self.volatility_matrix))

    def zero_coupon_bond_yield(self, maturity_date: date, purchase_date: Optional[date | datetime] = None) -> float:
        """
        Calculates the zero coupon bond yield of
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        if maturity_date <= purchase_date:
            raise ValueError(f'Maturity datetime {maturity_date} can not be less than or equal to the purchase datetime {purchase_date}.')

        state_variables = self.state_variables_diffusion_process(purchase_date)
        self._create_bond_yield_coeffs(maturity_date, purchase_date)
        yield_to_maturity = self.yield_state_variable_coeffs['intercept'] + self.yield_state_variable_coeffs['coefficients'] @ state_variables
        return yield_to_maturity


    def _create_bond_yield_coeffs(self, maturity_date: date, purchase_date: Optional[date | datetime] = None) -> None:
        """
        Helper method to create the affine intercept alpha and coefficients beta such that the time-T yield
        y_t^T = alpha + <beta, x_t>
        where x_t are the time-t state variables. The expressions can be deduced from
        e^{- (T-t) y_{t}^{T}} = e^{A_{t} + <B_t, x_t>} where A_t and B_t are the time-t affine intercept and coefficinets,
        respectively, which give the bond price.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        self._create_bond_price_coeffs(maturity_date, purchase_date)
        accrual = DayCountCalculator.compute_accrual_length(purchase_date, maturity_date, self.day_count_convention)
        self.yield_state_variable_coeffs = dict()
        self.yield_state_variable_coeffs['intercept'] = -self.price_state_variable_coeffs['intercept']/accrual
        self.yield_state_variable_coeffs['coefficients'] = -self.price_state_variable_coeffs['coefficients']/accrual


    def _create_bond_price_coeffs(self, maturity_date: date, purchase_date: Optional[date | datetime] = None) -> None:
        """
        Reference: Equation (18.136) of Rebonato's *Bond Pricing and Yield Curve Modeling*
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        accrual = DayCountCalculator.compute_accrual_length(purchase_date, maturity_date, self.day_count_convention)

        reversion_inv = self.reversion_matrix_eigenvectors @ \
                        np.diag(1 / self.reversion_matrix_eigenvalues) @ \
                        self.reversion_matrix_eigenvectors_inv

        B = self._create_bond_price_coefficient_term(accrual, reversion_inv)
        A = self._create_bond_price_intercept_term(accrual, reversion_inv)
        self.price_state_variable_coeffs = {'intercept': A, 'coefficients': B}

    def _create_bond_price_coefficient_term(self, accrual: float, reversion_inv: np.ndarray) -> np.array:
        """
        Helper method to calculate the bond price coefficient term. From Equation (18.141)
        """
        p = self.brownian_motion.dimension
        exp_negative_eigen_times_accrual = np.exp(-self.reversion_matrix_eigenvalues * accrual)
        exp_reversion = self.reversion_matrix_eigenvectors @ \
                        np.diag(exp_negative_eigen_times_accrual) @ \
                        self.reversion_matrix_eigenvectors_inv


        B = (exp_reversion.T - np.eye(p)) @ reversion_inv.T @ self.short_rate_coefficients
        return B

    def _create_bond_price_intercept_term(self, accrual: float, reversion_inv: np.ndarray) -> float:
        """
        Reference Equations (18.142) to (18.189).
        """
        p = self.brownian_motion.dimension
        reversion_transpose_inv_times_coeff = reversion_inv.T @ self.short_rate_coefficients
        Dt_mat = np.diag((1 - np.exp(-self.reversion_matrix_eigenvalues * accrual)) / self.reversion_matrix_eigenvalues)
        transformed_Dt = self.reversion_matrix_eigenvectors @ Dt_mat @ self.reversion_matrix_eigenvectors_inv
        M_mat = self.reversion_matrix_eigenvectors_inv @ self.C_mat @ self.reversion_matrix_eigenvectors_inv.T  # Equation (18.186)
        eigen_plus_eigen = np.ones((p, p)) * self.reversion_matrix_eigenvalues \
                           + (np.ones((p, p)) * self.reversion_matrix_eigenvalues.T).T

        F_mat = M_mat * (1 - np.exp(-eigen_plus_eigen*accrual)) / eigen_plus_eigen  # Equation (18.188)

        # Equation (18.150)
        int_1 = -self.short_rate_intercept * accrual

        # Equation (18.167)
        int_2 = self.short_rate_coefficients.T @ (transformed_Dt @ self.reversion_level - self.reversion_level * accrual)

        # Equation (18.173)
        int_3_d = 0.5 * reversion_transpose_inv_times_coeff.T @ self.C_mat @ reversion_transpose_inv_times_coeff * accrual

        # Equation (18.178)
        int_3_c = -0.5 * reversion_transpose_inv_times_coeff.T @ transformed_Dt @ self.C_mat @ reversion_transpose_inv_times_coeff

        # Equation (18.183)
        int_3_b = -0.5 * reversion_transpose_inv_times_coeff.T @ self.C_mat @ (self.reversion_matrix_eigenvectors_inv.T) @ Dt_mat @ \
                  np.diag(1/self.reversion_matrix_eigenvalues) @ self.reversion_matrix_eigenvectors.T @ \
                  self.short_rate_coefficients

        # Equations (18.186) - (18.188)
        int_3_a = 0.5 * reversion_transpose_inv_times_coeff.T @ self.reversion_matrix_eigenvectors @ F_mat @ \
                  np.diag(1/self.reversion_matrix_eigenvalues) @ self.reversion_matrix_eigenvectors.T @ self.short_rate_coefficients

        int_3 = int_3_a + int_3_b + int_3_c + int_3_d
        return int_1 + int_2 + int_3

    def price_convexities(self, maturity_dates: Iterable[datetime|date], purchase_date: Optional[datetime] = None
                          ) -> np.array:
        """
        Calculates the convexity of the constant-maturity prices of zero-coupon bonds for each of the provided maturities.

        See Equation (20.51) of Rebonato *Bond Pricing and Yield Curve Modeling*
        """
        maturity_dates = list(maturity_dates)
        if purchase_date is None:
            purchase_date = self.start_date_time

        if min(maturity_dates) <= purchase_date:
            raise ValueError(f'Minimum maturity datetime {min(maturity_dates)} can not be less than or equal to the purchase datetime {purchase_date}.')

        cov_matrix = self.yield_covariance_matrix( maturity_dates=maturity_dates, purchase_date=purchase_date)
        accruals = np.array([DayCountCalculator.compute_accrual_length(purchase_date, maturity_date, self.day_count_convention)
                            for maturity_date in maturity_dates])

        bond_prices = np.array([self.zero_coupon_bond_price(maturity_date=maturity_date, purchase_date=purchase_date)
                            for maturity_date in maturity_dates])

        return -0.5 * bond_prices * accruals**2 * np.diag(cov_matrix)

    def yield_convexity(self,  maturity_date: datetime | date, purchase_date: Optional[datetime] = None) -> float:
        """
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        if maturity_date <= purchase_date:
            raise ValueError(f'Maturity datetime {maturity_date} can not be less than or equal to the purchase datetime {purchase_date}.')

        yield_to_maturity = self.zero_coupon_bond_yield(maturity_date=maturity_date, purchase_date=purchase_date)
        avg_expected_short_rate = self.average_expected_short_rate(maturity_date=maturity_date, purchase_date=purchase_date)
        return yield_to_maturity - avg_expected_short_rate

    def yield_covariance_matrix(
            self, maturity_dates: Iterable[datetime | date], purchase_date: Optional[datetime] = None
    ) -> np.ndarray:
        """
        Returns the matrix of yield covariances, where entry Cov[i, j] is the
        covariance between yields y_{t}^{T_i} and y_{t}^{T_j} of zero-coupon bonds.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        if min(maturity_dates) <= purchase_date:
            raise ValueError(f'Minimum maturity datetime {min(maturity_dates)} can not be less than or equal to the purchase datetime {purchase_date}.')

        maturity_dates = list(maturity_dates)
        beta_mat = np.empty((self.dimension, len(maturity_dates)))
        for index, maturity in enumerate(maturity_dates):
            self._create_bond_yield_coeffs(maturity_date=maturity, purchase_date=purchase_date)
            beta_mat[:, index] = self.yield_state_variable_coeffs['coefficients']

        return beta_mat.T @ self.C_mat @ beta_mat

    def yield_volatility(self, maturity_date: datetime|date, purchase_date: Optional[date | datetime] = None) -> float:
        """
        Returns the volatility of the time-t yield for a zero-coupon bond maturing at the specified maturity datetime.
        the volatility is the square root of the variance for the yield.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        if maturity_date <= purchase_date:
            raise ValueError(f'Maturity datetime {maturity_date} can not be less than or equal to the purchase datetime {purchase_date}.')

        covar_mat = self.yield_covariance_matrix(maturity_dates=[maturity_date], purchase_date=purchase_date)
        return math.sqrt(float(covar_mat))

    def instantaneous_forward_rate(self, maturity_date: date, purchase_date: Optional[date | datetime] = None) -> float:
        """
        Calculates the instantaneous forward rate at a maturity date given a reference purchase date.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        derivatives = self._calculate_bond_price_coeff_derivatives_wrt_maturity(maturity_date=maturity_date,
                                                                                purchase_date=purchase_date)

        state_variables = self.state_variables_diffusion_process(purchase_date)
        forward_rate = -derivatives['intercept'] - derivatives['coefficients'] @ state_variables
        return forward_rate

    def instantaneous_forward_rate_covariance(self,
                                              maturity_dates: Iterable[date | datetime],
                                              purchase_date: Optional[date | datetime] = None) -> np.ndarray:
        """
        Returns the covariance matrix of the instantaneous forward rates of various maturity dates given the
        baseline purchase date.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        if min(maturity_dates) <= purchase_date:
            raise ValueError(
                f'Minimum maturity datetime {min(maturity_dates)} can not be less than or equal to the purchase datetime {purchase_date}.')

        maturity_dates = list(maturity_dates)
        B_mat = np.empty((self.dimension, len(maturity_dates)))
        for index, maturity in enumerate(maturity_dates):
            self._create_bond_yield_coeffs(maturity_date=maturity, purchase_date=purchase_date)
            B_mat[:, index] = -self._calculate_bond_price_coeff_derivatives_wrt_maturity(maturity_date=maturity,
                                                                                         purchase_date=purchase_date)['coefficients']
        return B_mat.T @ self.C_mat @ B_mat


    def instantaneous_forward_rate_volatility(self,
                                              maturity_date: date | datetime,
                                              purchase_date: Optional[date | datetime] = None) -> float:
        """
        Returns the volatility of the instantaneous forward rate at a maturity date given the
        baseline purchase date. The volatility is the square root of the variance.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        covar_mat = self.instantaneous_forward_rate_covariance(maturity_dates=[maturity_date],
                                                               purchase_date=purchase_date)

        return math.sqrt(float(covar_mat))

    def zero_coupon_price_deriv_wrt_maturity(self,
                                             maturity_date: date|datetime,
                                             purchase_date: Optional[date|datetime] = None) -> float:
        """
        Calculates the derivative d P^T_t / dT of the price of a zero-coupon bond with maturity T
        with respect to the maturity T.

        By time-homogeneity of the price of a zero-coupon bond in an affine model, this is equal
        to the negative of the derivative with respect to the purchase date.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        coeff_derivatives = self._calculate_bond_price_coeff_derivatives_wrt_maturity(maturity_date=maturity_date,
                                                                                      purchase_date=purchase_date)

        bond_price = self.zero_coupon_bond_price(maturity_date=maturity_date, purchase_date=purchase_date)

        state_variables = self.state_variables_diffusion_process(purchase_date)

        return bond_price * (coeff_derivatives['intercept'] + coeff_derivatives['coefficients'] @ state_variables)


    def _calculate_bond_price_coeff_derivatives_wrt_maturity(self, maturity_date: date|datetime, purchase_date: Optional[date|datetime] = None) -> dict[str, np.array]:
        """
        Helper function to calculate the derivatives with respect to maturity time T of the affine price
        intercept and coefficient terms A_t^{T} and B_{t}^{T}, where P_{t}^{T} = e^{ A_t^T + <B_t^T, x_t>}.
        """
        if purchase_date is None:
            purchase_date = self.start_date_time

        self._create_bond_price_coeffs(maturity_date, purchase_date)
        accrual = DayCountCalculator.compute_accrual_length(purchase_date,
                                                            maturity_date,
                                                            self.day_count_convention)

        derivatives = {
            'coefficients': self._calculate_coeff_deriv_wrt_maturity(accrual),
            'intercept': self._calculate_intercept_deriv_wrt_maturity(accrual)
        }
        return derivatives

    def _calculate_coeff_deriv_wrt_maturity(self, accrual: float) -> np.array:
        """
        Helper function to calculate the derivative with respect to maturity time T of the affine price
        coefficient term B_{t}^{T}, where P_{t}^{T} = e^{ A_t^T + <B_t^T, x_t>}.
        """
        exp_negative_eigen_times_accrual = np.exp(-self.reversion_matrix_eigenvalues * accrual)
        exp_reversion = self.reversion_matrix_eigenvectors @ \
                        np.diag(exp_negative_eigen_times_accrual) @ \
                        self.reversion_matrix_eigenvectors_inv

        return -exp_reversion @ self.short_rate_coefficients

    def _calculate_intercept_deriv_wrt_maturity(self, accrual: float) -> np.array:
        """
        Helper function to calculate the derivative with respect to maturity time T of the affine price
        intercept term A_{t}^{T}, where P_{t}^{T} = e^{ A_t^T + <B_t^T, x_t>}.
        """
        int_1_deriv = -self.short_rate_intercept
        int_2_deriv = self.price_state_variable_coeffs['coefficients'].T @ self.reversion_matrix @ self.reversion_level
        int_3_deriv = 0.5 * self.price_state_variable_coeffs['coefficients'].T @ self.C_mat @ self.price_state_variable_coeffs['coefficients']
        return int_1_deriv + int_2_deriv + int_3_deriv




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
                                                   increment=relativedelta(days=1),
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
        brownian_motion=brownian_motion,
        dt=timedelta(1)
    )

    starting_state_variables = np.array([0.03, 0.075])
    mvm.generate_path(starting_state_variables, set_path=True, seed=1)


    #--------------------------------------------
    # construct risk-less portfolio of zero-coupon bonds

    dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                   end_date=end_time,
                                                   increment=timedelta(1),
                                                   max_dates=1_000_000)

    short_rate_from_portfolio_value_change = []
    approximate_short_rates = []
    present_values = []
    for date_obj, next_date in itertools.pairwise(dates):
        accrual = DayCountCalculator.compute_accrual_length(date_obj, next_date, mvm.day_count_convention)

        maturity_dates = [date_obj + relativedelta(years=1),
                          date_obj + relativedelta(years=5),
                          date_obj + relativedelta(years=10),
                          ]  # must have 3 bonds for 2-dimensional state variable process

        # Construct approximation from convexity + drift (21.27)
        riskless_weights = mvm.weights_for_riskless_portfolio(maturity_dates=maturity_dates, purchase_date=date_obj)
        bond_prices = np.array([mvm.zero_coupon_bond_price(maturity_date=maturity_date, purchase_date=date_obj)
                                for maturity_date in maturity_dates])
        modified_weights = riskless_weights * bond_prices

        price_derivatives = -np.array([mvm.zero_coupon_price_deriv_wrt_maturity(maturity_date=maturity_date, purchase_date=date_obj)
                                      for maturity_date in maturity_dates])  # negative due to time-homogeneity

        convexity = mvm.riskless_portfolio_convexity(maturity_dates=maturity_dates, purchase_date=date_obj)

        reconstructed_short_rate = (convexity + sum(price_derivatives * modified_weights / bond_prices)) / sum(modified_weights)
        approximate_short_rates.append(reconstructed_short_rate)

        # construct short rate approximation via riskless portfolio growth
        new_bond_prices = np.array([mvm.zero_coupon_bond_price(maturity_date=maturity_date, purchase_date=next_date)
                                for maturity_date in maturity_dates])

        change_in_portfolio_value = sum(new_bond_prices * riskless_weights) - sum(modified_weights)
        short_rate_approx = change_in_portfolio_value/(sum(modified_weights) * accrual)
        short_rate_from_portfolio_value_change.append(short_rate_approx)

    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    plt.plot(dates[:-1], [mvm(date_obj)*100 for date_obj in dates[:-1]], color='black', linewidth=0.5)
    plt.plot(dates[:-1], [rate * 100 for rate in approximate_short_rates], color='lightgrey', alpha=0.75, linewidth=2)
    plt.plot(dates[:-1], [rate * 100 for rate in short_rate_from_portfolio_value_change], alpha=0.5, color='darkred', linestyle='dashed', linewidth=1.0)
    plt.grid(alpha=0.25)
    plt.ylabel('Rate (%)')
    plt.xlabel('Date')
    plt.legend(['Short Rate',
                'Approx. Short Rate from Riskless\nPortfolio Drift and Convexity',
                'Approx. Short Rate from\nRiskless Portfolio Growth'],
               frameon=False,
               bbox_to_anchor=(1.02, 0.65)
               )
    plt.title('No-Arbitrage in the Multivariate Mean Reverting Model\nModel Short Rate and Estimated Short Rates from Constructing Riskless Portfolios')
    fig.tight_layout()
    plt.savefig('../../../../../../fixedIncome/docs/images/no_arbitrage_multivariate_mean_reverting.png')
    plt.show()


