import numpy as np
from scipy.optimize import minimize
from typing import Optional


class PortfolioOptimizer:
    def __init__(self, mean_vector: np, covariance_mat):
        self._mean_vector = mean_vector
        self._covariance_matrix = covariance_mat
        self.num_assets = len(self._mean_vector)
        self.min_return = min(self._mean_vector)
        self.max_return = max(self._mean_vector)

    @property
    def mean_vector(self) -> np.array:
        return self._mean_vector

    @property
    def covariance_matrix(self) -> np.array:
        return self._covariance_matrix


    def find_minimum_variance_portfolio(self,
                                        target_return: float,
                                        allow_short_positions: bool = False
                                        ) -> float:
        ''' Calculates the asset weights of the minimum variance portfolio.
        '''

        equality_contraints = {'type': 'eq'}
        weight_coefficients = np.array([1.0 for _ in range(self.num_assets)])
        equality_constraint_matrix = np.array([weight_coefficients, self.mean_vector])
        target_vector = np.array([1.0, target_return])

        def equality_constraint_func(weights):
            return equality_constraint_matrix.dot(weights) - target_vector

        equality_contraints['fun'] = equality_constraint_func
        constraints = [equality_contraints]

        if not allow_short_positions:  # need additional inequality constraint on weights >= 0
            inequality_constrains = {'type': 'ineq'}

            def inequality_constraint_func(weights):
                return weights

            inequality_constrains['fun'] = inequality_constraint_func

            constraints.append(inequality_constrains)
            bounds = [(0.0, 1.0) for _ in range(self.num_assets)]
        else:
            bounds = [(float('-inf'), float('inf')) for _ in range(self.num_assets)]

        def objective_func(weights: np.array) -> float:
            return  weights.dot(self.covariance_matrix.dot(weights))

        def gradient(weights: np.array) -> np.array:
            return 2 * self.covariance_matrix.dot(weights)

        starting_weights = np.array([1/self.num_assets for _ in range(self.num_assets)])
        results = minimize(fun=objective_func, x0=starting_weights, jac=gradient,
                            bounds=bounds,
                           constraints=constraints,
                           method='SLSQP')
        return results['x']


    def calculate_efficient_frontier(self,
                                     lower_return: Optional[float] = None,
                                     upper_return: Optional[float] = None,
                                     return_increment: float = 0.001,
                                     allow_short_positions: bool = False) -> tuple[np.array, np.array, np.array]:
        ''' Calculates the weights for the minimum variance portfolios across a range of target returns.
        '''

        if lower_return is None:
            lower_return = self.min_return + return_increment

        if upper_return is None:
            upper_return = self.max_return - return_increment

        returns = np.arange(start=lower_return, stop=upper_return + return_increment, step=return_increment)
        weights = np.array([self.find_minimum_variance_portfolio(target_return, allow_short_positions)
                  for target_return in returns])

        volatilities = np.sqrt(np.array([weight.dot(self.covariance_matrix.dot(weight)) for weight in weights]))

        return returns, volatilities, weights

    def sharpe_ratio(self,
                     returns: np.array,
                     volatilities: np.array,
                     risk_free_rate: float ) -> np.array:
        ''' Calculates the sharpe ratio
            (return - risk_Free_rate) / volatility
        '''
        return (returns - risk_free_rate) / volatilities


    def find_maximum_utility_portfolio(self,
                                  initial_wealth: float,
                                  param: float,
                                  allow_short_positions: bool = False) -> np.array:
        ''' Maximizes the expected utility E U(1+R)
        where the utility function U(x, param) is equal to 1 - exp(- param * x )
        and R are assumed to be normally distributed.

        This maximization problem is of the form
            maximize_{weight} [ return_mean - (param * initial_wealth).dot(return_variance) /2 ]
        where return mean = weight.dot(mean_vector)
        and return variance is weight.dot(covariance_mat.dot(weight))
        '''
        equality_contraints = {'type': 'eq'}
        weight_coefficients = np.array([1.0 for _ in range(self.num_assets)])

        def equality_constraint_func(weights):
            return weight_coefficients.dot(weights) - 1.0

        equality_contraints['fun'] = equality_constraint_func
        constraints = [equality_contraints]

        # use negative of maximization object and minimize
        def objective_func(weights: np.array):
            mean_return = weights.dot(self.mean_vector)
            return_variance = weights.dot(self.covariance_matrix.dot(weights))
            scalar_factor = param * initial_wealth / 2
            return -mean_return + return_variance * scalar_factor

        def gradient(weights: np.array) -> np.array:
            scalar_factor = param * initial_wealth / 2
            return -self.mean_vector + 2 * scalar_factor * self.covariance_matrix.dot(weights)

        if allow_short_positions:
            bounds = [(float('-inf'), float('inf')) for _ in range(self.num_assets)]
        else:
            bounds = [(0.0, 1.0) for _ in range(self.num_assets)]

        starting_weights = np.array([1/self.num_assets for _ in range(self.num_assets)])
        results = minimize(fun=objective_func, x0=starting_weights, jac=gradient,
                           constraints=constraints, bounds=bounds, method='SLSQP')

        return results['x']


