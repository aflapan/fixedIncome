from typing import Optional
from datetime import date, datetime
import numpy as np
from fixedIncome.src.stochastics.brownian_motion import datetime_to_path_call, BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.base_short_rate_model import DriftDiffusionPair

def geometric_brownian_motion_drift_diffusion(drift: float, volatility: float) -> DriftDiffusionPair:
    """
    Produces the drift-diffusion named tuple of functions in the SDE for geometric brownian motion:
    dX_t = mu * X_t dt + sigma * X_t dW_t.
    """

    def drift_fxcn(time: float, current_value: float) -> float:
        return drift * current_value

    def diffusion_fxcn(time: float, current_value: float) -> float:
        return volatility * current_value

    return DriftDiffusionPair(drift=drift_fxcn, diffusion=diffusion_fxcn)

class GeometricBrownianMotion:
    """
    A class for generating geometric brownian motion sample paths.
    """
    def __init__(self,
                 drift: float,
                 volatility: float,
                 start_date_time: datetime | date,
                 end_date_time: datetime | date,
                 dimension: int = 1,
                 correlation_matrix: Optional[np.ndarray] = None,
                 dt: float = 1/1_000):
        self._start_date_time = start_date_time
        self._end_date_time = end_date_time
        self._dimension = dimension
        self.correlation_matrix = correlation_matrix if correlation_matrix is not None else np.eye(self._dimension)
        self.drift = drift
        self.volatility = volatility
        self._dt = dt
        self.drift_diffusion_pair = geometric_brownian_motion_drift_diffusion(drift=self.drift,
                                                                              volatility=self.volatility)

        self._brownian_motion = BrownianMotion(start_date_time=start_date_time,
                                               end_date_time=end_date_time,
                                               dimension=self.dimension,
                                               correlation_matrix=self.correlation_matrix)

        self._path = None

    @property
    def start_date_time(self) -> datetime:
        return self._start_date_time

    @property
    def end_date_time(self) -> datetime:
        return self._end_date_time

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def path(self):
        return self._path

    @property
    def brownian_motion(self) -> BrownianMotion:
        return self._brownian_motion

    def generate_path(self, starting_value: np.ndarray | float, set_path: bool = True, seed: Optional[int] = None
                      ) -> np.array:
        """
        Generates a Geometric Brownian Motion sample paths as np.arrays and stores them as the self._path
        attribute.
        """
        drift_fxcn, diffusion_fxcn = self.drift_diffusion_pair

        brownian_increments = self.brownian_motion.generate_increments(dt=self.dt, seed=seed)
        solution = np.empty((self.dimension,  brownian_increments.shape[1]+1))
        current_val = starting_value.flatten()
        time = 0
        for index, shock in enumerate(brownian_increments.T):
            solution[:, index] = current_val
            drift_increment = np.array([drift_fxcn(time, val) for val in current_val]) * self.dt
            diffusion_shock = np.array([diffusion_fxcn(time, val) for val in current_val]) * shock  # shock contains sqrt(dt) scaling
            current_val = current_val + drift_increment + diffusion_shock
            time += self.dt

        solution[:, brownian_increments.shape[1]] = current_val  # solution has one more slot

        if set_path:
            self._path = solution

        return solution

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
    end_time = datetime(2073, 10, 15, 0, 0, 0, 0)

    rho = 0.5
    correlation_mat = np.array([[1.0, rho, rho], [rho, 1.0, rho], [rho, rho, 1.0]])

    gbm = GeometricBrownianMotion(drift=0.05,
                                  volatility=0.25,
                                  start_date_time=start_time,
                                  end_date_time=end_time,
                                  dimension=3,
                                  correlation_matrix=correlation_mat)

    path = gbm.generate_path(starting_value=np.array([1.0, 1.0, 1.0]), seed=1)

    plt.figure(figsize=(13, 6))
    plt.plot(path.T, linewidth=0.5)
    plt.grid(alpha=0.25)
    plt.show()




