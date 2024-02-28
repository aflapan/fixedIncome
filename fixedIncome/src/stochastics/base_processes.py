"""

"""

from collections import abc
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Optional, NamedTuple, Callable
import numpy as np
from fixedIncome.src.stochastics.brownian_motion import BrownianMotion, datetime_to_path_call

class DriftDiffusionPair(NamedTuple):
    """
    A class to
    """
    drift: Callable[[float, np.array], np.array]
    diffusion: Callable[[float, np.array], np.array]


class DiffusionProcess(abc.Callable):

    def __init__(self,
                 drift_diffusion_collection: dict[str: DriftDiffusionPair],
                 brownian_motion: BrownianMotion,
                 dt: timedelta|relativedelta = relativedelta(hours=1)) -> None:

        self.brownian_motion = brownian_motion
        self._start_date_time = brownian_motion.start_date_time
        self._end_date_time = brownian_motion.end_date_time
        self._dt = dt

        self.drift_diffusion_collection = drift_diffusion_collection
        self._drift_diffusion_name_to_index = {name: index
                                               for index, name in enumerate(self.drift_diffusion_collection.keys())}

        self._dimension = len(self.drift_diffusion_collection)
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
    def dt(self) -> timedelta | relativedelta:
        return self._dt

    @property
    def path(self):
        return self._path


    def __call__(self, datetime_obj: datetime) -> np.array:
        """
        Shortcut to allow the user to directly call the MeanRevertingProcess object directly rather
        than index and interpolate the path.
        """
        return datetime_to_path_call(datetime_obj,
                                     start_date_time=self.start_date_time,
                                     end_date_time=self.end_date_time,
                                     path=self.path)

    def generate_path(self,
                      starting_value: np.ndarray | float,
                      set_path: bool = True,
                      seed: Optional[int] = None
                      ) -> np.ndarray:
        """
        """

        brownian_increments, dt_increments = self.brownian_motion.generate_increments(dt=self.dt, seed=seed)
        solution = np.empty((brownian_increments.shape[0], brownian_increments.shape[1] + 1))
        current_val = starting_value.flatten()
        time = 0
        for index, (shock, dt) in enumerate(zip(brownian_increments.T, dt_increments.T)):
            solution[:, index] = current_val
            next_step = np.zeros((self.dimension,))

            for drift_diffusion_name, drift_diffusion_functions in self.drift_diffusion_collection.items():
                row_index = self._drift_diffusion_name_to_index[drift_diffusion_name]
                drift_increment = drift_diffusion_functions.drift(time, current_val) * dt
                diffusion_shock = drift_diffusion_functions.diffusion(time, current_val) @ shock  # shock contains sqrt(dt) scaling
                next_step[row_index] = drift_increment + diffusion_shock

            current_val += next_step
            time += dt

        solution[:, brownian_increments.shape[1]] = current_val  # solution has one more slot
        if set_path:
            self._path = solution

        return solution



class DiffusionJumpProcess(DiffusionProcess):
    pass




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import math

    reversion_level = np.array([1.0, 2.0])

    reversion_directions = np.array([[1.0/math.sqrt(2), 1.0/math.sqrt(2)],
                                     [1.0/math.sqrt(2), -1.0/math.sqrt(2)]])
    reversion_mat = reversion_directions @ np.array([[1.0, 0.0], [0.0, 0.1]]) @ reversion_directions.T
    eigen_values, eigenvectors = np.linalg.eig(reversion_mat)

    rho = 0.0
    stand_dev_1 = 1
    stand_dev_2 = 1
    correlation_matrix = np.array([[stand_dev_1**2, stand_dev_1 * stand_dev_2* rho],
                                   [stand_dev_1 * stand_dev_2 * rho, stand_dev_2**2]])

    start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
    end_time = datetime(2063, 10, 15, 0, 0, 0, 0)

    bm = BrownianMotion(start_date_time=start_time,
                        end_date_time=end_time,
                        dimension=2,
                        correlation_matrix=None)
    def drift_1(t, Xt):
        return reversion_mat[0, :] @ (reversion_level - Xt)

    def diffusion_1(t, shocks):
        return correlation_matrix[0, :]

    def drift_2(t, Xt):
        return reversion_mat[1, :] @ (reversion_level - Xt)

    def diffusion_2(t, shocks):
        return correlation_matrix[1, :]


    dimension1_drift_diffusion = DriftDiffusionPair(drift=drift_1, diffusion=diffusion_1)
    dimension2_drift_diffusion = DriftDiffusionPair(drift=drift_2, diffusion=diffusion_2)

    drift_diffusion_collection = {
        'dimension 1': dimension1_drift_diffusion,
        'dimension 2': dimension2_drift_diffusion
    }

    diffusion = DiffusionProcess(drift_diffusion_collection=drift_diffusion_collection,
                                 brownian_motion=bm)

    starting_value = np.array([1.0, 1.0])
    path = diffusion.generate_path(starting_value=starting_value, set_path=True, seed=1)
    scaled_eigenvalues = 1 / np.sqrt(eigen_values)
    #scaled_eigenvalues /= max(scaled_eigenvalues)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_aspect('equal', adjustable='box')
    plt.plot(path[0, :], path[1, :], linewidth=0.5, alpha=1, label='Sample Path')
    #plt.plot(*tuple(starting_value), 'x', markersize=8, color='crimson')

    eigen_vec = eigenvectors[:, 0] * scaled_eigenvalues[0] + reversion_level
    plt.plot((reversion_level[0], eigen_vec[0]),
             (reversion_level[1], eigen_vec[1]), color='tab:red', linewidth=2,
             label='Reversion Matrix Principal Direction\nScaled Inversely to Root Eigenvalue')

    eigen_vec = eigenvectors[:, 1] * scaled_eigenvalues[1] + reversion_level
    #scaled_eigenvalue = 1
    plt.plot((reversion_level[0], eigen_vec[0] ),
             (reversion_level[1], eigen_vec[1] ), color='tab:red', linewidth=2)

    plt.plot(*tuple(reversion_level), 'D', markersize=8, color='tab:red', label='Reversion Level')
    plt.title('Two-Dimensional Mean-Reverting Process')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    ax.legend(bbox_to_anchor=(1.0, 0.6), frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig('../../../../fixedIncome/docs/images/two_dimensional_mean_reverting_process.png')
    plt.show()


    plt.figure(figsize=(12, 6))
    plt.title('Two Dimensional Mean-Reverting Process Coordinate Paths')
    plt.plot(path[0,:], linewidth=0.75, color='tab:blue')
    plt.axhline(reversion_level[0], linestyle='dashed', color='tab:blue')
    plt.plot(path[1,:], linewidth=0.75, color='mediumaquamarine')
    plt.axhline(reversion_level[1], linestyle='dashed', color='mediumaquamarine')
    plt.grid(alpha=0.25)
    plt.show()