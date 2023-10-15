
from datetime import date, datetime
from typing import NamedTuple, Callable
import math
import numpy as np
from typing import NamedTuple, Optional
from collections.abc import Callable


class DriftDiffusionPair(NamedTuple):
    drift: Callable[[float, float], float]
    diffusion: Callable[[float, float], float]



class ShortRateModel:

    def __init__(self,
                 drift_diffusion_collection: dict[DriftDiffusionPair],
                 brownian_motion
                 ) -> None:
        pass


    def generate_sample_path(self, num_increments_per_day: int, num_days: int, ) -> np.array:
        pass


    def generate_sde_euler_paths(
            drift_function: Callable[[float, float], float],
            diffusion_function: Callable[[float, float], float],
            num_paths: int,
            num_steps: int,
            end_time: float,
            starting_value: float = 0.0,
            correlation_matrix: Optional[np.array] = None
    ) -> np.array:
        """
        Generates discretized solution paths to the stochastic differential equation
        dX_t = mu(t, X_t) dt + sigma(t, X_t) dW_t
        where mu and sigma are the drift and diffusion functions, respectively.
        """
        dt = end_time / num_steps
        solutions = np.zeros((num_paths, num_steps + 1))
        solutions[:, 0] = starting_value
        brownian_increments = generate_brownian_increments(num_paths, num_steps, end_time, correlation_matrix)

        for step in range(num_steps):
            time = step * dt / end_time
            drift = np.array([drift_function(time, value) for value in solutions[:, step]])
            diffusion = np.array([diffusion_function(time, value) for value in solutions[:, step]])
            increments = drift * dt + diffusion * brownian_increments[:, step]
            solutions[:, step + 1] = solutions[:, step] + increments

        return solutions

