from datetime import datetime
import numpy as np
from typing import NamedTuple, Optional
from collections.abc import Callable
from abc import abstractmethod
from fixedIncome.src.stochastics.brownian_motion import datetime_to_path_call


class DriftDiffusionPair(NamedTuple):
    drift: Callable[[float, ...], float]
    diffusion: Callable[[float, ...], float]


class ShortRateModel:

    def __init__(self,
                 drift_diffusion_collection: dict[str, DriftDiffusionPair],
                 brownian_motion
                 ) -> None:
        self.drift_diffusion_collection = drift_diffusion_collection
        self.brownian_motion = brownian_motion
        self._path = None

    @property
    def path(self) -> np.ndarray:
        return self._path

    def __call__(self, datetime_obj: datetime) -> float | np.ndarray:
        """
        Shortcut to allow the user to directly call the Short Rate model using a datetime rather
        than index and interpolate the path directly.
        """
        values = datetime_to_path_call(datetime_obj,
                                     start_date_time=self.brownian_motion.start_date_time,
                                     end_date_time=self.brownian_motion.end_date_time,
                                     path=self.path)
        return values

    @abstractmethod
    def show_drift_diffusion_collection_keys(self) -> tuple[str]:
        """
        An interface method to have the model display a tuple of
        all keys which index the drift and diffusion collection
        to give an individual drift-diffusion pair of functions.
        """

    @abstractmethod
    def generate_path(
            self, dt: float, starting_values: np.ndarray | float, set_path: bool = True, seed: Optional[int] = None
            ) -> np.array:
        """
        An abstract method for any ShortRate Model to generate a sample path from the drift diffusion
        SDE collection provided when the object was instantiated.
        """




