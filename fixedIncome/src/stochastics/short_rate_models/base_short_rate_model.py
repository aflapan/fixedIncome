
from enum import Enum
import numpy as np
from typing import NamedTuple, Optional
from collections.abc import Callable
from abc import abstractmethod


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

    @abstractmethod
    def show_drift_diffusion_collection_keys(self) -> tuple[str]:
        """

        """


    @abstractmethod
    def generate_path(
            self, dt: float, starting_values: np.ndarray | float, set_path: bool = True, seed: Optional[int] = None
            ) -> np.array:
        """
        An abstract method for any ShortRate Model to generate a sample path from the drift diffusion
        SDE collection provided when the object was instantiated.
        """




