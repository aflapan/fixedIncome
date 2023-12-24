from datetime import date, datetime, time
from dateutil.relativedelta import relativedelta
import numpy as np
import math
from typing import NamedTuple, Optional
from collections.abc import Callable
from abc import abstractmethod
import itertools
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.stochastics.brownian_motion import datetime_to_path_call, BrownianMotion
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountConvention
from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.curves.base_curve import DiscountCurve, KnotValuePair, Curve
from fixedIncome.src.curves.curve_enumerations import CurveIndex, EndBehavior, InterpolationMethod

class DriftDiffusionPair(NamedTuple):
    drift: Callable[[float, ...], float]
    diffusion: Callable[[float, ...], float]


class ShortRateModel:

    def __init__(self,
                 drift_diffusion_collection: dict[str, DriftDiffusionPair],
                 brownian_motion: BrownianMotion,
                 day_count_convention: DayCountConvention,
                 dt: float = 1/1_000  # dt increment in units of years
                 ) -> None:
        self.drift_diffusion_collection = drift_diffusion_collection
        self.brownian_motion = brownian_motion
        self.day_count_convention = day_count_convention
        self._dt = dt
        self._rate_index = 0  # row index of the self.path numpy array which corresponds to the short rate
                              # Subclass models will often generate multiple different intermediary rates/values
        self._path = None
        self._integrated_path = None
        self._continuously_compounded_accrual_path = None
        self._discount_curve = None

    @property
    def path(self) -> np.ndarray:
        return self._path

    @property
    def rate_index(self) -> int:
        return self._rate_index
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def integrated_path(self) -> np.array:
        return self._integrated_path

    @property
    def discount_curve(self) -> Optional[DiscountCurve]:
        return self._discount_curve

    def _reset_paths_and_curves(self) -> None:
        self._path = None
        self._integrated_path = None
        self._continuously_compounded_accrual_path = None
        self._discount_curve = None

    def set_dt(self, new_dt: float) -> None:
        """ Sets a new increment dt. The old path is set to None because any path
        is no longer valid if """
        self._path = None  # path generated from old dt no longer valid
        self._dt = new_dt

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
            self, starting_values: np.ndarray | float, set_path: bool = True, seed: Optional[int] = None
            ) -> np.array:
        """
        An abstract method for any ShortRate Model to generate a sample path from the drift diffusion
        SDE collection provided when the object was instantiated.
        """

    def generate_integrated_path(self,
                        start_date_time: Optional[datetime] = None,
                        end_date_time: Optional[datetime] = None,
                        datetimes: Optional[list[datetime | date]] = None,
                        ) -> np.array:
        """
        """
        if self.path is None:
            raise ValueError("Path is currently None. First generate a sample path of spot rates.")

        start_date_time = self.brownian_motion.start_date_time if start_date_time is None else start_date_time
        end_date_time = self.brownian_motion.end_date_time if end_date_time is None else end_date_time
        if datetimes is None:
            datetimes = Scheduler.generate_dates_by_increments(start_date=start_date_time,
                                                               end_date=end_date_time,
                                                               increment=relativedelta(seconds=self.dt * DayCountCalculator.SECONDS_PER_YEAR),
                                                               max_dates=1_000_000)
        running_integral = 0.0
        interpolation_values = [0.0]

        for start_dt, end_dt in itertools.pairwise(datetimes):
            start_rate, end_rate = self(start_dt)[self.rate_index], self(end_dt)[self.rate_index]
            min_rate, max_rate = min(start_rate, end_rate), max(start_rate, end_rate)
            accrual = DayCountCalculator.compute_accrual_length(start_dt, end_dt, self.day_count_convention)

            if min_rate < 0:
                next_trapezoid_val = max_rate * accrual + (min_rate - max_rate) * accrual / 2
            else:
                next_trapezoid_val = min_rate * accrual + (max_rate - min_rate) * accrual / 2
            # compute integral
            running_integral += next_trapezoid_val
            interpolation_values.append(running_integral)

        self._integrated_path = np.array(interpolation_values)
        return self._integrated_path

    def accrual_curve(self,
                      start_date_time: Optional[datetime] = None,
                      end_date_time: Optional[datetime] = None,
                      datetimes: Optional[list[datetime | date]] = None,
                      day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_OVER_ACTUAL
                      ) -> Curve:
        """
        """
        if self.path is None:
            raise ValueError("Path is currently None. First generate a sample path of spot rates.")

        start_date_time = self.brownian_motion.start_date_time if start_date_time is None else start_date_time
        end_date_time = self.brownian_motion.end_date_time if end_date_time is None else end_date_time
        if datetimes is None:
            datetimes = Scheduler.generate_dates_by_increments(start_date=start_date_time,
                                                               end_date=end_date_time,
                                                               increment=relativedelta(seconds=self.dt * DayCountCalculator.SECONDS_PER_YEAR),
                                                               max_dates=1_000_000)

        if self._integrated_path is None:
            self.generate_integrated_path(start_date_time, end_date_time, datetimes)

        compound_interpolation_values = [KnotValuePair(start_date_time, 1.0)]
        compound_interpolation_values += [KnotValuePair(datetime_obj, math.exp(integral))
                                          for datetime_obj, integral in zip(datetimes, self._integrated_path)]

        self._continuously_compounded_accrual_curve = Curve(interpolation_values=compound_interpolation_values,
                                                            interpolation_method=InterpolationMethod.LINEAR,
                                                            interpolation_day_count_convention=day_count_convention,
                                                            reference_date=start_date_time,
                                                            left_end_behavior=EndBehavior.ERROR,
                                                            right_end_behavior=EndBehavior.ERROR)

        return self._continuously_compounded_accrual_curve


    def discount_curve(self,
                       start_date_time: Optional[datetime] = None,
                       end_date_time: Optional[datetime] = None,
                       datetimes: Optional[list[datetime | date]] = None,
                       curve_index: CurveIndex = CurveIndex.NONE,
                       day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_OVER_ACTUAL) -> DiscountCurve:

        if self.path is None:
            raise ValueError("Path is currently None. First generate a sample path of spot rates.")

        start_date_time = self.brownian_motion.start_date_time if start_date_time is None else start_date_time
        end_date_time = self.brownian_motion.end_date_time if end_date_time is None else end_date_time
        if datetimes is None:
            datetimes = Scheduler.generate_dates_by_increments(start_date=start_date_time,
                                                               end_date=end_date_time,
                                                               increment=relativedelta(seconds=self.dt * DayCountCalculator.SECONDS_PER_YEAR),
                                                               max_dates=1_000_000)

        if self._integrated_path is None:
            self.generate_integrated_path(start_date_time, end_date_time, datetimes)

        discount_interpolation_values = np.exp(-self._integrated_path)
        discount_knot_values = [KnotValuePair(dateobj, df)
                                for dateobj, df in zip(datetimes, discount_interpolation_values)]

        self._discount_curve = DiscountCurve(interpolation_values=discount_knot_values,
                                             interpolation_method=InterpolationMethod.LINEAR,
                                             index=curve_index,
                                             interpolation_day_count_convention=day_count_convention,
                                             reference_date=self.brownian_motion.start_date_time,
                                             left_end_behavior=EndBehavior.ERROR,
                                             right_end_behavior=EndBehavior.ERROR)

        return self._discount_curve




