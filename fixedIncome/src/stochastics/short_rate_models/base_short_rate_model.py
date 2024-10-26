from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import math
from typing import Optional, Callable
from abc import abstractmethod
import itertools
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.stochastics.base_processes import DriftDiffusionPair
from fixedIncome.src.stochastics.brownian_motion import datetime_to_path_call, BrownianMotion
from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention
from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.curves.base_curve import DiscountCurve, KnotValuePair, Curve
from fixedIncome.src.curves.curve_enumerations import CurveIndex, EndBehavior, InterpolationMethod
from fixedIncome.src.stochastics.base_processes import DiffusionProcess


class ShortRateModel(DiffusionProcess):

    def __init__(self,
                 short_rate_transformation: Callable[[np.array], float],
                 state_variables_diffusion_process: DiffusionProcess
                 ) -> None:

        self.short_rate_transformation = short_rate_transformation
        self.state_variables_diffusion_process = state_variables_diffusion_process
        super().__init__(drift_diffusion_collection=self.state_variables_diffusion_process.drift_diffusion_collection,
                         brownian_motion=self.state_variables_diffusion_process.brownian_motion,
                         dt=self.state_variables_diffusion_process.dt)

        self._short_rate_path = None
        self._integrated_path = None
        self._continuously_compounded_accrual_path = None
        self._discount_curve = None

    @property
    def short_rate_path(self) -> np.array:
        return self._short_rate_path

    @property
    def integrated_path(self) -> np.array:
        return self._integrated_path

    @property
    def discount_curve(self) -> Optional[DiscountCurve]:
        return self._discount_curve

    def __call__(self, datetime_obj) -> float:
        """
        Returns the shortrate for the corresponding datetime.
        The short rate is the provided transformation of the underlying state variables.
        """
        state_variables = self.state_variables_diffusion_process(datetime_obj)
        return self.short_rate_transformation(state_variables)

    def generate_path(self,
                      starting_state_space_values: np.ndarray | float,
                      set_path: bool = True,
                      seed: Optional[int] = None
                      ) -> np.array:
        """
        A function which generates the short rate sample path from the underlying
        state variable diffusion sample paths.
        """
        #self._reset_paths_and_curves()
        self.state_variables_diffusion_process.generate_path(starting_value=starting_state_space_values,
                                                             set_path=set_path,
                                                             seed=seed)

        path_length = self.state_variables_diffusion_process.path.shape[1]
        self._path = np.zeros(shape=(path_length,))
        for index in range(path_length):
            self._path[index] = self.short_rate_transformation(self.state_variables_diffusion_process.path[:, index])


    def _reset_paths_and_curves(self) -> None:
        """
        Helper function to set all the paths to None.
        """
        super()._reset_paths_and_curves()  # set the underlying state variable paths to None
        self._short_rate_path = None
        self._integrated_path = None
        self._continuously_compounded_accrual_path = None
        self._discount_curve = None

    def generate_integrated_path(self, datetimes: Optional[list[datetime | date]] = None) -> np.array:
        """
        Method for creating a path which represents the integral of the short rate path
        t -> int_{t_0}^{t} r_s  ds
        """
        if self.path is None:
            raise ValueError("Path is currently None. First generate a sample path of spot rates.")

        start_date_time = self.brownian_motion.start_date_time
        end_date_time = self.brownian_motion.end_date_time
        if datetimes is None:
            datetimes = Scheduler.generate_dates_by_increments(start_date=start_date_time,
                                                               end_date=end_date_time,
                                                               increment=self.dt,
                                                               max_dates=1_000_000)
        running_integral = 0.0
        old_time_accrual = 0
        interpolation_values = [0.0]

        for start_dt, end_dt in itertools.pairwise(datetimes):
            start_rate, end_rate = self(start_dt), self(end_dt)
            min_rate, max_rate = min(start_rate, end_rate), max(start_rate, end_rate)
            new_time_accrual = DayCountCalculator.compute_accrual_length(start_date_time, end_dt, self.day_count_convention)
            accrual = new_time_accrual - old_time_accrual


            if min_rate < 0:
                next_trapezoid_val = max_rate * accrual + (min_rate - max_rate) * accrual / 2
            else:
                next_trapezoid_val = min_rate * accrual + (max_rate - min_rate) * accrual / 2
            # compute integral
            running_integral += next_trapezoid_val
            interpolation_values.append(running_integral)

        self._integrated_path = np.array(interpolation_values)
        return self._integrated_path


    def accrual_curve(self, datetimes: Optional[list[datetime | date]] = None) -> Curve:
        """
        Function which generates the curve resulting from continuously-compounding the
        short rate path.
        """
        if self.path is None:
            raise ValueError("Path is currently None. First generate a sample path of spot rates.")

        start_date_time = self.brownian_motion.start_date_time
        end_date_time = self.brownian_motion.end_date_time
        if datetimes is None:
            datetimes = Scheduler.generate_dates_by_increments(start_date=start_date_time,
                                                               end_date=end_date_time,
                                                               increment=self.dt,
                                                               max_dates=1_000_000)

        if self._integrated_path is None:
            self.generate_integrated_path(start_date_time, end_date_time, datetimes)

        compound_interpolation_values = [KnotValuePair(start_date_time, 1.0)]
        compound_interpolation_values += [KnotValuePair(datetime_obj, math.exp(integral))
                                          for datetime_obj, integral in zip(datetimes, self._integrated_path)]

        self._continuously_compounded_accrual_curve = Curve(interpolation_values=compound_interpolation_values,
                                                            interpolation_method=InterpolationMethod.LINEAR,
                                                            interpolation_day_count_convention=self.day_count_convention,
                                                            reference_date=start_date_time,
                                                            left_end_behavior=EndBehavior.ERROR,
                                                            right_end_behavior=EndBehavior.ERROR)

        return self._continuously_compounded_accrual_curve

    def create_discount_curve(self,
                       datetimes: Optional[list[datetime | date]] = None,
                       curve_index: CurveIndex = CurveIndex.NONE) -> DiscountCurve:

        if self.path is None:
            raise ValueError("Path is currently None. First generate a sample path of spot rates.")

        start_date_time = self.brownian_motion.start_date_time
        end_date_time = self.brownian_motion.end_date_time
        if datetimes is None:
            datetimes = Scheduler.generate_dates_by_increments(start_date=start_date_time,
                                                               end_date=end_date_time,
                                                               increment=self.dt,
                                                               max_dates=1_000_000)

        if self._integrated_path is None:
            self.generate_integrated_path(datetimes)

        discount_interpolation_values = np.exp(-self._integrated_path)
        discount_knot_values = [KnotValuePair(datetime_obj, df)
                                for datetime_obj, df in zip(datetimes, discount_interpolation_values)]

        self._discount_curve = DiscountCurve(interpolation_values=discount_knot_values,
                                             interpolation_method=InterpolationMethod.LINEAR,
                                             index=curve_index,
                                             interpolation_day_count_convention=self.day_count_convention,
                                             reference_date=self.brownian_motion.start_date_time,
                                             left_end_behavior=EndBehavior.ERROR,
                                             right_end_behavior=EndBehavior.ERROR)

        return self._discount_curve


