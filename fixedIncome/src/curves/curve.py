"""
This script contains a base Curve class from which all other curves will subclass.
This class contains basic interpolation functionality, functionality for converting
from dates to floats on the interpolation x axis.

Each subclass will provide additional functionality.
"""

from datetime import date
from typing import NamedTuple, Iterable, Optional, Union
from collections.abc import Callable
from enum import Enum
import scipy
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountConvention, DayCountCalculator


class InterpolationMethod(Enum):
    PREVIOUS = 'previous'
    LINEAR = 'linear'
    QUADRATIC_SPLINE = 'quadratic'
    CUBIC_SPLINE = 'cubic'


class EndBehavior(Enum):
    ERROR = 0
    CONSTANT = 1


class KnotValuePair(NamedTuple):
    knot: date
    value: float


class Curve(Callable):
    """
    A base curve object which interpolates between the provided date, value pairs and
    which handles the necessary conversions from date values into floats.
    Subsequent curve objects (i.e. interests rate, survival, and yield curves)
    """

    def __init__(self,
                 interpolation_values: Iterable[KnotValuePair],
                 interpolation_method: InterpolationMethod,
                 interpolation_day_count_convention: DayCountConvention,
                 reference_date: Optional[date] = None,
                 left_end_behavior: EndBehavior = EndBehavior.ERROR,
                 right_end_behavior: EndBehavior = EndBehavior.ERROR) -> None:

        self.interpolation_values = sorted(list(interpolation_values), key=lambda interp_val: interp_val.knot)
        self.interpolation_method = interpolation_method
        self.interpolation_day_count_convention = interpolation_day_count_convention
        self.reference_date = reference_date if reference_date is not None else self.interpolation_values[0].knot
        self.left_end_behavior = left_end_behavior
        self.right_end_behavior = right_end_behavior

        self.interpolator: Callable[[date], float]
        self._create_interpolation_object()


    def __call__(self, date_obj: date) -> float:
        """ Shortcut to calling the interpolate method which allows the user to call the object directly. """
        return self.interpolate(date_obj)


    def date_to_interpolation_axis(self, date_obj: date) -> float:
        """
        Converts from a date to the interpolation x-axis value based on the
        reference date provided when the YieldCurve object is instantiated,
        the user-provided date, and the interpolation day-count convention string
        provided when the YieldCurve object is instantiated.
        day_count(reference_date, date) -> x-axis float value.
        """
        interpolation_axis_value = DayCountCalculator.compute_accrual_length(
            start_date=self.reference_date, end_date=date_obj, dcc=self.interpolation_day_count_convention
        )
        return interpolation_axis_value


    def _create_interpolation_object(self) -> None:
        """
        Creates the interpolation object, a function which maps dates into floats.
        """

        x_values = [self.date_to_interpolation_axis(knot_val_pair.knot) for knot_val_pair in self.interpolation_values]
        y_values = [knot_val_pair.value for knot_val_pair in self.interpolation_values]

        interpolator = scipy.interpolate.interp1d(x=x_values,
                                                  y=y_values,
                                                  kind=self.interpolation_method.value,
                                                  assume_sorted=True)
        self.interpolator = interpolator


    def interpolate(self, date_obj: date, adjustment: Optional[Callable[[date], float]] = None) -> float:
        """
        Method which uses the interpolation object
        """
        adjustment = adjustment if adjustment is not None else lambda _: 0
        first_date, first_val = self.interpolation_values[0]
        last_date, last_val = self.interpolation_values[-1]

        if date_obj < first_date:

            match self.left_end_behavior:

                case EndBehavior.CONSTANT:
                    return first_val + adjustment(date_obj)

                case EndBehavior.ERROR:
                    raise ValueError(f'Error in interpolate left end behavior. '
                                     f'The value {date_obj!r} is less than the first knot value {first_date!r}.')
                case _:
                    raise ValueError(f'Error in interpolate. {self.left_end_behavior} is not a valid case.')

        elif date_obj > last_date:

            match self.right_end_behavior:
                case EndBehavior.CONSTANT:
                    return last_val + adjustment(date)

                case EndBehavior.ERROR:
                    raise ValueError(f'Error in interpolate right end behavior. '
                                     f'The value {date_obj!r} is greater than the last knot value {last_date!r}.')
                case _:
                    raise ValueError(f'Error in interpolate. {self.right_end_behavior} is not a valid case.')

        else:
            x_axis_val = self.date_to_interpolation_axis(date_obj)
            return self.interpolator(x_axis_val) + adjustment(date_obj)








