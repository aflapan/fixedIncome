import numpy as np
import pandas as pd
import datetime
from typing import NamedTuple
import scipy
from day_count_calculator import DayCountCalculator
from bond import Bond



class KnotValuePair(NamedTuple):
    knot: float
    value: float

    def __str__(self):
        return f"knot point {self.knot}, value {self.value}"

class YieldCurve(object):

    def __init__(self,
                 interpolation_values:pd.Series,
                 interpolation_method:str,
                 interpolation_space:str,
                 interpolation_day_count_convention:str,
                 reference_date:datetime.date,
                 left_end_behavior:str,
                 right_end_behavior:str) -> None:
        """
        Creates an instance of a yield curve object.

        Parameters:
            interpolation_values: pd.Series whose indices are the dates used as knot points
            interpolation_method: A string representing the interpolation method to use. Valid inputs are
                                  'linear', ‘quadratic’, ‘cubic’, ‘previous’.
            interpolation_day_count_convention: A string represeting the day count convention
            reference_date: The date from which
            left_end_behavior: A string determining the behavior of evaluating x values below the first
        """

        self.interpolation_values = interpolation_values
        self.interpolation_method = interpolation_method
        self.interpolation_space = interpolation_space
        self.interpolation_dcc = interpolation_day_count_convention
        self.reference_date = reference_date
        self.left_end_behavior = left_end_behavior
        self.right_end_behavior = right_end_behavior
        self.dcc_calculator_obj = DayCountCalculator()

        self.valid_interpolation_methods = {'linear', 'quadratic', 'cubic', 'previous'}
        self.valid_interpolation_spaces = {'discount_factor', 'forward_rate'}

        assert self.interpolation_method in self.valid_interpolation_methods
        assert self.interpolation_space in self.valid_interpolation_spaces

        self.knot_value_tuples_list = None
        self._preprocess_interpolation_values()
        self.interpolator = self._create_interpolation_object()



    #-----------------------------------------------------------
    # Functions for creating the interpolation method.

    def date_to_interpolation_axis(self, date:datetime.date) -> float:
        """
        converts from a date to the interpolation x-axis value based on the
        reference date provided when the YieldCurve object is instantiated,
        the user-provided date, and the interpolation day-count convention string
        provided when the YieldCurve object is instantiated.

        day_count(reference_date, date) -> x-axis float value.
        """
        interpolation_axis_val = self.dcc_calculator_obj.compute_accrual_length(
            start_date=self.reference_date, end_date=date, dcc=self.interpolation_dcc
        )

        return interpolation_axis_val


    def _preprocess_interpolation_values(self) -> list:
        """
        Pre-processes the interpolation dates into interpolation x-axis values.
        Returns an ordered list of tuples (x_val, y_val) of the x-axis knot points
        and the corresponding y values used for interpolation.

        Returns a list of KnotValuePair namedTuples, sorted by knot points.
        """

        self.interpolation_values.sort_index(inplace=True) # sorts by x values

        x_dates, y_values = self.interpolation_values.index.tolist(), self.interpolation_values.values.tolist()

        first_date_is_reference = x_dates[0] == self.reference_date

        # if interpolation space is discount factor space
        if self.interpolation_space == 'discount_factor' and not first_date_is_reference:
            x_dates = [self.reference_date] + x_dates
            y_values = [1.0] + y_values # DF for reference date ('today') is always 1.0

        # preprocesses dates into interpolation x axis values
        x_values = list(map(lambda date: self.date_to_interpolation_axis(date), x_dates))

        return [KnotValuePair(knot_val, y_val) for (knot_val, y_val) in zip(x_values, y_values)]



    def _create_interpolation_object(self):
        """
        Creates the interpolation object.
        """

        self.knot_value_tuples_list = self._preprocess_interpolation_values()

        x_values = [knot_val_pair.knot for knot_val_pair in self.knot_value_tuples_list]
        y_values = [knot_val_pair.value for knot_val_pair in self.knot_value_tuples_list]

        interpolator = scipy.interpolate.interp1d(x=x_values,
                                                  y=y_values,
                                                  kind=self.interpolation_method,
                                                  assume_sorted=True)

        return interpolator


    def interpolate(self, date:datetime.date) -> float:
        """
        Method which uses the interpolation object
        """
        x_axis_val = self.date_to_interpolation_axis(date)

        if x_axis_val < self.knot_value_tuples_list[0].knot:

            match(self.left_end_behavior):
                case 'constant':
                    return self.knot_value_tuples_list[0].value

                case _:
                    return ValueError(f'Error in interpolate. {self.left_end_behavior} not a valid input.')

        elif x_axis_val > self.knot_value_tuples_list[-1].knot:

            match(self.right_end_behavior):
                case 'constant':
                    return self.knot_value_tuples_list[-1].value

                case _:
                    raise ValueError(f'Error in interpolate. {self.right_end_behavior} not a valid input.')

        else:
            return self.interpolator(x_axis_val).item()







t_bond = Bond(price=100.6875,
              coupon=2.375,
              principal=100,
              tenor='30Y',
              purchase_date=datetime.date(2021, 5, 20),
              maturity_date=datetime.date(2051, 5, 15))


schedule = t_bond.get_payment_schedule()

values = pd.Series(
    1/(1+3.0/2)**np.arange(1, len(schedule)+1),
    index=schedule['Date'])

yc_obj = YieldCurve(interpolation_values=values,
                    interpolation_method='linear',
                    interpolation_space='discount_factor',
                    interpolation_day_count_convention='act/act',
                    reference_date=datetime.date(2021, 5, 15),
                    left_end_behavior='constant',
                    right_end_behavior='constant')

yc_obj._preprocess_interpolation_values()

