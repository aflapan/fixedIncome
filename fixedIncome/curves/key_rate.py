from __future__ import annotations

import bisect
from datetime import date
from typing import Optional, Callable, Union, Iterable

from fixedIncome.utils.day_count_calculator import DayCountCalculator


class KeyRate:
    """
    Instantiates a KeyRate object.
    """

    def __init__(self,
                 day_count_convention: str,
                 key_rate_date: date,
                 prior_key_rate_date: Optional[date] = None,
                 next_key_rate_date: Optional[date] = None) -> None:

        if (prior_key_rate_date is None) and (next_key_rate_date is None):
            raise ValueError(f"Provided prior_key_rate_date and next_key_rate_date are both None Type. "
                             f"Please provide a both a prior_key_rate_date and an next_key_rate_date for any key "
                             f"rate in the middle of the yield curve, a next_key_rate_date for the first key rate on " 
                             f"the yield curve, and a prior_key_rate for the last key rate on the yield curve."
                             )

        self.key_rate_date = key_rate_date
        self.prior_key_rate_date = prior_key_rate_date
        self.next_key_rate_date = next_key_rate_date
        self.adjustment_fxcn = self.create_adjustment_function()
        self.day_count_convention = day_count_convention

    def __eq__(self, other: KeyRate) -> bool:
        """
        Implements equality of key rates based day count convention, key rate dates, prior dates,
        and following dates.
        """
        day_count_conventions_equal = self.day_count_convention == other.day_count_convention
        key_rate_dates_equal = self.key_rate_date == other.key_rate_date
        next_date_equal = self.next_key_rate_date == other.next_key_rate_date
        prior_date_equal = self.prior_key_rate_date == other.prior_key_rate_date

        all_equal = day_count_conventions_equal and key_rate_dates_equal and next_date_equal and prior_date_equal
        return all_equal

    def __le__(self, other: KeyRate) -> bool:
        """
        Implements the less-than-equal-to comparison of Key Rates by comparing key rate dates.
        """
        return self.key_rate_date <= other.key_rate_date

    def __lt__(self, other: KeyRate) -> bool:
        """ Implements the less-than comparison of Key Rates by comparing key rate dates. """

        return self.key_rate_date < other.key_rate_date

    def __gt__(self, other: KeyRate) -> bool:
        """
        Implements the greater-than comparison of Key Rates by comparing key rate dates.
        """
        return self.key_rate_date > other.key_rate_date

    def __call__(self, date_obj: date) -> float:
        """
        Implements a shortcut to calling the key rate adjustment function by allowing
        the user to call the object directly.
        """
        return self.adjustment_fxcn(date_obj)

    def create_adjustment_function(self, bump_amount: float = 0.01):
        """
        returns a function corresponding to the key rate bump.

        bump_amount is a float corresponding to the key rate bump of the yield in percent (%).
        Default value is a basis point, or 0.01%.
        """

        match (self.prior_key_rate_date, self.key_rate_date, self.next_key_rate_date):

            case (None, middle, following): # first key rate

                def adjustment_fxcn(input_date: date) -> float:

                    if input_date <= middle:
                        return bump_amount

                    elif input_date >= following:
                        return 0.0

                    else:
                        time_to_next = DayCountCalculator.compute_accrual_length(
                            input_date, following, self.day_count_convention
                        )

                        total_length = DayCountCalculator.compute_accrual_length(
                            middle, following, self.day_count_convention
                        )

                        return bump_amount * (time_to_next / total_length)

            case (previous, middle, None): # last key rate

                def adjustment_fxcn(input_date: date) -> float :
                    if input_date >= middle:
                        return bump_amount

                    elif input_date <= previous:
                        return 0.0

                    else:
                        time_to_previous = DayCountCalculator.compute_accrual_length(
                            previous, input_date, self.day_count_convention
                        )
                        total_length = DayCountCalculator.compute_accrual_length(
                            previous, middle, self.day_count_convention
                        )

                        return (time_to_previous / total_length) * bump_amount

            case (previous, middle, following): # one of the middle key rates

                def adjustment_fxcn(input_date: date) -> float:

                    if (input_date >= following) or (input_date <= previous):
                        return 0.0

                    elif input_date <= middle:

                        time_since_prior = DayCountCalculator.compute_accrual_length(
                            previous, input_date, self.day_count_convention
                        )

                        total_time = DayCountCalculator.compute_accrual_length(
                            previous, middle, self.day_count_convention
                        )

                        return (time_since_prior / total_time) * bump_amount

                    else:
                        time_to_next = DayCountCalculator.compute_accrual_length(
                            input_date, following, self.day_count_convention
                        )

                        total_time = DayCountCalculator.compute_accrual_length(
                            middle, following, self.day_count_convention
                        )

                        return (time_to_next / total_time) * bump_amount

            case _:
                raise ValueError(f'Case ({self.prior_key_rate_date}, {self.key_rate_date}, {self.next_key_rate_date})'
                                 f' not matched in call to self.create_bump_function method.')

        return adjustment_fxcn

    def set_adjustment_level(self, bump: float) -> None:
        """
        Setter method for bump key rate adjustment. Creates a new adjustment function which
        triangulates from 0, to the provided bump amount, back to 0.

        Sets this new function as the objects adjustment function.
        """
        self.adjustment_fxcn = self.create_adjustment_function(bump_amount=bump)


    def get_adjustment_function(self):
        """ returns the key rate bump function generated by self.create_bump_function. """

        return self.adjustment_fxcn




class KeyRateCollection:
    """
    Object which contains a collection of key rates.
    """

    def __init__(self, key_rates: Iterable[KeyRate]) -> None:
        """ Instantiates a KeyRateCollection object. """
        self.iter_index = 0
        self.key_rates = sorted(key_rates)  # Sorts based on key_rate_date for each KeyRate object

        self.compatible = self._test_key_rate_dates()

        if not self.compatible:
            raise ValueError

        self.adjustment_fxcn = self._create_combined_adjustment_function()

    def __call__(self, date_val: date) -> float:
        """ Shortcut to calling the adjustment function by calling the collection directly. """
        return self.adjustment_fxcn(date_val)

    def __getitem__(self, position):
        """  Retrieves a KeyRate object from the collection. """
        return self.key_rates[position]

    def __len__(self):
        """ Returns the number of KeyRate objects in the collection. """
        return len(self.key_rates)

    def __bool__(self):
        """ Truthy if the list of KeyRates is non-empty and valid (as determined by _test_key_rate_collection). """
        return bool(self.key_rates) and self._test_key_rate_dates()

    def __next__(self):
        """
        Iterates through the sorted list of KeyRate objects.
        """
        try:
            key_rate_obj = self.key_rates[self.iter_index]

        except IndexError:
            raise StopIteration()

        self.iter_index += 1
        return key_rate_obj

    def __iter__(self):
        return self

    def __add__(self, other: Union[KeyRateCollection, KeyRate]) -> KeyRateCollection:
        """
        Builds collections of keys rates by adding the current collection with another one
        to form a new collection or by inserting a new key rate and modifying the collection
        in place.
        Returns a new KeyRateCollection or a reference to self.
        """

        match other:

            case KeyRateCollection(other):  # other is a KeyRateCollection
                try:
                    return self._add_key_rate_collection(other_collection=other)
                except ValueError:
                    return self

            case KeyRate(other):  # other is an individual KeyRate
                try:
                    self._add_key_rate(other_key_rate=other)
                    return self

                except ValueError:
                    return self

            case _:  # other is another type. Raise an error.
                raise TypeError

    def _add_key_rate_collection(self, other_collection: KeyRateCollection) -> KeyRateCollection:
        """
        Private method for adding two KeyRateCollection objects together. Combines the two
        KeyRate seqyences together and forms a
        """

        try:
            original_key_rates = set(self.key_rates)
            other_key_rates = set(other_collection.key_rates)
            all_key_rates = list(original_key_rates | other_key_rates)  # sets used to drop duplicate key rates

            return KeyRateCollection(all_key_rates)

        except ValueError:
            error_str = f'Could not combine key rate collections {self} and {other_collection}.'
            print(error_str)
            return self

    def _add_key_rate(self, other_key_rate: KeyRate) -> None:
        """
        Method for adding an individual key rate

        Optimized to be faster than _add_key_rate_collection, as the self is modified in place.
        """
        # finish this
        bisect.bisect_right()
        pass


    #------------------------------------------------------------------------

    def _test_key_rate_dates(self) -> bool:
        """
        Tests whether the key rate dates provided are compatible with each other.
        Compatibility means that, when sorted, the key rates satisfy the following criteria:

        (1) Prior Date Compatibility: for any key rate, the key rate date is before or equal to the
            subsequent key rate's prior key rate date.

        (2) Following date compatibility: Each key rate's following date is before or equal to the
            subsequent key rate's key rate date.

        (3) If a key rate's following date is strictly before the subsequent key rate's key rate date,
            then it's following date must be before or equal to the subsequent key rate's prior date.
            Otherwise, it is impossible to insert a key rate between them to form a parallel shift.

        If conditions (1), (2), and (3) hold for all but the last key rate (which is an end key rate, so by default
        the conditions hold), then the collection is said to be compatible.

        Note: we do not assume that, for a given key rate, its following date must exactly equal the
        subsequent key rate's key rate date.
        """
        key_rates_compatible: bool = True

        for kr_index, key_rate in enumerate(self.key_rates[:-1]):  # no need to check last Key Rate.
            # Last key rate does not have subsequent key rate to check compatibility with.


            prior_date_compatibility = key_rate.key_rate_date <= self.key_rates[kr_index+1].prior_key_rate_date
            following_date_compatibility = key_rate.next_key_rate_date <= self.key_rates[kr_index+1].key_rate_date

            following_before_next_prior = True

            if key_rate.next_key_rate_date < self.key_rates[kr_index+1].key_rate_date:  # only applies with <
                # Used because otherwise one could not insert a key rate between key_rate and the subsequent
                # one which would yield a parallel shift when summing all functions.
                following_before_next_prior = key_rate.next_key_rate_date <= self.key_rates[kr_index+1].prior_key_rate_date

            compatibility_flag = prior_date_compatibility and following_date_compatibility and following_before_next_prior
            key_rates_compatible = key_rates_compatible and compatibility_flag

            if not key_rates_compatible:
                print(f'Key Rates {key_rate} and {self.key_rates[kr_index+1]} are incompatible.')
                return False

        return key_rates_compatible


    #------------------------------------------------------------------------


    def _create_combined_adjustment_function(self) -> Callable[[date], float]:
        """ creates a key rate adjustment function for the KeyRateCollection object
        by summing the individual key rate adjustment function evaluations."""

        def combined_adjustment_fxcn(date_val: date) -> float:
            return sum(key_rate(date_val) for key_rate in self.key_rates)

        return combined_adjustment_fxcn














