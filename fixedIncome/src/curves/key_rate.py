from __future__ import annotations


from collections.abc import MutableSequence
import operator
import bisect
from datetime import date
from typing import Optional, Callable, Union, Iterable

from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator


class KeyRate:
    """
    Instantiates a KeyRate object.
    """
    __bump_val: float = 0.01  # Rate amount ion percent (%) used to create adjustment
                            # functions for key rate bump functions.
                            # Default value is 1 bp, or 0.01%.

    __adjustment_fxcn: Callable[[date], float] = None

    def __init__(self,
                 day_count_convention: str,
                 key_rate_date: date,
                 prior_date: Optional[date] = None,
                 next_date: Optional[date] = None) -> None:

        self.__key_rate_date = key_rate_date
        self.__prior_date = prior_date
        self.__next_date = next_date
        self.__day_count_convention = day_count_convention
        self.create_adjustment_function()  # uses default 1 bp bump

    @property
    def key_rate_date(self):
        return self.__key_rate_date
    @property
    def prior_date(self):
        return self.__prior_date

    @property
    def next_key_rate_date(self):
        return self.__next_date

    @property
    def day_count_convention(self):
        return self.__day_count_convention

    @property
    def adjustment_fxcn(self):
        return self.__adjustment_fxcn

    @property
    def bump_val(self):
        return self.__bump_val


    def __eq__(self, other: KeyRate) -> bool:
        """
        Implements equality of key rates based day count convention and key rate dates.
        Previous date and following date, along with the adjustment function,
        are subject to change. Thus, these are not used for comparison.
        """
        day_count_conventions_equal = self.day_count_convention == other.day_count_convention
        key_rate_dates_equal = self.key_rate_date == other.key_rate_date

        all_equal = day_count_conventions_equal and key_rate_dates_equal
        return all_equal

    def __le__(self, other: KeyRate) -> bool:
        """
        Implements the less-than-or-equal-to comparison of Key Rates by comparing key rate dates.
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

    def __ge__(self, other: KeyRate) -> bool:
        """
        Implements the greater-than-or-equal-to comparison of Key Rates by comparing key rate dates.
        """
        return self.key_rate_date >= other.key_rate_date

    def __call__(self, date_obj: date) -> float:
        """
        Implements a shortcut to calling the key rate adjustment function by allowing
        the user to call the object directly.
        """
        try:
            return self.adjustment_fxcn(date_obj)
        except TypeError:
            raise TypeError(f'Object {date_obj} could not be passed into the adjustment function.'
                            f'Using callable feature requires a datetime.date object.')

    def __hash__(self):
        """
        Implements a hash based on immutable attributes day count convention and the
        key rate date. Prior, and next days are mutable, and so are not used in the construction
        of the has.
        """
        attribute_tuple = (self.day_count_convention, self.key_rate_date)

        return hash(attribute_tuple)

    #-------------------------------------------------------------------
    # Functionality for creating bump functions

    def set_bump_val(self, new_bump_val: Optional[float] = None) -> None:
        """
        Setter method for bump value amount used in adjustment function.
        Sets the value and then re-constructs the adjustment function
        """
        self.__bump_val = new_bump_val
        self.create_adjustment_function()

    def set_prior_date(self, new_prior_date: Optional[date] = None) -> None:
        """
        Setter method to change the prior date for the KeyRateObject which also automatically makes
        the appropriate change to the adjustment function.

        If the prior date is set to None, then the adjustment function is constant at the bump_level
        for all dates less-than-or-equal-to the key rate date.

        If the prior date is set to some value strictly less than the key rate date, then the adjustment
        function is set to interpolation between 0 at the prior date and the current bump value at
        the key rate date.

        If the prior date is set to some value at or greater than the key rate date, ValueError is raised.
        """
        if self.key_rate_date <= new_prior_date:
            raise ValueError('Prior date cannot be at or after key rate date.')

        self.__prior_date = new_prior_date
        self.create_adjustment_function()

    def set_next_date(self, new_next_date: Optional[date] = None) -> None:
        """
        Setter method to change the next date for the KeyRateObject which also automatically makes
        the appropriate change to the adjustment function.

        If the next date is set to None, then the adjustment function is constant at the bump_level
        for all dates greater-than-or-equal-to the key rate date.

        If the next date is set to some value strictly greater than the key rate date, then the adjustment
        function is set to interpolation between bump_val at the key rate date and 0 at
        the next date.

        If the next date is set to some value at or less-than the key rate date, ValueError is raised.
        """
        if new_next_date <= self.key_rate_date:
            raise ValueError('Next date cannot be at or before key rate date.')

        self.__next_date = new_next_date
        self.create_adjustment_function()

    def create_adjustment_function(self) -> None:
        """
        Creates a adjustment function corresponding to key rate bump function
        using the self.__bump_val, self.__prior_date, self.__key_rate_date, self.__next_date, and
        interpolated between dates using the self.__day_count_convention.
        """

        match (self.prior_date, self.key_rate_date, self.next_key_rate_date):

            case (None, middle, following):  # first key rate

                def adjustment_fxcn(input_date: date) -> float:

                    if input_date <= middle:
                        return self.bump_val

                    elif input_date >= following:
                        return 0.0

                    else:
                        time_to_next = DayCountCalculator.compute_accrual_length(
                            input_date, following, self.day_count_convention
                        )

                        total_length = DayCountCalculator.compute_accrual_length(
                            middle, following, self.day_count_convention
                        )

                        return self.bump_val * (time_to_next / total_length)

            case (previous, middle, None):  # last key rate

                def adjustment_fxcn(input_date: date) -> float:
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

                        return (time_to_previous / total_length) * self.bump_val

            case (previous, middle, following):  # one of the middle key rates

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

                        return (time_since_prior / total_time) * self.bump_val

                    else:
                        time_to_next = DayCountCalculator.compute_accrual_length(
                            input_date, following, self.day_count_convention
                        )

                        total_time = DayCountCalculator.compute_accrual_length(
                            middle, following, self.day_count_convention
                        )

                        return (time_to_next / total_time) * self.bump_val

            case (None, middle, None):  # only key rate
                def adjustment_fxcn(input_date: date) -> float:
                    return self.bump_val

            case _:
                raise ValueError(f'Case ({self.prior_date}, {self.key_rate_date}, {self.next_key_rate_date})'
                                 f' not matched in call to self.create_bump_function method.')

        self.__adjustment_fxcn = adjustment_fxcn





class KeyRateCollection(MutableSequence):
    """
    An ordered sequence-type object containing a collection of key rates
    with an adjustment function built from summing the individual adjustment
    functions for each KeyRate object in the collection.
    """

    def __init__(self, key_rates: Iterable[KeyRate]) -> None:
        """ Instantiates a KeyRateCollection object. """
        self.iter_index = 0
        self.key_rates = sorted(list(key_rates))  # Sorts based on key_rate_date for each KeyRate object
        self.compatible = self._test_key_rate_dates()

        if not self.compatible:
            raise ValueError

        self.__adjustment_fxcn = self._create_combined_adjustment_function()

    @property
    def adjustment_fxcn(self):
        return self.__adjustment_fxcn

    def __call__(self, date_val: date) -> float:
        """ Shortcut to calling the adjustment function by calling the KeyRateCollection object directly. """
        return self.adjustment_fxcn(date_val)

    def __getitem__(self, key):
        """  Retrieves a KeyRate object from the collection. """
        if isinstance(key, slice):
            cls = type(self)
            return cls(self.key_rates[key])

        index = operator.index(key)
        return self.key_rates[index]

    def __setitem__(self, key: int, value: KeyRate):
        """
        places a new KeyRateObject
        """
        pass

    def __delitem__(self, key):
        """
        Removes a KeyRate object from the collection by index and re-constructs the adjustment
        function with the remaining KeyRate objects in collection.
        """
        del self.key_rates[key]


        self.__adjustment_fxcn = self._create_combined_adjustment_function()



    def __len__(self):
        """ Returns the number of KeyRate objects in the collection. """
        return len(self.key_rates)

    def __bool__(self):
        """ Truthy if the list of KeyRates is non-empty and valid (as determined by _test_key_rate_collection). """
        return bool(self.key_rates) and self.compatible

    def __next__(self):
        """
        Iterates through the sorted list of KeyRate objects.
        """
        try:
            key_rate_obj = self.key_rates[self.iter_index]

        except IndexError:
            self.iter_index = 0
            raise StopIteration()

        self.iter_index += 1
        return key_rate_obj

    def __iter__(self):
        """ Implements iteration through the KeyRateCollection by returning a generator for the KeyRate objects. """
        return (key_rate for key_rate in self.key_rates)

    def __eq__(self, other: KeyRateCollection) -> bool:
        """
        Implements equality of KeyRateCollection objects by testing for equality of their internal
        list of KeyRate objects.
        """
        return self.key_rates == other.key_rates

    def __add__(self, other: Union[KeyRateCollection, KeyRate]) -> KeyRateCollection:
        """
        Builds collections of keys rates by adding the current collection with another one
        to form a new collection or by inserting a new key rate and modifying the collection
        in place.
        Returns a new KeyRateCollection or a reference to self.
        """

        if isinstance(other, KeyRateCollection):
            try:
                return self._add_key_rate_collection(other_collection=other)
            except ValueError:
                return self

        elif isinstance(other, KeyRate):
            try:
                self._add_key_rate(other_key_rate=other)
                return self

            except ValueError:
                return self

        else:
            raise TypeError(f'Type {other.__class__} cannot be added to KeyRateCollection.')


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

    def _add_key_rate(self, other_key_rate: KeyRate) -> KeyRateCollection:
        """
        Method for adding an individual key rate

        Optimized to be faster than _add_key_rate_collection, as the self is modified in place.
        Returns a reference to self.
        """

        bisect.insort_right(self.key_rates, other_key_rate)
        self._test_key_rate_dates()
        self._create_combined_adjustment_function()  # recreate adjustment function with new KeyRate included

        return self


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

            prior_date_compatibility = key_rate.key_rate_date <= self.key_rates[kr_index+1].prior_date
            following_date_compatibility = key_rate.next_key_rate_date <= self.key_rates[kr_index+1].key_rate_date

            following_before_next_prior = True

            if key_rate.next_key_rate_date < self.key_rates[kr_index+1].key_rate_date:  # only applies with <
                # Used because otherwise one could not insert a key rate between key_rate and the subsequent
                # one which would yield a parallel shift when summing all functions.
                following_before_next_prior = key_rate.next_key_rate_date <= self.key_rates[kr_index+1].prior_date

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














