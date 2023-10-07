"""
This file contains the unit tests for the KeyRate and KeyRateCollection objects
found in fixedIncome.src.curves.key_rates.py
"""

from datetime import date
from random import shuffle
from copy import deepcopy
import datetime

import pandas as pd  # type: ignore
import pytest

from fixedIncome.src.curves.key_rate import KeyRate, KeyRateCollection
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountConvention

#----------------------------------------------------------------------
# Construct the objects to test

purchase_date = date(2023, 2, 27)

four_wk_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                     key_rate_date=date(2023, 3, 28),
                     prior_date=None,
                     next_date=date(2024, 2, 22))


one_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                    key_rate_date=date(2024, 2, 22),
                    prior_date=date(2023, 3, 28),
                    next_date=date(2025, 2, 28))

two_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                    key_rate_date=date(2025, 2, 28),
                    prior_date=date(2024, 2, 22),
                    next_date=date(2026, 2, 15))

three_year_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                        key_rate_date=date(2026, 2, 15),
                        prior_date=date(2025, 2, 28),
                        next_date=date(2030, 2, 28))

seven_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                      key_rate_date=date(2030, 2, 28),
                      prior_date=date(2026, 2, 15),
                      next_date=date(2033, 2, 15))

ten_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                    key_rate_date=date(2033, 2, 15),
                    prior_date=date(2030, 2, 28),
                    next_date=date(2043, 2, 15))

twenty_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                       key_rate_date=date(2043, 2, 15),
                       prior_date=date(2033, 2, 15),
                       next_date=date(2053, 2, 15))

thirty_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                       key_rate_date=date(2053, 2, 15),
                       prior_date=date(2043, 2, 15),
                       next_date=None)


key_rate_list = [four_wk_kr, one_yr_kr, two_yr_kr, three_year_kr, seven_yr_kr, ten_yr_kr, twenty_yr_kr, thirty_yr_kr]
kr_collection = KeyRateCollection(key_rate_list)

PASS_THRESH = 1e-8

#--------------------------------------------------------------------------

def test_KeyRate_equality():
    """
    Tests the equality of key rates by comparing a collection with a deepcopy of itself.
    Want to compare all dates in the collection and return True. We do not want
    to compare the memory locations.
    """

    copy_of_key_rate_list = deepcopy(key_rate_list)
    assert(copy_of_key_rate_list == key_rate_list)


def test_KeyRate_TypeError_calling():
    """
    Tests that the KeyRate object throws the correct type error
    when called with a float rather than a datetime.date object.
    """
    input = 1.0  # a float
    expected_err_str = f'Object {input!r} could not be passed into the adjustment function. ' \
                       f'Using callable feature requires a datetime.date object.'

    with pytest.raises(TypeError, match=expected_err_str):
        three_year_kr(input)


#----------------------------------------------------------------------------
# Testing Key Rate Collection

def test_key_rate_collection_date_setting_sets_next_dates_correctly():
    """
    Tests that the _set_dates_in_collection() method sets all next dates
    to equal the following key_rate object's key_rate_date.
    """
    kr_collection._set_dates_in_collection()
    next_dates = [key_rate.next_date for key_rate in kr_collection]
    key_rate_date = [key_rate.key_rate_date for key_rate in kr_collection]
    assert all(next_date == kr_date for next_date, kr_date in zip(next_dates[:-1], key_rate_date[1:]))

def test_key_rate_collection_date_setting_sets_prior_dates_correctly():
    """
    Tests that the _set_dates_in_collection() method sets all prior dates
    to equal the previous key_rate object's key_rate_date.
    """
    kr_collection._set_dates_in_collection()
    previous_dates = [key_rate.prior_date for key_rate in kr_collection]
    key_rate_date = [key_rate.key_rate_date for key_rate in kr_collection]
    assert all(prior_date == kr_date for prior_date, kr_date in zip(previous_dates[1:], key_rate_date[:-1]))



def test_KeyRateCollection_iterator():

    iterator_list = [key_rate_obj for key_rate_obj in kr_collection]
    assert key_rate_list == iterator_list


def test_KeyRateCollection_indexing():

    zero_index_test = (kr_collection[0] == four_wk_kr)
    last_intext_test = (kr_collection[-1] == thirty_yr_kr)
    assert zero_index_test and last_intext_test


def test_KeyRate_less_than_comaprison():

    shuffle(key_rate_list)
    shuffled_kr_collection = KeyRateCollection(deepcopy(key_rate_list))  # deepcopy to avoid aliasing and having
                                                                         # KeyRateCollection sort original list
    kr_collection_list = [key_rate_obj for key_rate_obj in shuffled_kr_collection]
    assert kr_collection_list == sorted(key_rate_list, key=lambda kr_obj: kr_obj.key_rate_date)


def test_default_KeyRate_adjustment_function_is_default_bump_at_key_rate_date():
    """
    Tests the individual key rate adjustment functions. Default adjustment function moves
    the key_rate_date by 0.01 (1 bp), and then linearly interpolates to being 0 at both the
    prior and next key rate dates if they exist (otherwise, the key rate evaluates to 1).
    """

    interpolation_values = [kr_obj(kr_obj.key_rate_date) for kr_obj in key_rate_list]
    assert all(abs(val - 0.01) < PASS_THRESH for val in interpolation_values)


def test_left_dates_adjust_function_evaluation_for_key_rate_with_prior_None():
    """
    Tests whether dates to the left of the key rate date all evaluate to default bump level
    when prior date is None.
    """

    four_wk_kr.set_prior_date(None)
    test_dates = pd.date_range(start=date(2000, 1, 1), end=four_wk_kr.key_rate_date)  # creates timestamps
    assert all(abs(four_wk_kr(date.date()) - 0.01) < PASS_THRESH for date in test_dates)


def test_right_dates_adjust_function_evaluation_for_key_rate_with_next_None():
    """
    Tests whether dates to the right of the key rate date all evaluate to default bump level
    when next date is None.
    """

    thirty_yr_kr.set_next_date(None)
    test_dates = pd.date_range(start=thirty_yr_kr.key_rate_date, end=date(2100, 1, 1))  # creates timestamps
    assert all(abs(thirty_yr_kr(date.date()) - 0.01) < PASS_THRESH for date in test_dates)


def test_KeyRateCollection_adjustment_function_is_parallel_shift():
    """ Tests that the adjustment function for the KeyRateCollection object
    is a parallel shift. The key rate list key_rate_list was constructed
    so that prior key rates align with subsequent key rate dates, and that
    key rate dates align with subsequent prior dates. Hence, the sum of all
    the default adjustment functions should be a uniform 0.01 (1 bp) bump across
    all dates.
    """

    kr_collection._set_dates_in_collection()
    test_dates = pd.date_range(start=date(2000, 1, 1), end=date(2100, 1, 1))
    collection_vals = (kr_collection(date_val.date()) for date_val in test_dates)
    assert all(abs(val - 0.01) < PASS_THRESH for val in collection_vals)


def test_KeyRateCollection_addition_of_halves_gives_collection():
    """ Tests that splitting the set of key rates, forming individual
    collections, and then summing them, gives an equal KeyRateCollection
    object as forming the collection on the entire set of KeyRates.
    """

    kr_collection_front = KeyRateCollection(key_rate_list[:4])
    kr_collection_back = KeyRateCollection(key_rate_list[4:])

    sum_key_rate_collection = kr_collection_front + kr_collection_back

    assert kr_collection == sum_key_rate_collection

def test_KeyRateCollection_addition_with_KeyRate_makes_valid_collection():
    """
    Tests whether addition is implemented correctly by adding the first individual KeyRate,
    with a KeyRateCollection constructed from the remaining list. Tests whether the result KeyRateCollection
    is valid.
    """

    (first_kr, *rest_kr) = key_rate_list
    rest_kr_collection = KeyRateCollection(rest_kr)
    sum_collection = rest_kr_collection + KeyRateCollection([first_kr])

    assert sum_collection

def test_KeyRateCollection_addition_with_KeyRate_gives_collection():
    """
    Tests that splitting the set of key rates, forming an individual
    KeyRate and a collection from the remaining KeyRates, and then summing them,
    gives an equal KeyRateCollection object as forming the collection on the
    original set of all KeyRates.
    """

    (first_kr, *rest_kr) = key_rate_list
    rest_kr_collection = KeyRateCollection(rest_kr)
    sum_collection = rest_kr_collection + KeyRateCollection([first_kr])

    assert sum_collection == kr_collection

def test_KeyRateCollection_slicing__returns_KeyRateCollection():
    """
    Tests if slicing a KeyRateCollection returns another
    instance of a KeyRateCollection.
    """

    sliced_kr_collection = kr_collection[0:3]
    assert isinstance(sliced_kr_collection, KeyRateCollection)

def test_KeyRateCollection_index_gives_KeyRate():
    """
    Tests if indexing the KeyRateCollection by a single indexer returns a KeyRate object.
    """
    num_key_rates = len(kr_collection)

    # Want to index collection, will test iteration separately.
    assert all(isinstance(kr_collection[index], KeyRate) for index in range(num_key_rates))



