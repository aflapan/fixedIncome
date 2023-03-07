"""
This file contains the unit tests for the KeyRate and KeyRateCollection objects.
"""

from random import shuffle
from copy import deepcopy
import datetime

import pandas as pd  # type: ignore

from fixedIncome.curves.key_rate import KeyRate, KeyRateCollection

#----------------------------------------------------------------------
# Construct the objects to test

purchase_date = datetime.date(2023, 2, 27)

four_wk_kr = KeyRate(day_count_convention='act/act',
                     key_rate_date=datetime.date(2023, 3, 28),
                     prior_key_rate_date=None,
                     next_key_rate_date=datetime.date(2024, 2, 22))


one_yr_kr = KeyRate(day_count_convention='act/act',
                    key_rate_date=datetime.date(2024, 2, 22),
                    prior_key_rate_date=datetime.date(2023, 3, 28),
                    next_key_rate_date=datetime.date(2025, 2, 28))

two_yr_kr = KeyRate(day_count_convention='act/act',
                    key_rate_date=datetime.date(2025, 2, 28),
                    prior_key_rate_date=datetime.date(2024, 2, 22),
                    next_key_rate_date=datetime.date(2026, 2, 15))

three_year_kr = KeyRate(day_count_convention='act/act',
                        key_rate_date=datetime.date(2026, 2, 15),
                        prior_key_rate_date=datetime.date(2025, 2, 28),
                        next_key_rate_date=datetime.date(2030, 2, 28))

seven_yr_kr = KeyRate(day_count_convention='act/act',
                      key_rate_date=datetime.date(2030, 2, 28),
                      prior_key_rate_date=datetime.date(2026, 2, 15),
                      next_key_rate_date=datetime.date(2033, 2, 15))

ten_yr_kr = KeyRate(day_count_convention='act/act',
                    key_rate_date=datetime.date(2033, 2, 15),
                    prior_key_rate_date=datetime.date(2030, 2, 28),
                    next_key_rate_date=datetime.date(2043, 2, 15))

twenty_yr_kr = KeyRate(day_count_convention='act/act',
                       key_rate_date=datetime.date(2043, 2, 15),
                       prior_key_rate_date=datetime.date(2033, 2, 15),
                       next_key_rate_date=datetime.date(2053, 2, 15))

thirty_yr_kr = KeyRate(day_count_convention='act/act',
                       key_rate_date=datetime.date(2053, 2, 15),
                       prior_key_rate_date=datetime.date(2043, 2, 15),
                       next_key_rate_date=None)


key_rate_list = [four_wk_kr, one_yr_kr, two_yr_kr, three_year_kr, seven_yr_kr, ten_yr_kr, twenty_yr_kr, thirty_yr_kr]
kr_collection = KeyRateCollection(key_rate_list)


#--------------------------------------------------------------------------

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

    pass_thresh = 1e-8
    interpolation_values = [kr_obj(kr_obj.key_rate_date) for kr_obj in key_rate_list]
    assert all([abs(val - 0.01) < pass_thresh for val in interpolation_values])


def test_left_dates_adjust_function_evaluation_for_key_rate_with_prior_None():
    """
    Tests whether dates to the left of the key rate date all evaluate to default bump level
    when prior date is None.
    """

    pass_thresh = 1e-8
    test_dates = pd.date_range(start=datetime.date(2000, 1, 1), end=four_wk_kr.key_rate_date)  # creates timestamps
    assert all([abs(four_wk_kr(date.date()) - 0.01) < pass_thresh for date in test_dates])


def test_right_dates_adjust_function_evaluation_for_key_rate_with_next_None():
    """
    Tests whether dates to the right of the key rate date all evaluate to default bump level
    when next date is None.
    """

    pass_thresh = 1e-8
    test_dates = pd.date_range(start=thirty_yr_kr.key_rate_date, end=datetime.date(2100, 1, 1))  # creates timestamps
    assert all([abs(thirty_yr_kr(date.date()) - 0.01) < pass_thresh for date in test_dates])


def test_KeyRateCollection_adjustment_function_is_parallel_shift():
    """ Tests that the adjustment function for the KeyRateCollection object
    is a parallel shift. The key rate list key_rate_list was constructed
    so that prior key rates align with subsequent key rate dates, and that
    key rate dates align with subsequent prior dates. Hence, the sum of all
    the default adjustment functions should be a uniform 0.01 (1 bp) bump across
    all dates.
    """

    test_dates = pd.date_range(start=datetime.date(2000, 1, 1), end=datetime.date(2100, 1, 1))
    collection_vals = (kr_collection(date_val.date()) for date_val in test_dates)
    pass_thresh = 1e-8
    assert all([abs(val - 0.01) < pass_thresh for val in collection_vals])



