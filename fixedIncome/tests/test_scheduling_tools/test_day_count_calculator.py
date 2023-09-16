"""
Unit tests for the DayCountCalculator class in day_count_calculator.py
"""
from datetime import date
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator


#-----------------------------------------------------------------------------------------
start_date, end_date = date(2016, 6, 1), date(2016, 9, 1)
THRESH = 1e-12  # need the calculations to be exactly correct


def test_act_act_accrual_periods():
    """ Tests the accrual periods from 2016-6-1 to 2016-9-1 under act/act,convention.
    Reference: Pricing and Trading Interest Rate Derivatives, 3rd ed. by J.H.M Darbyshire
    """
    calc_act_act_accrual = DayCountCalculator.compute_accrual_length(start_date, end_date, 'act/act')
    true_act_act_accrual = 92/366  # taken from reference, page 11

    assert abs(calc_act_act_accrual - true_act_act_accrual) < THRESH


def test_act_360_accrual_periods():
    """ Tests the accrual periods from 2016-6-1 to 2016-9-1 under act/360 convention.
    Reference: Pricing and Trading Interest Rate Derivatives, 3rd ed. by J.H.M Darbyshire
    """
    calc_act_360_accrual = DayCountCalculator.compute_accrual_length(start_date, end_date, 'act/360')
    true_act_360_accrual = 92/360  # Modified ACT/365 value from reference, page 11

    assert abs(calc_act_360_accrual - true_act_360_accrual) < THRESH


def test_act_365_accrual_periods():
    """ Tests the accrual periods from 2016-6-1 to 2016-9-1 under act/365 convention.
    Reference: Pricing and Trading Interest Rate Derivatives, 3rd ed. by J.H.M Darbyshire
    """
    calc_act_365_accrual = DayCountCalculator.compute_accrual_length(start_date, end_date, 'act/365')
    true_act_365_accrual = 92/365  # Taken from reference, page 11

    assert abs(calc_act_365_accrual - true_act_365_accrual) < THRESH


def test_30_360_accrual_periods():
    """ Tests the accrual periods from 2016-6-1 to 2016-9-1 under 30/360 convention.
    Reference: Pricing and Trading Interest Rate Derivatives, 3rd ed. by J.H.M Darbyshire
    """
    calc_30_360_accrual = DayCountCalculator.compute_accrual_length(start_date, end_date, '30/360')
    true_30_360_accrual = 90/360  # Taken from reference, page 11

    assert abs(calc_30_360_accrual - true_30_360_accrual) < THRESH





