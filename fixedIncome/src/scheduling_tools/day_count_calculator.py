import datetime

class DayCountCalculator(object):

    #--------------------------------------------------------------------
    # accrual day count conventions

    @staticmethod
    def compute_accrual_length(start_date:datetime.date, end_date:datetime.date, dcc:str) -> float:
        """
        Method to compute the time between start_date and end_date given the day count convention specified
        by the dcc string. Returns a float representing the number of years.

        Parameters:
            start_date: a datetime date obejct to specify the start of the accrual period.
            end_date: a datetime date object to specify the end of the accrual period.
            dcc: a string specifying the day count convention to use. Valid values are 'act/360' for
                 actual over 360, 'act/365' for actual over 365', '30/360' for 30 over 360, and
                 'act/act' for actual over actual.

        Returns:
            A float representing the time, as a proportion of 1 year, between start_date and end_date.
        """

        DayCountCalculator.check_dates(start_date=start_date, end_date=end_date)

        match dcc:
            case "act/360":
                return DayCountCalculator._dcc_act_over_360(start_date=start_date, end_date=end_date)

            case "act/365":
                return DayCountCalculator._dcc_act_over_365(start_date=start_date, end_date=end_date)

            case "30/360":
                return DayCountCalculator._dcc_30_over_360(start_date=start_date, end_date=end_date)

            case "act/act":
                return DayCountCalculator._dcc_act_over_act(start_date=start_date, end_date=end_date)

            case _:
                raise TypeError(f"Day count convention string {dcc} is invalid. "
                                f"Acceptable types are 'act/360', 'act/365', '30/360', and 'act/act'.")

    #-----------------------------------------------------------------------
    # Implementations

    @staticmethod
    def _dcc_act_over_360(start_date:datetime.date, end_date:datetime.date) -> float:
        """ Method to compute accrual periods based on actual over 360 day count convention.
        Takes the actual number of days between start_date and end_date and divides it by 360."""

        return (end_date - start_date).days / 360.0

    @staticmethod
    def _dcc_act_over_365(start_date:datetime.date, end_date:datetime.date) -> float:
        """ Method to compute accrual periods based on actual over 360 day count convention.
        Takes the actual number of days between start_date and end_date and divides it by 365."""

        return (end_date - start_date).days / 365.0

    @staticmethod
    def _dcc_30_over_360(start_date:datetime.date, end_date:datetime.date) -> float:
        """
        Method to compute accrual periods based on 30/360 day count conventions.

        This calculation makes a series of adjustments to start_Date and end_date according to
        https://en.wikipedia.org/wiki/Day_count_convention#30/360_US

        (1) If the investment is EOM and (Date1 is the last day of February) and (Date2 is the last day of February),
        then change D2 to 30.

        (2) If the investment is EOM and (Date1 is the last day of February), then change D1 to 30.

        (3) If D2 is 31 and D1 is 30 or 31, then change D2 to 30.

        (4) If D1 is 31, then change D1 to 30.
        """

        # record year, month, and days for later adjustments
        end_year, end_month, end_day = end_date.year, end_date.month, end_date.day
        start_year, start_month, start_day = start_date.year, start_date.month, start_date.day

        # date adjustments in order according to https://en.wikipedia.org/wiki/Day_count_convention#30/360_US

        start_date_last_day_in_feb = (start_month == 2) and (start_day == 28)
        end_date_last_day_in_feb = (end_month == 2) and (end_day == 28)

        # adjustment (1)
        if start_date_last_day_in_feb and end_date_last_day_in_feb:
            end_day = 30

        # adjustment (2)
        if start_date_last_day_in_feb:
            start_day = 30

        # adjsutment (3)
        if end_day == 31 and (start_day == 31 or start_day == 30):
            end_day = 30

        # adjustment (4)
        if start_day == 31:
            start_day = 30

        total_adjusted_days = 360 * (end_year - start_year) + 30 * (end_month - start_month) + (end_day - start_day)
        return total_adjusted_days/360.0

    @staticmethod
    def _dcc_act_over_act(start_date:datetime.date, end_date:datetime.date) -> float:
        """ Method to compute accrual periods based on actual over actual day count convention."""

        # computes the start and end dates of the year containing start_date
        start_date_end_of_this_year = datetime.date(start_date.year, 12, 31) # Always Dec 31st this year
        start_date_end_of_last_year = datetime.date(start_date.year-1, 12, 31) # Always Dec 31st last year

        # computes the start date of the year containing
        end_date_end_of_this_year = datetime.date(end_date.year, 12, 31) # Always Dec 31st this year
        end_date_end_of_last_year = datetime.date(end_date.year-1, 12, 31) # Always Dec 31st last year

        # if both dates lie in same year, can compute fraction directly
        if start_date.year == end_date.year:
            return (end_date - start_date).days / (end_date_end_of_this_year - end_date_end_of_last_year).days

        # compute whole year differences
        whole_year_difference = end_date.year - start_date.year

        # Compute start_date residual fraction of year
        start_residual_fraction = (start_date_end_of_this_year - start_date).days / (start_date_end_of_this_year - start_date_end_of_last_year).days

        # Compute end_date residual fraction of year
        end_residual_fraction = (end_date - end_date_end_of_last_year).days / (end_date_end_of_this_year - end_date_end_of_last_year).days

        return (whole_year_difference - 1) + start_residual_fraction + end_residual_fraction

    #---------------------------------------------------------------------------

    @staticmethod
    def check_dates(start_date:datetime.date, end_date:datetime.date) -> bool:

        start_date_is_date = isinstance(start_date, datetime.date)

        end_date_is_date = isinstance(end_date, datetime.date)

        start_date_before_end_date = start_date <= end_date

        return start_date_is_date and end_date_is_date and start_date_before_end_date


