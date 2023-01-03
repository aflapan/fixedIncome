import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import math
import scipy

import scheduler
import day_count_calculator

class Bond(object):

    def __init__(self,
                 price:float,
                 coupon:float,
                 principal:int,
                 tenor:str,
                 purchase_date:datetime.date,
                 maturity_date: datetime.date,
                 settlement_convention:str = 'T+1',
                 payment_frequency:str = 'semi-annual',
                 day_count_convention:str = 'act/act',
                 business_day_adjustment:str = 'following') -> None:
        """
        Creates an instance of the bond class given the bond specifics detailed below.

        Parameters:
            price: The market quote of the treasury bond. Quoted as per market convention.
            coupon: A float representing the coupon rate in percent (%)
            principal: A int representing the principal amount of the bond, on which coupon payments are made.
            purchase_date: The date on which the purchase the U.S. Treasury Bond is made.
            settlement_convention: A str representing the settlement convention.
                                   Valid strings are 'T+0', 'T+1', and 'T+2'.
            maturity_date: The date on which the bond matures, i.e. the date where the principal is returned.
        """

        self.price = price
        self.coupon = coupon
        self.principal = principal
        self.tenor = tenor
        self.purchase_date = purchase_date
        self.settlement_convention = settlement_convention
        self.payment_frequency = payment_frequency
        self.maturity_date = maturity_date
        self.day_count_convention = day_count_convention
        self.business_day_adjustment = business_day_adjustment

        self.coupon_payment = self._calculate_coupon_payment()                        # coupon payment in USD ($)

        self.scheduler_obj = scheduler.Scheduler(
            self.tenor,
            self.purchase_date,
            self.maturity_date,
            self.settlement_convention,
            self.payment_frequency,
            self.business_day_adjustment
        )

        self.payment_schedule = self.scheduler_obj.get_payment_schedule()
        self._add_payments_to_schedule()

        self.day_count_calculator_obj = day_count_calculator.DayCountCalculator()

    def __repr__(self) -> str:
        string_rep = f'Bond(prince={self.price}, coupon={self.coupon}, principal={self.principal}, ' \
                     f'tenor={self.tenor}, purchase_date={self.purchase_date}, maturity_date={self.maturity_date}, ' \
                     f'settlement_convention={self.settlement_convention}, payment_frequency={self.payment_frequency}, ' \
                     f'day_count_convention={self.day_count_convention}, ' \
                     f'business_day_adjustment={self.business_day_adjustment})'

        return string_rep



    def calculate_yield_to_maturity(self) -> float:
        """
        Calculates the yield to maturity (YTM) of a bond, which is defined as the
        single rate for which discounting the bond's cash flow gives the original price.
        For example, consider a bond with coupon rate C and Principal and which has three semi-annual payments.

        Price = ((C / 2*100) * Principal) / (1 + YTM / 2*100)           # first coupon payment discounted
              + (C / 2*100) * Principal) /(1 + YTM / 2*100)**2          # Second coupon payment discounted
              + ((C / 2*100 + 1) * Principal) /(1 + YTM / 2*100)**3     # Third coupon payment and principal discounted

        Returns a float which represents the yield to maturity.
        """

        def calculate_discounted_cashflow(yield_candidate:np.array) -> np.array:
            """
            Helper Function which computes the discounted cash flow of the
            treasury bond given a fixed yield. Assumes semi-annual
            payments which are in multiples of six months away from today.

            :param
                yield_candidate: np.array containing a single float which represents the yield in percent (%)
            """

            # Calculate all discount factors given the fixed yield
            # df = 1/(1 + yield/2 * 100)**power
            df_from_yield = 1 / (1 + yield_candidate/(2*100))**np.arange(1, len(self.payment_schedule)+1)


            # calculate sum of discounted cash flow
            df_cashflow = np.dot(df_from_yield, self.payment_schedule['Payment ($)'] * 100 /self.principal)

            return df_cashflow


        solution = scipy.optimize.root(lambda yield_candidate: calculate_discounted_cashflow(yield_candidate) - self.price,
                                       x0 = np.array([self.coupon]),
                                       tol = 1E-12)

        return round(solution['x'].item(), 3)


    def calculate_ex_post_given_reinvestment_rates(self, reinvestment_rates:pd.Series) -> float:
        """
        Calculates the final value of the bond when each
        scheduled payment is reinvested at the rate with semi-annual compounding
        until maturity. The dates in the index of reinvestment_rates
        should one-to-one correspond to the dates in self.payment_schedule, as
        that payment is assumed to be reinvested at the matching index's rate.

        ---------------------------------------------------
        Parameters:
            reinvestment_rates: pd.Series of the same length and with the same indices as self.payment_schedule.
                                Values in the series correspond to rates in percent (%).
        Returns:
            ex_post_value: A float corresponding to the final value of the bond after all payments have been
                           made and all coupons reinvested with semi-annual compounding at the provided rates.
        """

        assert (self.payment_schedule.index == reinvestment_rates.index).all()

        def num_accrual_periods_left(payment_date:datetime.date) -> int:
            """ Helper to return the number of acrrual periods left
            between a provided payment date and the maturity date
            of the bond.
            """

            diff_in_years = self.maturity_date.year - payment_date.year
            diff_in_months = self.maturity_date.month - payment_date.month
            return 2 * diff_in_years + int(math.ceil(diff_in_months/6))

        powers = pd.Series([num_accrual_periods_left(date) for date in self.payment_schedule.index],
                           index = self.payment_schedule.index, name='Number of Acrual Periods')

        growth_factors = (1 + reinvestment_rates / (2 * 100)) ** powers

        return round((self.payment_schedule * growth_factors).sum(),3)


    def calculate_spot_rate_from_ex_post(self, ex_post:float) -> float:
        """
        The math is based on solving for the rate r in the equation
        Price (1 + r / 2*100 )**n = ex_post
        where Price is the initial price of the bond, r is the spot rate to be solved for,
        n is the number of accrual periods (also the length of the payment schedule),
        and ex_post the user-given ex post value of the bond after all
        coupon payments and possible reinvestments.

        Steps of the inversion are:

        Price (1 + r / 2*100 )**n = ex_post           # Starting Equation

        (1 + r / 2*100 )**n = ex_post / Price         # Step (1)

        r / 2 * 100 = (ex_post / Price) ** 1/n - 1    # Step (2)

        r = 2 * 100 * ((ex_post / Price) ** 1/n - 1 ) # Step (3)

        #-----------------------------------------------------
        :param ex_post:
        :return:
        spot_rate: float representing the spot rate
        """

        num_accrual_periods = len(self.payment_schedule)

        ratio_of_final_to_price = ex_post / self.price  # step (1)

        root_minus_one = ratio_of_final_to_price**(1/num_accrual_periods) - 1  # Step (2)

        return round(root_minus_one * 2 * 100, 3) # Step 3


    def get_payment_schedule(self) -> pd.DataFrame:
        """
        Returns the payment schedule for the bond.
        """
        return self.payment_schedule

    #----------------------------------------------------
    # Utility functions
    def _calculate_coupon_payment(self) -> float:
        """
        Calculates the coupon payment in USD ($).
        Formula is Principal * Coupon (%) /(100 * number of payments per year).

        Here, the 100 in the denominator converts the coupon from percent (%) into decimal values.
        Number of payments per year is based on payment_frequency. Equals 1 for 'annual',
        2 for 'semi-annual', and 4 for 'quarterly'.

        Returns a float.
        """
        match(self.payment_frequency):

            case 'quarterly':
                return (self.principal * self.coupon/100.0) / 4

            case 'semi-annual':
                return (self.principal * self.coupon/100.0) / 2

            case 'annual':
                return self.principal * self.coupon/100.0

            case _:
                raise ValueError(f'Error in self._calculate_coupon_payment. '
                                 f'{self.payment_frequency} is not a valid payment frequency string.'
                                 f'Valid options are "quarterly", "semi-annual", and "annual".')

    def _add_payments_to_schedule(self) -> None:

        """
        Adds the payment amount to the payment schedule generated by the scheduler_obj.
        Assumes self._calculate_coupon_payment has already been invoked.

        Modifies the payment schedule in place, and so the method does not return a value.
        """
        self.payment_schedule['Payment ($)'] = [self.coupon_payment for _ in range(len(self.payment_schedule))]

        maturity_index = self.payment_schedule[self.payment_schedule['Date Type'] == 'maturity date'].index

        self.payment_schedule.loc[maturity_index, 'Payment ($)'] = self.principal + self.coupon_payment









t_bond = Bond(price=100.6875,
              coupon=2.375,
              principal=100,
              tenor='30Y',
              purchase_date=datetime.date(2021, 5, 20),
              maturity_date=datetime.date(2051, 5, 15))




#ytm = t_bond.calculate_yield_to_maturity()

#schedule = t_bond.get_payment_schedule()

#reinvestment_rates = pd.Series(ytm, index=schedule['Date'], name='Reinvestment Rate')

#reinvest_at_ytm_value = t_bond.calculate_ex_post_given_reinvestment_rates(reinvestment_rates)

#print(t_bond.calculate_spot_rate_from_ex_post(reinvest_at_ytm_value))

#reinvest_at_zero = t_bond.calculate_ex_post_given_reinvestment_rates(
#    pd.Series(0, index=schedule.index, name='Reinvestment Rate')
#)

#print(t_bond.calculate_spot_rate_from_ex_post(reinvest_at_zero))

#reinvest_at_five = t_bond.calculate_ex_post_given_reinvestment_rates(
#    pd.Series(5.0, index = schedule.index, name = 'Reinvestment Rate')
#)

#print(t_bond.calculate_spot_rate_from_ex_post(reinvest_at_five))
