from datetime import date # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import math  # type: ignore
import scipy  # type: ignore
import bisect
from typing import Optional

from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.scheduling_tools.schedule_enumerations import PaymentFrequency, BusinessDayAdjustment
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.assets.base_cashflow import Payment, Cashflow, CashflowCollection, CashflowKeys
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instrument import UsTreasuryInstrument

ONE_BASIS_POINT = 0.01  # a basis point in percent (%) value


class Bond(UsTreasuryInstrument):

    def __init__(self,
                 market_quote: float,
                 coupon: float,
                 principal: int,
                 tenor: str,
                 purchase_date: date,
                 maturity_date: date,
                 cusip: Optional[str] = None,
                 settlement_convention: str = 'T+1 business',
                 payment_frequency: PaymentFrequency = 'semi-annual',
                 day_count_convention: str = 'act/act',
                 business_day_adjustment: BusinessDayAdjustment = BusinessDayAdjustment.FOLLOWING) -> None:
        """
        Creates an instance of the us_treasury_instruments class given the us_treasury_instruments specifics detailed below.

        Parameters:
            price: The market quote of the treasury us_treasury_instruments. Quoted as per market convention.
            coupon: A float representing the coupon rate in percent (%)
            principal: A int representing the principal amount of the us_treasury_instruments, on which coupon payments are made.
            purchase_date: The date on which the purchase the U.S. Treasury Bond is made.
            settlement_convention: A str representing the settlement convention.
                                   Valid strings are 'T+0', 'T+1', and 'T+2'.
            maturity_date: The date on which the us_treasury_instruments matures, i.e. the date where the principal is returned.
        """
        cashflowkeys = [CashflowKeys.FIXED_LEG]


        self.market_quote = market_quote
        self.coupon = coupon
        self.principal = principal
        self.tenor = tenor
        self.purchase_date = purchase_date
        self.cusip = cusip
        self.settlement_convention = settlement_convention
        self.payment_frequency = payment_frequency
        self.maturity_date = maturity_date
        self.day_count_convention = day_count_convention
        self.business_day_adjustment = business_day_adjustment

        self.num_payments_per_year = self._calculate_num_payments_per_year()
        self.coupon_payment = self._calculate_coupon_payment()   # coupon payment in USD ($)

        self.scheduler_obj = Scheduler(
            self.tenor,
            self.purchase_date,
            self.maturity_date,
            self.settlement_convention,
            self.payment_frequency,
            self.business_day_adjustment
        )

        self.payment_schedule = self.scheduler_obj.get_payment_schedule()
        self._add_payments_to_schedule()
        self.dated_date = self.scheduler_obj.dated_date
        self.settlement_date = self.scheduler_obj.settlement_date
        self.accrued_interest = self.calculate_accrued_interest()
        self.full_price = self.get_full_price()
       # self.continuously_compounding_rate = self.calculate_continuously_compounding_rate()

    def __repr__(self) -> str:
        string_rep = f'Bond(price={self.price},\ncoupon={self.coupon},\nprincipal={self.principal},\n' \
                     f'tenor={self.tenor},\npurchase_date={self.purchase_date},\nmaturity_date={self.maturity_date},\n' \
                     f'settlement_convention={self.settlement_convention},\npayment_frequency={self.payment_frequency},\n' \
                     f'day_count_convention={self.day_count_convention},\n' \
                     f'business_day_adjustment={self.business_day_adjustment})'

        return string_rep

    #--------------------------------------------------------------------------
    # Pricing utility functions


    def calculate_accrued_interest(self) -> float:
        """
        Computes the amount of accrued interest between the last payment date
        or dated-date and the settlement date.

        Returns a float representing the amount of accrued interest.
        """

        if self.settlement_date < self.dated_date:

            raise ValueError(f'Settlement date {self.settlement_date} is before the dated date {self.dated_date}'
                             f' for us_treasury_instruments\n{self}.')

        # Find last payment accrual date.
        # if the first coupon payment date is in the future, reference date is when interest beings accruing
        elif self.payment_schedule['Date'].loc[0] >= self.settlement_date:
            reference_date = self.dated_date
            following_date = self.payment_schedule['Date'].loc[0]

        else:
            date_index = bisect.bisect_right(self.payment_schedule['Date'], self.purchase_date)
            previous_date = self.payment_schedule['Date'].loc[date_index-1]
            following_date = self.payment_schedule['Date'].loc[date_index]
            reference_date = previous_date

        accrual_period = DayCountCalculator.compute_accrual_length(
            reference_date, self.settlement_date, dcc=self.day_count_convention
        )

        accrual_fraction = accrual_period / DayCountCalculator.compute_accrual_length(
            reference_date, following_date, dcc=self.day_count_convention
        )

        return accrual_fraction * self.coupon_payment

    def get_full_price(self) -> float:
        """
        Calculates the full price (also referred to as the 'dirty' or 'invoice'
        price.) Which is equal to the clean market price plus the amount
        of accrued interest.
        """
        if self.accrued_interest is None:
            self.accrued_interest = self.calculate_accrued_interest()

        # normalize the amount of accrued interest by the principal amount
        # Put into 100 principal convention.

        normalized_accrued_interest = (self.accrued_interest / self.principal) * 100.0

        self.full_price = self.price + normalized_accrued_interest

        return self.full_price

    #---------------------------------------------------------------------------
    #
    def calculate_payment_accrual_factors(self, purchase_date: date = None) -> pd.Series:
        """
        Calculates the exponent of the discount factor generated from compounding.
        Can be used, along with the spot-rate r, to create discount factors of the form
        1 / (1 + r/payment_freq)^exponent.

        Returns a pandas series whose indices are the payment dates and whose
        entries are the
        """

        if purchase_date is None:
            purchase_date = self.purchase_date

        received_payments = self._is_payment_received(purchase_date)

        exponent_factors = [
            DayCountCalculator.compute_accrual_length(
                start_date=purchase_date, end_date=adjusted_date, dcc=self.day_count_convention
            ) * self.num_payments_per_year
            for adjusted_date in self.payment_schedule.loc[received_payments, 'Adjusted Date']
        ]

        return pd.Series(exponent_factors, index=self.payment_schedule.loc[received_payments, 'Adjusted Date'])


    #---------------------------------------------------------------------------
    # Yield Calculations

    def _is_payment_received(self, purchase_date: Optional[date] = None) -> pd.Series:
        """
        Determines if each payment will be received by the holder of the us_treasury_instruments
        if they purchase the us_treasury_instruments on the provided purchase date. Returns a pandas
        series of boolean entries whose indices are the same as
        self.payment_schedule.
        """
        if purchase_date is None:
            purchase_date = self.purchase_date

        settlement_date = self.scheduler_obj.add_new_york_business_days(purchase_date, 1)
        return self.payment_schedule['Date'] >= settlement_date


    def calculate_present_value_for_fixed_yield(self, yield_rate, purchase_date: date = None) -> float:
        """
        Calculates the present value of us_treasury_instruments cash flows
        when discount factors are constructed from compounding
        at the provided rate.

        Discount factors have the form
        df = 1/ (1 + yield / (100 * num_payments))^Time
        """

        if purchase_date is None:
            purchase_date = self.purchase_date

        # Calculate Exponents used for compounding.
        # exponents is automatically sub-selected to only those payments which will be received
        # by the purchaser of the us_treasury_instruments on purchase_date

        time_to_payments = self.calculate_payment_accrual_factors(purchase_date)  # pd.Series
        payment_flag = self.payment_schedule['Adjusted Date'].isin(time_to_payments.index)

        future_payment_amounts = self.payment_schedule.loc[
            payment_flag, ['Adjusted Date', 'Payment ($)']].set_index('Adjusted Date', drop=True)

        # dividing by 100 converts out of percent
        discount_factors = 1 / (1 + yield_rate / (100 * self.num_payments_per_year)) ** time_to_payments

        # multiply each payment by its corresponding discount factor and sum
        discounted_cashflow = discount_factors.dot(future_payment_amounts).item()

        return discounted_cashflow


    def calculate_yield_to_maturity(self, purchase_date: date = None) -> float:
        """
        Calculates the yield to maturity (YTM) of a us_treasury_instruments, which is defined as the
        single rate for which discounting the us_treasury_instruments's cash flow gives the original price.
        For example, consider a us_treasury_instruments with coupon rate C and Principal and which has three semi-annual payments.

        Price = ((C / 2*100) * Principal) / (1 + YTM / 2*100)           # first coupon payment discounted
              + (C / 2*100) * Principal) /(1 + YTM / 2*100)**2          # Second coupon payment discounted
              + ((C / 2*100 + 1) * Principal) /(1 + YTM / 2*100)**3     # Third coupon payment and principal discounted

        Returns a float which represents the yield to maturity.
        """
        if purchase_date is None:
            purchase_date = self.purchase_date


        # Check this in the case of zero-coupon us_treasury_instruments
        solution = scipy.optimize.root(lambda yield_rate:
                                       self.calculate_present_value_for_fixed_yield(yield_rate, purchase_date) - self.full_price,
                                       x0=np.array([0.0]),
                                       tol=1E-10)

        return solution['x'].item()


    def calculate_continuously_compounded_yield_to_maturity(self, purchase_date: date = None,
                                                dcc: str = 'act/act') -> float:
        """
        Calculates the continuously-compounding rate in percent (%) for the given day-count-convention
        which results in the market price of the us_treasury_instruments.
        """

        if purchase_date is None:
            purchase_date = self.purchase_date

        # Calculate Exponents used for compounding.
        # exponents are automatically sub-selected to only those payments which will be received
        # by the purchaser of the us_treasury_instruments on purchase_date

        received_payments = self._is_payment_received(purchase_date)

        time_to_payments = pd.Series(
            [
                DayCountCalculator.compute_accrual_length(
                    start_date=purchase_date, end_date=adjusted_date, dcc=dcc
                )
                for adjusted_date in self.payment_schedule.loc[received_payments, 'Adjusted Date']],
            index=self.payment_schedule.loc[received_payments, 'Adjusted Date']
        )

        future_payment_amounts = self.payment_schedule.loc[
            received_payments, ['Adjusted Date', 'Payment ($)']].set_index('Adjusted Date', drop=True).iloc[:, 0]

        # Set discounted cash flows equal to full market price
        solution = scipy.optimize.root(lambda y: future_payment_amounts.dot(np.exp(-y * time_to_payments)) - self.full_price,
                                       x0=np.array([0.05]),
                                       tol=1E-10)

        return solution['x'].item()*100 # Multiply by 100 to convert to %


    def modified_duration(self, purchase_date: Optional[date] = None) -> float:
        """
        Calculates the modified duration of a us_treasury_instruments give a 1 bp bump in its yield-to-maturity.
        Modified duration is defined to be -1/P * dP/dy, where P is the full priuce of the us_treasury_instruments and
        y is the yield-to-maturity.
        """
        ytm = self.calculate_yield_to_maturity(purchase_date)

        upper_bumped_ytm = ytm + self.ONE_BASIS_POINT/2
        upper_bumped_price = self.calculate_present_value_for_fixed_yield(yield_rate=upper_bumped_ytm,
                                                                          purchase_date=purchase_date)

        lower_bumped_ytm = ytm - self.ONE_BASIS_POINT/2
        lower_bumped_price = self.calculate_present_value_for_fixed_yield(yield_rate=lower_bumped_ytm,
                                                                          purchase_date=purchase_date)

        price_deriv = (upper_bumped_price - lower_bumped_price)/self.ONE_BASIS_POINT
        return -price_deriv/self.full_price



    #-----------------------------------------------------------------------------
    # Coupon calculations given yield to maturity
    def calculate_coupon_from_ytm(self, yield_to_maturity) -> None:

        pass

    #---------------------------------------------------------------------------

    # Come back to fix this
    def calculate_ex_post_given_reinvestment_rates(self, reinvestment_rates:pd.Series) -> float:
        """
        Calculates the final value of the us_treasury_instruments when each
        scheduled payment is reinvested at the rate with semi-annual compounding
        until maturity. The dates in the index of reinvestment_rates
        should one-to-one correspond to the dates in self.payment_schedule, as
        that payment is assumed to be reinvested at the matching index's rate.

        ---------------------------------------------------
        Parameters:
            reinvestment_rates: pd.Series of the same length and with the same indices as self.payment_schedule.
                                Values in the series correspond to rates in percent (%).
        Returns:
            ex_post_value: A float corresponding to the final value of the us_treasury_instruments after all payments have been
                           made and all coupons reinvested with semi-annual compounding at the provided rates.
        """

        assert (self.payment_schedule.index == reinvestment_rates.index).all()

        def num_accrual_periods_left(payment_date: date) -> int:
            """ Helper to return the number of acrrual periods left
            between a provided payment date and the maturity date
            of the us_treasury_instruments.
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
        where Price is the initial price of the us_treasury_instruments, r is the spot rate to be solved for,
        n is the number of accrual periods (also the length of the payment schedule),
        and ex_post the user-given ex post value of the us_treasury_instruments after all
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



    #---------------------------------------------------------------------------
    # Payment Utility functions

    def _add_payments_to_schedule(self) -> None:

        """
        Adds the cashflow payments as a column to the payment schedule generated by the scheduler_obj.
        This is a pre-processing step to be called by other methods, and it should not be called by the user explicitly.
        Assumes self._calculate_coupon_payment has already been invoked.

        Modifies the payment schedule in place, and so the method does not return a value.
        """

        self.payment_schedule['Payment ($)'] = [self.coupon_payment for _ in range(len(self.payment_schedule))]

        maturity_index = self.payment_schedule[self.payment_schedule['Date Type'] == 'maturity date'].index

        self.payment_schedule.loc[maturity_index, 'Payment ($)'] = self.principal + self.coupon_payment


    def get_payment_schedule(self) -> pd.DataFrame:
        """
        Returns the payment schedule for the us_treasury_instruments as a pd.DataFrame.
        """
        return self.payment_schedule


    def _calculate_num_payments_per_year(self) -> int:
        """
        Returns an integer representing the number of coupon payments
        made by the us_treasury_instruments per year. Dependent on the provided string
        `payment_frequency', and the results table is

        'zero-coupon' -> 0
        `annual` -> 1
        'semi-annual' -> 2
        'quarterly' -> 4
        """

        match self.payment_frequency:
            case 'zero-coupon':
                return 0

            case 'annual':
                return 1

            case 'semi-annual':
                return 2

            case 'quarterly':
                return 4

            case _:
                raise ValueError(f'{self.payment_frequency} is not a valid payment frequency.')


    def _calculate_coupon_payment(self) -> float:
        """
        Calculates the coupon payment in USD ($).
        Formula is Principal * Coupon (%) /(100 * number of payments per year).

        Here, the 100 in the denominator converts the coupon from percent (%) into decimal values.
        Number of payments per year is based on the num_payments_per_year calculated in
        self._calc_num_payments_per_year.

        Returns a float.
        """

        match self.payment_frequency:

            case 'zero-coupon':
                return 0

            case _:
                return self.principal * self.coupon / (100 * self.num_payments_per_year)



