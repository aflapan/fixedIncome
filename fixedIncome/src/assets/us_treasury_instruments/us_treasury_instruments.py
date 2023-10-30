from datetime import date  # type: ignore
from dateutil.relativedelta import relativedelta  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import math  # type: ignore
import scipy  # type: ignore
import bisect
from typing import Optional, TypeAlias, Callable
from dataclasses import dataclass

from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.scheduling_tools.holidays import Holiday, US_FEDERAL_HOLIDAYS
from fixedIncome.src.scheduling_tools.schedule_enumerations import (PaymentFrequency,
                                                                    BusinessDayAdjustment,
                                                                    DayCountConvention,
                                                                    SettlementConvention,
                                                                    ImmMonths)

from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.curves.curve_enumerations import KnotValuePair
from fixedIncome.src.curves.base_curve import DiscountCurve
from fixedIncome.src.assets.base_cashflow import CashflowCollection, CashflowKeys, Cashflow, Payment


ONE_BASIS_POINT = 0.0001
ONE_PERCENT = 0.01

@dataclass
class BondPayment(Payment):
    unadjusted_payment_date: date


class UsTreasuryBond(CashflowCollection):

    def __init__(self,
                 price: float,
                 coupon_rate: float,
                 principal: int,
                 tenor: str,
                 purchase_date: date,
                 maturity_date: date,
                 cusip: Optional[str] = None,
                 settlement_convention: SettlementConvention = SettlementConvention.T_PLUS_ONE_BUSINESS,
                 payment_frequency: PaymentFrequency = PaymentFrequency.SEMI_ANNUAL,
                 day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_OVER_ACTUAL,
                 business_day_adjustment: BusinessDayAdjustment = BusinessDayAdjustment.FOLLOWING,
                 holiday_calendar: dict[str, Holiday] = US_FEDERAL_HOLIDAYS) -> None:

        self.price = price
        self.coupon_rate = coupon_rate  # assumed to be in %
        self.coupon_payment = self.calculate_coupon_payment()   # coupon payment in USD ($)
        self.principal = principal
        self.tenor = tenor
        self.purchase_date = purchase_date
        self.maturity_date = maturity_date
        self.cusip = cusip
        self.settlement_convention = settlement_convention
        self.day_count_convention = day_count_convention
        self.business_day_adjustment = business_day_adjustment
        self.adjustment_function = self.create_date_adjustment_function()

        self.payment_frequency = payment_frequency
        self.holiday_calendar = holiday_calendar

        self.num_payments_per_year = self.payment_frequency.value  # enumeration is encoded with these values
        self.settlement_date = self.calculate_settlement_date()
        self.dated_date = self.calculate_dated_date()

        self.accrued_interest = self.calculate_accrued_interest()
        self.full_price = self.get_full_price()

        coupon_cashflow = self.create_coupon_payments_cashflow()
        principal_repayment_cashflow = self.create_principal_repayment_cashflow()

        super().__init__([coupon_cashflow, principal_repayment_cashflow],
                         [CashflowKeys.COUPON_PAYMENTS.value, CashflowKeys.SINGLE_PAYMENT.value])

    def __repr__(self) -> str:
        string_rep = f'Bond(price={self.price},\ncoupon={self.coupon},\nprincipal={self.principal},\n' \
                     f'tenor={self.tenor},\npurchase_date={self.purchase_date},\nmaturity_date={self.maturity_date},\n' \
                     f'settlement_convention={self.settlement_convention},\npayment_frequency={self.payment_frequency},\n' \
                     f'day_count_convention={self.day_count_convention},\n' \
                     f'business_day_adjustment={self.business_day_adjustment})'

        return string_rep

    #-------------------------------------------------------------------------
    # Interface methods
    def to_knot_value_pair(self) -> KnotValuePair:
        """
        Returns a knot-value pair of the clean price (which does not include
        accrued interest) and the maturity date.
        """
        if len(self[CashflowKeys.COUPON_PAYMENTS]) > 0:
            last_exposure = max(self[CashflowKeys.COUPON_PAYMENTS][-1].payment_date,
                                self[CashflowKeys.SINGLE_PAYMENT][0].payment_date)
        else:
            last_exposure = self[CashflowKeys.SINGLE_PAYMENT][0].payment_date

        return KnotValuePair(knot=last_exposure, value=self.price)

    def present_value(self, curve: DiscountCurve) -> float:
        """
        Returns the sum of the discounted coupon payments and principal repayment.
        """
        #TODO: Determine if accrued interest should be included and if it should be discounted.
        # YES! Check YieldCurve Factory

        present_value_coupons = sum(curve(payment.payment_date) * payment.payment
                                    for payment in self[CashflowKeys.COUPON_PAYMENTS]
                                    if self.is_payment_received(payment))

        present_value_principal = sum(curve(payment.payment_date) * payment.payment
                                      for payment in self[CashflowKeys.SINGLE_PAYMENT]
                                      if self.is_payment_received(payment))

        return present_value_coupons + present_value_principal

    #--------------------------------------------------------------------------
    # Pricing utility functions
    #TODO: fix this
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

    def calculate_coupon_payment(self) -> float:
        """
        Calculates the coupon payment in USD ($).
        Formula is Principal * Coupon (%) /(100 * number of payments per year).

        Here, the 100 in the denominator converts the coupon from percent (%) into decimal values.
        Number of payments per year is based on the num_payments_per_year.
        Returns a float.
        """

        match self.payment_frequency:

            case PaymentFrequency.ZERO_COUPON:
                return 0

            case _:
                return self.principal * self.coupon_rate / (100 * self.num_payments_per_year)  # coupon_rate is assumed to be in %

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

    def calculate_dated_date(self) -> date:
        """
        Calculates the dated date, the date on which interest begins accruing.
        Assumes self.tenor is of the form {int}Y or {int}M for a number of years or months, respectively.
        """
        length_str_slice = slice(-1)
        length = int(self.tenor[length_str_slice])

        match(self.tenor[-1]):
            case 'Y':
                dated_date = self.maturity_date - relativedelta(years=length)
            case 'M':
                dated_date = self.maturity_date - relativedelta(months=length)
            case _:
                raise ValueError(f'{self.tenor[-1]} is not a valid tenor unit indicator. '
                                 f'Tenor must be in the form [int]Y or [int]M.')

        return dated_date

    #----------------------------------------------------
    # Payment schedules
    def create_date_adjustment_function(self) -> Callable[[date], date]:
        """
        Creates the date adjustment function for adjusting payment days which don't fall on
        a business day. The adjustment used is dictated by the BusinessDayAdjustment.
        """

        match self.business_day_adjustment:
            case BusinessDayAdjustment.FOLLOWING:
                return lambda date_obj: Scheduler.following_date_adjustment(date_obj,
                                                                            holiday_calendar=self.holiday_calendar)
            case BusinessDayAdjustment.MODIFIED_FOLLOWING:
                return lambda date_obj: Scheduler.modified_following_date_adjustment(date_obj,
                                                                                     holiday_calendar=self.holiday_calendar)
            case _:
                raise ValueError(f" Business day adjustment {self.business_day_adjustment} is invalid.")

    def is_payment_received(self, payment: BondPayment, reference_date: Optional[date] = None) -> bool:
        """
        Determines if a given payment will be received by the holder of the us_treasury_instruments
        if they purchase the us_treasury_instruments on the provided reference date.
        If no reference date is provided, it is assumed to be the bond settlement date.
        """
        if reference_date is None:
            settlement_date = self.settlement_date
        else:
            settlement_date = Scheduler.add_business_days(reference_date, 1, US_FEDERAL_HOLIDAYS)
        return payment.unadjusted_payment_date >= settlement_date

    def create_principal_repayment_cashflow(self) -> Cashflow:
        """
        Creates a cashflow containing the principal re-payment.
        """
        adjusted_date = self.adjustment_function(self.maturity_date)
        principal_payment = Payment(payment_date=adjusted_date, payment=self.principal)
        return Cashflow([principal_payment])

    def create_coupon_payments_cashflow(self) -> Cashflow:
        """
        Creates the coupon payments cashflow containing all coupon payments in the Bond's history, even
        past ones before the settlement date.
        """

        match self.payment_frequency:
            case PaymentFrequency.ZERO_COUPON:
                increment = None

            case PaymentFrequency.QUARTERLY:
                increment = relativedelta(months=-3)

            case PaymentFrequency.SEMI_ANNUAL:
                increment = relativedelta(months=-6)

            case PaymentFrequency.ANNUAL:
                increment = relativedelta(years=-1)

            case _:
                raise ValueError(f"{self.payment_frequency} is not a valid payment frequency.")

        if increment is not None:
            unadjusted_coupon_dates = Scheduler.generate_dates_by_increments(start_date=self.maturity_date,
                                                                             end_date=self.dated_date,
                                                                             increment=increment)

            adjusted_coupon_payment_date = [self.adjustment_function(coupon_date) for coupon_date in unadjusted_coupon_dates]
            coupon_payments = [Payment(payment_date=date_obj, payment=self.coupon_payment) for date_obj in adjusted_coupon_payment_date]
        else:
            coupon_payments = []

        coupon_payments.sort()  # Sort by payment date
        return Cashflow(coupon_payments)

    def calculate_settlement_date(self) -> date:
        """
        Method to compute the settlement date based on the purchase date and the settlement_convention.
        """
        match self.settlement_convention:
            case SettlementConvention.T_PLUS_ZERO_BUSINESS:
                return Scheduler.add_business_days(self.purchase_date,
                                                   business_days=0,
                                                   holiday_calendar=self.holiday_calendar)

            case SettlementConvention.T_PLUS_ONE_BUSINESS:
                return Scheduler.add_business_days(self.purchase_date,
                                                   business_days=1,
                                                   holiday_calendar=self.holiday_calendar)

            case SettlementConvention.T_PLUS_TWO_BUSINESS:
                return Scheduler.add_business_days(self.purchase_date,
                                                   business_days=2,
                                                   holiday_calendar=self.holiday_calendar)

            case SettlementConvention.T_PLUS_THREE_BUSINESS:
                return Scheduler.add_business_days(self.purchase_date,
                                                   business_days=3,
                                                   holiday_calendar=self.holiday_calendar)

            case _:
                raise ValueError(f"Settlement Convention {self.settlement_convention} is invalid.")

    #-------------------------------------------------------------------
    def modified_duration(self, purchase_date: Optional[date] = None) -> float:
        """
        Calculates the modified duration of a us_treasury_instruments give a 1 bp bump in its yield-to-maturity.
        Modified duration is defined to be -1/P * dP/dy, where P is the full price of the us_treasury_instrument and
        y is the yield-to-maturity.
        """
        ytm = self.calculate_yield_to_maturity(purchase_date)
        upper_bumped_ytm = ytm + ONE_BASIS_POINT/2
        upper_bumped_price = self.calculate_present_value_for_fixed_yield(yield_rate=upper_bumped_ytm,
                                                                          purchase_date=purchase_date)
        lower_bumped_ytm = ytm - ONE_BASIS_POINT/2
        lower_bumped_price = self.calculate_present_value_for_fixed_yield(yield_rate=lower_bumped_ytm,
                                                                          purchase_date=purchase_date)
        price_deriv = (upper_bumped_price - lower_bumped_price)/ONE_BASIS_POINT
        return -price_deriv/self.full_price

    #TODO: fix this
    def convexity(self, purchase_date: Optional[date] = None) -> float:
        """

        """

        ytm = self.calculate_yield_to_maturity(purchase_date)
        upper_bumped_ytm = ytm + ONE_BASIS_POINT / 2
        upper_bumped_price = self.calculate_present_value_for_fixed_yield(yield_rate=upper_bumped_ytm,
                                                                          purchase_date=purchase_date)
        lower_bumped_ytm = ytm - ONE_BASIS_POINT / 2
        lower_bumped_price = self.calculate_present_value_for_fixed_yield(yield_rate=lower_bumped_ytm,
                                                                          purchase_date=purchase_date)
        price_deriv = (upper_bumped_price - lower_bumped_price) / ONE_BASIS_POINT
        return -price_deriv / self.full_price

    #--------------------------------------------------------------------
    # Pricing Methods

    #TODO: Re-work these
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

        assert all(self.payment_schedule.index == reinvestment_rates.index)

        def num_accrual_periods_left(payment_date: date) -> int:
            """ Helper to return the number of acrrual periods left
            between a provided payment date and the maturity date
            of the us_treasury_instruments.
            """

            diff_in_years = self.maturity_date.year - payment_date.year
            diff_in_months = self.maturity_date.month - payment_date.month
            return 2 * diff_in_years + int(math.ceil(diff_in_months/6))

        powers = pd.Series([num_accrual_periods_left(date) for date in self.payment_schedule.index],
                           index=self.payment_schedule.index, name='Number of Acrual Periods')

        growth_factors = (1 + reinvestment_rates / (2 * 100)) ** powers
        return (self.payment_schedule * growth_factors).sum()

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
        """

        num_accrual_periods = len(self.payment_schedule)
        ratio_of_final_to_price = ex_post / self.price  # step (1)
        root_minus_one = ratio_of_final_to_price**(1/num_accrual_periods) - 1  # Step (2)

        return root_minus_one * 2 * 100  # Step (3)
    def calculate_payment_accrual_factors(self, reference_date: date = None) -> pd.Series:
        """
        Calculates the exponent of the discount factor generated from compounding.
        Can be used, along with the spot-rate r, to create discount factors of the form
        1 / (1 + r/payment_freq)^exponent.

        Returns a pandas series whose indices are the payment dates and whose
        entries are the
        """

        if reference_date is None:
            reference_date = self.purchase_date

        received_payments = [self.is_payment_received(payment, reference_date) for payment in self[]]

        exponent_factors = [
            DayCountCalculator.compute_accrual_length(
                start_date=reference_date, end_date=adjusted_date, dcc=self.day_count_convention
            ) * self.num_payments_per_year
            for adjusted_date in self.payment_schedule.loc[received_payments, 'Adjusted Date']
        ]

        return pd.Series(exponent_factors, index=self.payment_schedule.loc[received_payments, 'Adjusted Date'])


    def calculate_present_value_for_fixed_yield(self, yield_rate, purchase_date: date = None) -> float:
        """
        Calculates the present value of treasury bond cash flows
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
                                       self.calculate_present_value_for_fixed_yield(yield_rate, purchase_date)
                                       - self.full_price,
                                       x0=np.array([0.0]),
                                       tol=1E-10)

        return float(solution['x'])


    def calculate_continuously_compounded_yield_to_maturity(
            self, purchase_date: date = None, dcc: DayCountConvention = DayCountConvention.ACTUAL_OVER_ACTUAL
    ) -> float:
        """
        Calculates the continuously-compounding rate in percent (%) for the given day-count-convention
        which results in the market price of the us_treasury_instruments.
        """

        if purchase_date is None:
            purchase_date = self.purchase_date

        # Calculate Exponents used for compounding.
        # exponents are automatically sub-selected to only those payments which will be received
        # by the purchaser of the us_treasury_instruments on purchase_date

        received_payments = self.is_payment_received(purchase_date)

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

        return solution['x'].item()





class UsTreasuryFuture(CashflowCollection):
    def __init__(self,
                 tenor: str,
                 deliverables_basket: set[UsTreasuryBond],
                 purchase_date: date,
                 maturity_month: ImmMonths,
                 cusip: Optional[str] = None,
                 principal: int = 100_000,
                 settlement_convention: SettlementConvention = SettlementConvention.T_PLUS_ONE_BUSINESS) -> None:

        settlement_date = None
        business_days = Scheduler.generate_business_days(from_date=settlement_date,
                                                         to_date=None,
                                                         Holidays=US_FEDERAL_HOLIDAYS)

        mark_to_market_cashflow = Cashflow([Payment(payment_date=business_day, payment=None)
                                            for business_day in business_days])

        super().__init__((mark_to_market_cashflow,), (CashflowKeys.MARK_TO_MARKET,))








#-------------------------------------------------------------------

UsTreasuryInstrument: TypeAlias = UsTreasuryBond | UsTreasuryFuture
