from typing import NamedTuple, Optional, Iterable, Callable
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments import UsTreasuryBond, BondPayment
from fixedIncome.src.assets.base_cashflow import CashflowCollection
from fixedIncome.src.scheduling_tools.holidays import Holiday, US_FEDERAL_HOLIDAYS
from fixedIncome.src.risk.risk_metrics import ONE_BASIS_POINT, ONE_PERCENT
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.scheduling_tools.schedule_enumerations import (PaymentFrequency,
                                                                    BusinessDayAdjustment,
                                                                    DayCountConvention,
                                                                    SettlementConvention,
                                                                    )
from fixedIncome.src.curves.base_curve import DiscountCurve, KnotValuePair
from fixedIncome.src.assets.base_cashflow import CashflowCollection, CashflowKeys, Cashflow, Payment

from dateutil.relativedelta import relativedelta
from datetime import date, timedelta
import scipy  # type: ignore
import bisect
import numpy as np

class CallOption(NamedTuple):
    CallDate: date
    CallPrice: float | int



class FixedRateBond(CashflowCollection):

    def __init__(self,
                 price: float,
                 coupon_rate: float,
                 principal: int,
                 tenor: str,
                 purchase_date: date,
                 maturity_date: date,
                 isin: Optional[str] = None,
                 benchmark_treasury: Optional[UsTreasuryBond] = None,
                 settlement_convention: SettlementConvention = SettlementConvention.T_PLUS_ONE_BUSINESS,
                 payment_frequency: PaymentFrequency = PaymentFrequency.SEMI_ANNUAL,
                 day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_OVER_ACTUAL,
                 business_day_adjustment: BusinessDayAdjustment = BusinessDayAdjustment.FOLLOWING,
                 holiday_calendar: dict[str, Holiday] = US_FEDERAL_HOLIDAYS
                 ) -> None:

        self._price = price
        self._coupon_rate = coupon_rate  # in percentage (%)
        self._principal = principal
        self._tenor = tenor
        self._purchase_date = purchase_date
        self._maturity_date = maturity_date
        self._isin = isin
        self._benchmark_treasury = benchmark_treasury
        self._settlement_convention = settlement_convention
        self._payment_frequency = payment_frequency
        self._business_day_adjustment = business_day_adjustment
        self._holiday_calendar = holiday_calendar
        self._day_count_convention = day_count_convention

        self.num_payments_per_year = self.payment_frequency.value  # enumeration is encoded with these values
        self.coupon_payment = self.calculate_coupon_payment()   # coupon payment in USD ($)
        self.adjustment_function = self.create_date_adjustment_function()

        self.dated_date = self.calculate_dated_date()
        coupon_cashflow = self.create_coupon_payments_cashflow()
        principal_repayment_cashflow = self.create_principal_repayment_cashflow()
        super().__init__([coupon_cashflow, principal_repayment_cashflow],
                         [CashflowKeys.COUPON_PAYMENTS, CashflowKeys.SINGLE_PAYMENT])

        self.settlement_date = Scheduler.calculate_settlement_date(purchase_date=self._purchase_date,
                                                                   settlement_convention=self._settlement_convention,
                                                                   holiday_calendar=self._holiday_calendar)
        self.accrued_interest = self.calculate_accrued_interest()
        self.full_price = self.get_full_price()

    @property
    def payment_frequency(self) -> PaymentFrequency:
        return self._payment_frequency

    @property
    def principal(self) -> float | int:
        return self._principal

    @property
    def coupon_rate(self) -> float:
        return self._coupon_rate

    @property
    def maturity_date(self) -> date:
        return self._maturity_date

    @property
    def tenor(self) -> str:
        return self._tenor

    @property
    def business_day_adjustment(self) -> BusinessDayAdjustment:
        return self._business_day_adjustment

    @property
    def holiday_calendar(self) -> dict:
        return self._holiday_calendar

    @property
    def day_count_convention(self) -> DayCountConvention:
        return self._day_count_convention

    @property
    def purchase_date(self) -> date:
        return self._purchase_date

    @property
    def price(self) -> float:
        return self._price



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
                                    if self.is_payment_received(payment, reference_date=curve.reference_date))

        present_value_principal = sum(curve(payment.payment_date) * payment.payment
                                      for payment in self[CashflowKeys.SINGLE_PAYMENT]
                                      if self.is_payment_received(payment, reference_date=curve.reference_date))

        return present_value_coupons + present_value_principal

    def is_payment_received(self, payment: BondPayment, reference_date: Optional[date] = None) -> bool:
        """
        Determines if a given payment will be received by the holder of the us_treasury_instruments
        if they purchase the us_treasury_instruments on the provided reference date.
        If no reference date is provided, it is assumed to be the bond settlement date.
        """
        if reference_date is None:
            settlement_date = self.settlement_date
        else:
            settlement_date = Scheduler.add_business_days(reference_date, 1, self.holiday_calendar)
        return payment.unadjusted_payment_date >= settlement_date



    def calculate_dated_date(self) -> date:
        """
        Calculates the dated date, the date on which interest begins accruing.
        Assumes self.tenor is of the form {int}Y or {int}M for a number of years or months, respectively.
        """
        length_str_slice = slice(-1)
        length = int(self.tenor[length_str_slice])

        match self.tenor[-1]:
            case 'Y':
                dated_date = self.maturity_date - relativedelta(years=length)
            case 'M':
                dated_date = self.maturity_date - relativedelta(months=length)
            case _:
                raise ValueError(f'{self.tenor[-1]} is not a valid tenor unit indicator. '
                                 f'Tenor must be in the form [int]Y or [int]M.')

        return dated_date

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

    def create_principal_repayment_cashflow(self) -> Cashflow:
        """
        Creates a cashflow containing the principal re-payment.
        """
        adjusted_date = self.adjustment_function(self.maturity_date)
        principal_payment = BondPayment(unadjusted_payment_date=self.maturity_date,
                                        payment_date=adjusted_date,
                                        payment=self.principal)
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
                                                                             end_date=self.dated_date + timedelta(1),
                                                                             increment=increment)

            adjusted_coupon_payment_dates = [self.adjustment_function(coupon_date) for coupon_date in unadjusted_coupon_dates]
            coupon_payments = [BondPayment(unadjusted_payment_date=unadj_date, payment_date=adj_date, payment=self.coupon_payment)
                               for unadj_date, adj_date in zip(unadjusted_coupon_dates, adjusted_coupon_payment_dates)]
        else:
            coupon_payments = []

        coupon_payments.sort(key=lambda payment: payment.payment_date)  # Sort by payment date
        return Cashflow(coupon_payments)

    def calculate_accrued_interest(self, reference_date: Optional[date] = None) -> float:
        """
        Computes the amount of accrued interest between the last payment date
        or dated-date and the settlement date.

        Returns a float representing the amount of accrued interest.
        """
        if self.payment_frequency == PaymentFrequency.ZERO_COUPON:
            return 0.0

        settlement_date = self.settlement_date if reference_date is None else Scheduler.calculate_settlement_date(purchase_date=reference_date,
                                                                                                                  settlement_convention=self.settlement_convention,
                                                                                                                  holiday_calendar=self.holiday_calendar)
        if self.settlement_date < self.dated_date:
            raise ValueError(f'Settlement date {self.settlement_date} is before the dated date {self.dated_date}'
                             f' for us_treasury_instruments\n{self}.')

        # Find last payment accrual date.
        # if the first coupon payment date is in the future, reference date is when interest beings accruing
        elif self[CashflowKeys.COUPON_PAYMENTS][0].unadjusted_payment_date >= settlement_date:
            reference_date = self.dated_date
            following_date = self[CashflowKeys.COUPON_PAYMENTS][0].unadjusted_payment_date

        else:
            date_index = bisect.bisect_right([payment.payment_date for payment in self[CashflowKeys.COUPON_PAYMENTS]], self.purchase_date)
            previous_date = self[CashflowKeys.COUPON_PAYMENTS][date_index-1].unadjusted_payment_date
            following_date = self[CashflowKeys.COUPON_PAYMENTS][date_index].unadjusted_payment_date
            reference_date = previous_date

        accrual_period = DayCountCalculator.compute_accrual_length(
            reference_date, self.settlement_date, dcc=self.day_count_convention
        )

        accrual_fraction = accrual_period / DayCountCalculator.compute_accrual_length(
            reference_date, following_date, dcc=self.day_count_convention
        )

        return accrual_fraction * self.coupon_payment

    def get_full_price(self, reference_date: Optional[date] = None) -> float:
        """
        Calculates the full price (also referred to as the 'dirty' or 'invoice'
        price.) Which is equal to the clean market price plus the amount
        of accrued interest.
        """
        reference_date = reference_date if reference_date is not None else self.purchase_date

        if self.accrued_interest is None:
            self.accrued_interest = self.calculate_accrued_interest(reference_date)

        # normalize the amount of accrued interest by the principal amount
        # Put into 100 principal convention.
        normalized_accrued_interest = (self.accrued_interest / self.principal) * 100.0
        self.full_price = self.price + normalized_accrued_interest

        return self.full_price

    def discount_cashflows_by_fixed_rate(self, fixed_rate: float | np.ndarray, reference_date: Optional[date] = None) -> float:
        """
        Discounts the future cashflows
        """
        fixed_rate = float(fixed_rate)
        if reference_date is None:
            reference_date = self.purchase_date

        present_value = 0.0
        # Accumulate discounted coupon cashflows
        for payment in self[CashflowKeys.COUPON_PAYMENTS]:  # Zero-coupon bonds should have empty iterable here
            if self.is_payment_received(payment, reference_date=reference_date):
                accrual = DayCountCalculator.compute_accrual_length(reference_date, payment.payment_date, self.day_count_convention)
                accrual_times_num_payments = accrual * self.num_payments_per_year
                accrual_df = 1 / (1 + fixed_rate / self.num_payments_per_year) ** accrual_times_num_payments
                present_value += accrual_df * payment.payment

        # add discounted principal re-payment
        for payment in self[CashflowKeys.SINGLE_PAYMENT]:
            if self.is_payment_received(payment, reference_date=reference_date):
                accrual = DayCountCalculator.compute_accrual_length(reference_date, payment.payment_date,
                                                                    self.day_count_convention)
                accrual_times_num_payments = accrual * self.num_payments_per_year
                accrual_df = 1 / (1 + fixed_rate / self.num_payments_per_year) ** accrual_times_num_payments
                present_value += accrual_df * payment.payment

        return present_value


    def yield_to_maturity(self, reference_date: Optional[date] = None) -> float:
        """
        Calculates the yield to maturity (YTM) of a us_treasury_instruments, which is defined as the
        single rate for which discounting the us_treasury_instruments's cash flow gives the original price.
        For example, consider a us_treasury_instruments with coupon rate C and Principal and which has three semi-annual payments.

        Price = ((C / 2*100) * Principal) / (1 + YTM / 2*100)           # first coupon payment discounted
              + (C / 2*100) * Principal) /(1 + YTM / 2*100)**2          # Second coupon payment discounted
              + ((C / 2*100 + 1) * Principal) /(1 + YTM / 2*100)**3     # Third coupon payment and principal discounted

        Returns a float which represents the yield to maturity.
        """
        if reference_date is None:
            reference_date = self.purchase_date

        full_price = self.get_full_price(reference_date)
        target_price = full_price * self.principal / 100
        def diff_from_full_price(fixed_rate) -> np.array:
            return np.array([target_price - self.discount_cashflows_by_fixed_rate(fixed_rate, reference_date)])

        solution = scipy.optimize.root(diff_from_full_price,
                                       x0=np.array([0.0]),
                                       tol=1E-8)

        return float(solution['x'])


    def pv_deriv(self, offset: float = 0, ytm: Optional[float] = None) -> float:
        """
        Computes the derivative of the present value with respect to the bond's yield (Yield to Maturity).
        It is numerically calculated as
        ( PV(ytm + offset + ONE_BASIS_POINT/2) - PV(ytm + offset - ONE_BASIS_POINT/2) ) / ONE_BASIS_POINT
        """
        if ytm is None:
            ytm = self.yield_to_maturity()

        bumped_up_pv = self.discount_cashflows_by_fixed_rate(ytm + offset + ONE_BASIS_POINT/2)
        bumped_down_py = self.discount_cashflows_by_fixed_rate(ytm + offset - ONE_BASIS_POINT/2)
        return (bumped_up_pv - bumped_down_py) / ONE_BASIS_POINT

    def macaulay_duration(self, fixed_rate, reference_date: Optional[date] = None) -> float:
        ''' Calculates the Macaulay duration of a bond defined to be the weighted average time until'''

        fixed_rate = float(fixed_rate)
        if reference_date is None:
            reference_date = self.purchase_date

        pv_times_accrual = 0.0
        # Accumulate discounted coupon cashflows
        for payment in self[CashflowKeys.COUPON_PAYMENTS]:  # Zero-coupon bonds should have empty iterable here
            if self.is_payment_received(payment):
                accrual = DayCountCalculator.compute_accrual_length(reference_date, payment.payment_date, self.day_count_convention)
                accrual_times_num_payments = accrual * self.num_payments_per_year
                accrual_df = 1 / (1 + fixed_rate / self.num_payments_per_year) ** accrual_times_num_payments
                pv_times_accrual += accrual_df * payment.payment * accrual

        # add discounted principal re-payment
        for payment in self[CashflowKeys.SINGLE_PAYMENT]:
            if self.is_payment_received(payment):
                accrual = DayCountCalculator.compute_accrual_length(reference_date, payment.payment_date,
                                                                    self.day_count_convention)
                accrual_times_num_payments = accrual * self.num_payments_per_year
                accrual_df = 1 / (1 + fixed_rate / self.num_payments_per_year) ** accrual_times_num_payments
                pv_times_accrual += accrual_df * payment.payment * accrual

        scaled_full_price = self.full_price * self.principal / 100

        return pv_times_accrual/scaled_full_price


    def duration(self) -> float:
        """
        Calculates the DV01 of the bond with respect to its yield to maturity,
        defined to be:
        Duration = -1/P * dP/d ytm
        """
        return -self.pv_deriv() / self.full_price  #TODO: determine if I need to use full price of clean price here


    def convexity(self) -> float:
        """
        calculate the convexity of a us_treasury_instruments, defined as C := 1/P * d^2 P/d^2y
        Reference: Tuckman and Serrat, 4th ed. equation (4.14).
        """
        ytm = self.yield_to_maturity()
        derivative_positive_bump = self.calculate_pv_deriv(offset=ONE_BASIS_POINT/2, ytm=ytm)
        derivative_negative_bump = self.calculate_pv_deriv(offset=-ONE_BASIS_POINT/2, ytm=ytm)
        second_derivative = (derivative_positive_bump - derivative_negative_bump) / ONE_BASIS_POINT
        return second_derivative / self.full_price  # TODO: Check if full price or clean price is needed



class CallableFixedRateBond(FixedRateBond):
    ''' A class representing fixed rate bonds which also include call options. '''
    def __init__(self,
                price: float,
                coupon_rate: float,
                principal: int,
                purchase_date: date,
                maturity_date: date,
                call_schedule: Iterable[CallOption],
                isin: Optional[str] = None,
                settlement_convention: SettlementConvention = SettlementConvention.T_PLUS_ONE_BUSINESS,
                payment_frequency: PaymentFrequency = PaymentFrequency.SEMI_ANNUAL,
                day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_OVER_ACTUAL,
                business_day_adjustment: BusinessDayAdjustment = BusinessDayAdjustment.FOLLOWING,
                holiday_calendar: dict[str, Holiday] = US_FEDERAL_HOLIDAYS) -> None:
        self._price = price
        self._call_schedule = sorted(list(call_schedule) , key=lambda call_option: call_option.CallDate)



class FixedToFloatBond(CashflowCollection):
    pass



class FloatingRateBond(CashflowCollection):
    pass


if __name__ == '__main__':

    corporate_bond = FixedRateBond(
        price=100,
        coupon_rate=5.0,
        principal=1_000_000,
        tenor='10Y',
        purchase_date=date(2026, 1, 1),
        maturity_date=date(2036, 1, 1),
    )

    ytm = corporate_bond.yield_to_maturity()
    print(corporate_bond.macaulay_duration(fixed_rate=ytm))

