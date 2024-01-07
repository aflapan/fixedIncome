
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import NamedTuple, Optional, Callable
from dataclasses import dataclass
import itertools

from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.scheduling_tools.holidays import Holiday
from fixedIncome.src.assets.base_cashflow import Payment, Cashflow, CashflowKeys, CashflowCollection
from fixedIncome.src.curves.base_curve import Curve
from fixedIncome.src.curves.curve_enumerations import CurveIndex
from fixedIncome.src.scheduling_tools.schedule_enumerations import (BusinessDayAdjustment,
                                                                    SettlementConvention,
                                                                    PaymentFrequency,
                                                                    DayCountConvention)
class SwapAccrual(NamedTuple):
    start_accrual: date
    end_accrual: date
@dataclass
class SwapPayment(Payment):
    unadjusted_payment_date: date

class InterestRateSwap(CashflowCollection):
    def __init__(self,
                 float_index: CurveIndex,
                 fixed_rate: float,
                 notional: int,
                 start_accrual_date: date,
                 end_accrual_date: date,
                 purchase_date: date,
                 floating_leg_payment_frequency: PaymentFrequency,
                 fixed_leg_payment_frequency: PaymentFrequency,
                 float_leg_day_count_convention: DayCountConvention,
                 fixed_leg_day_count_convention: DayCountConvention,
                 holiday_calendar: dict[str, Holiday],
                 payment_delay: SettlementConvention = SettlementConvention.T_PLUS_TWO_BUSINESS,
                 business_day_adjustment: BusinessDayAdjustment = BusinessDayAdjustment.MODIFIED_FOLLOWING
                 ) -> None:

        self._float_index = float_index
        self._fixed_rate = fixed_rate
        self._notional = notional
        self._start_accrual_date = start_accrual_date
        self._end_accrual_date = end_accrual_date
        self._purchase_date = purchase_date
        self._floating_leg_payment_frequency = floating_leg_payment_frequency
        self._fixed_leg_payment_frequency = fixed_leg_payment_frequency
        self._float_leg_day_count_convention = float_leg_day_count_convention
        self._fixed_leg_day_count_convention = fixed_leg_day_count_convention
        self._holiday_calendar = holiday_calendar
        self._payment_delay = payment_delay
        self._payment_delay_integer = self.generate_num_business_days_payment_delay()
        self._business_day_adjustment = business_day_adjustment
        self._date_adjustment_function = self.create_date_adjustment_function()

        cashflows = (self.generate_fixed_leg_cashflow_schedule(),
                     self.generate_fixed_leg_cashflow_schedule())

        cashflow_keys = (CashflowKeys.FIXED_LEG, CashflowKeys.FLOATING_LEG)
        super().__init__(cashflows=cashflows, cashflow_keys=cashflow_keys)

    @property
    def fixed_rate(self) -> float:
        return self._fixed_rate

    @property
    def notioanl(self) -> int:
        return self._notional

    @property
    def start_accrual_date(self) -> date:
        return self._start_accrual_date

    @property
    def end_accrual_date(self) -> date:
        return self._end_accrual_date

    @property
    def purchase_date(self) -> date:
        return self._purchase_date

    @property
    def float_index(self) -> CurveIndex:
        return self._float_index

    @property
    def floating_leg(self) -> Cashflow:
        return self[CashflowKeys.FLOATING_LEG]

    @property
    def fixed_leg(self) -> Cashflow:
        return self[CashflowKeys.FIXED_LEG]

    @property
    def business_day_adjustment(self) -> BusinessDayAdjustment:
        return self._business_day_adjustment

    @property
    def holiday_calendar(self) -> dict[str, Holiday]:
        return self._holiday_calendar

    @property
    def date_adjustment_function(self) -> Callable[[date], date]:
        return self._date_adjustment_function

    @property
    def floating_leg_payment_frequency(self) -> PaymentFrequency:
        return self._floating_leg_payment_frequency

    @property
    def fixed_leg_payment_frequency(self) -> PaymentFrequency:
        return self._fixed_leg_payment_frequency

    @property
    def float_leg_day_count_convention(self) -> DayCountConvention:
        return self._float_leg_day_count_convention

    @property
    def fixed_leg_day_count_convention(self) -> DayCountConvention:
        return self._fixed_leg_day_count_convention

    @property
    def payment_delay(self) -> SettlementConvention:
        return self._payment_delay

    def generate_num_business_days_payment_delay(self) -> int:
        match self.payment_delay:
            case SettlementConvention.T_PLUS_ZERO_BUSINESS:
                return 0

            case SettlementConvention.T_PLUS_ONE_BUSINESS:
                return 1

            case SettlementConvention.T_PLUS_TWO_BUSINESS:
                return 2

            case _:
                raise ValueError(f'Settlement convention {self.payment_delay} for payment delay is invalid.')


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


    def generate_floating_leg_cashflow_schedule(self, payment_frequency: Optional[PaymentFrequency] = None) -> Cashflow:
        """
        Returns
        """
        if payment_frequency is None:
            payment_frequency = self.floating_leg_payment_frequency

        match payment_frequency:
            case PaymentFrequency.ANNUAL:
                increment = relativedelta(years=-1)

            case PaymentFrequency.SEMI_ANNUAL:
                increment = relativedelta(months=-6)

            case PaymentFrequency.QUARTERLY:
                increment = relativedelta(months=-3)
            case _:
                raise ValueError(f'Payment Frequency {payment_frequency} is not valid to generate a floating leg payment schedule.')

        unadjusted_payment_dates = Scheduler.generate_dates_by_increments(start_date=self.end_accrual_date,
                                                                          end_date=self.start_accrual_date,
                                                                          increment=increment)
        unadjusted_payment_dates.reverse()

        if self.start_accrual_date == unadjusted_payment_dates[0]:
            unadjusted_payment_dates.pop(0)

        adjusted_payment_dates = [self.date_adjustment_function(payment_date) for payment_date in unadjusted_payment_dates]

        payment_schedule = [SwapPayment(unadjusted_payment_date=unadjusted_payment_date,
                                        payment_date=adjusted_payment_date,
                                        payment=None) for unadjusted_payment_date, adjusted_payment_date
                            in zip(unadjusted_payment_dates, adjusted_payment_dates)]

        return Cashflow(payment_schedule)


    def generate_fixed_leg_cashflow_schedule(self, payment_frequency: Optional[PaymentFrequency] = None) -> Cashflow:
        if payment_frequency is None:
            payment_frequency = self.fixed_leg_payment_frequency

        match payment_frequency:
            case PaymentFrequency.ANNUAL:
                increment = relativedelta(years=-1)

            case PaymentFrequency.SEMI_ANNUAL:
                increment = relativedelta(months=-6)

            case PaymentFrequency.QUARTERLY:
                increment = relativedelta(months=-3)
            case _:
                raise ValueError(
                    f'Payment Frequency {payment_frequency} is not valid to generate a floating leg payment schedule.')

        unadjusted_payment_dates = Scheduler.generate_dates_by_increments(start_date=self.end_accrual_date,
                                                                          end_date=self.start_accrual_date,
                                                                          increment=increment)
        unadjusted_payment_dates.reverse()

        fixed_leg_payments = []

        for start_accrual, end_accrual in itertools.pairwise(unadjusted_payment_dates):
            accrual = DayCountCalculator.compute_accrual_length(start_accrual,
                                                                end_accrual,
                                                                self.fixed_leg_day_count_convention)
            payment_amount = accrual * self.notioanl * self.fixed_rate
            adjusted_end_accrual = self.date_adjustment_function(end_accrual)


            payment_date = Scheduler.add_business_days(adjusted_end_accrual,
                                                       self._payment_delay_integer,
                                                       self.holiday_calendar)

            fixed_leg_payment = SwapPayment(unadjusted_payment_date=adjusted_end_accrual,
                                            payment_date=payment_date,
                                            payment=payment_amount)

            fixed_leg_payments.append(fixed_leg_payment)

        return Cashflow(fixed_leg_payments)

    def fill_floating_payment_schedule(self, interest_rate_curve: Curve) -> None:
        """
        Fills the floating rate payment legs
        """
        assert interest_rate_curve.index == self.float_index






class OvernightIndexSwap(InterestRateSwap):
    def __init__(self):
        pass


#---------------------------------------------------------------------------

if __name__ == '__main__':
    from fixedIncome.src.scheduling_tools.holidays import US_FEDERAL_HOLIDAYS

    test_libor_swap = InterestRateSwap(
        float_index=CurveIndex.LIBOR_3M,
        fixed_rate=0.055,
        notional=1_000_000,
        start_accrual_date=date(2024, 1, 1),
        end_accrual_date=date(2034, 1, 1),
        purchase_date=date(2024, 1, 1),
        floating_leg_payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        fixed_leg_payment_frequency=PaymentFrequency.QUARTERLY,
        float_leg_day_count_convention=DayCountConvention.ACTUAL_OVER_360,
        fixed_leg_day_count_convention=DayCountConvention.THIRTY_OVER_THREESIXTY,
        holiday_calendar=US_FEDERAL_HOLIDAYS,
        payment_delay=SettlementConvention.T_PLUS_ZERO_BUSINESS,
        business_day_adjustment=BusinessDayAdjustment.MODIFIED_FOLLOWING)

