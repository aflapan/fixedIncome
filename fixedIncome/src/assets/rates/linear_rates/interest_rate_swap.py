"""
This script contains classes for a Term interest rate swap and Overnight Interest Rate Swap.

Unit tests contained in tests/test_assets/test_rates/test_linear_rates/test_interest_rate_swap.py
"""
from enum import Enum
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import NamedTuple, Optional, Callable
import itertools
from abc import abstractmethod

from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.scheduling_tools.holidays import Holiday
from fixedIncome.src.assets.base_cashflow import Payment, Cashflow, CashflowKeys, CashflowCollection
from fixedIncome.src.curves.base_curve import Curve, DiscountCurve, KnotValuePair
from fixedIncome.src.curves.curve_enumerations import CurveIndex
from fixedIncome.src.scheduling_tools.schedule_enumerations import (BusinessDayAdjustment,
                                                                    SettlementConvention,
                                                                    PaymentFrequency,
                                                                    DayCountConvention)


class InterestRateSwapDirection(Enum):
    PAYER_FIXED = 0
    RECEIVER_FIXED = 1


class SwapAccrual(NamedTuple):
    start_accrual: date
    end_accrual: date
    fixing_date: Optional[date] = None
    accrual_factor: Optional[float] = None


class FixedToFloatInterestRateSwap(CashflowCollection):
    def __init__(self,
                 float_index: CurveIndex,
                 direction: InterestRateSwapDirection,
                 fixed_rate: float,
                 notional: int,
                 purchase_date: date,
                 settlement_convention: SettlementConvention,
                 tenor: str,
                 floating_leg_payment_frequency: PaymentFrequency,
                 fixed_leg_payment_frequency: PaymentFrequency,
                 floating_leg_day_count_convention: DayCountConvention,
                 fixed_leg_day_count_convention: DayCountConvention,
                 holiday_calendar: dict[str, Holiday],
                 fixing_date_for_accrual_period: SettlementConvention = SettlementConvention.T_MINUS_TWO_BUSINESS,
                 payment_delay: SettlementConvention = SettlementConvention.T_PLUS_TWO_BUSINESS,
                 business_day_adjustment: BusinessDayAdjustment = BusinessDayAdjustment.MODIFIED_FOLLOWING
                 ) -> None:

        self._float_index = float_index
        self._direction = direction
        self._fixed_rate = fixed_rate
        self._notional = notional
        self._settlement_convention = settlement_convention
        self._tenor = tenor
        self._purchase_date = purchase_date
        self._floating_leg_payment_frequency = floating_leg_payment_frequency
        self._fixed_leg_payment_frequency = fixed_leg_payment_frequency
        self._floating_leg_day_count_convention = floating_leg_day_count_convention
        self._fixed_leg_day_count_convention = fixed_leg_day_count_convention
        self._holiday_calendar = holiday_calendar
        self._fixing_date_for_accrual_period = fixing_date_for_accrual_period
        self._payment_delay = payment_delay
        self._business_day_adjustment = business_day_adjustment
        self._date_adjustment_function = self._create_date_adjustment_function()
        self._start_accrual_date, self._end_accrual_date = self.calculate_start_end_accrual_dates()

        cashflows = (None, None)

        cashflow_keys = (CashflowKeys.FIXED_LEG, CashflowKeys.FLOATING_LEG)
        super().__init__(cashflows=cashflows, cashflow_keys=cashflow_keys)

    @property
    def fixed_rate(self) -> float:
        return self._fixed_rate

    @property
    def direction(self) -> InterestRateSwapDirection:
        return self._direction

    @property
    def notional(self) -> int:
        return self._notional

    @property
    def settlement_convention(self) -> SettlementConvention:
        return self._settlement_convention

    @property
    def tenor(self) -> str:
        return self._tenor

    @property
    def purchase_date(self) -> date:
        return self._purchase_date

    @property
    def start_accrual_date(self) -> date:
        return self._start_accrual_date
    @property
    def end_accrual_date(self) -> date:
        return self._end_accrual_date

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
    def fixing_date_for_accrual_period(self) -> SettlementConvention:
        return self._fixing_date_for_accrual_period

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
    def floating_leg_day_count_convention(self) -> DayCountConvention:
        return self._floating_leg_day_count_convention

    @property
    def fixed_leg_day_count_convention(self) -> DayCountConvention:
        return self._fixed_leg_day_count_convention

    @property
    def payment_delay(self) -> SettlementConvention:
        return self._payment_delay


    def calculate_start_end_accrual_dates(self) -> tuple[date, date]:
        """
        Returns a tuple (start, end) of start accrual date and end accrual dates.
        """
        start_accrual = Scheduler.calculate_settlement_date(purchase_date=self.purchase_date,
                                                            settlement_convention=self.settlement_convention,
                                                            holiday_calendar=self.holiday_calendar)

        length_str_slice = slice(-1)
        length = int(self.tenor[length_str_slice])

        match self.tenor[-1]:
            case 'Y':
                end_accrual = start_accrual + relativedelta(years=length)
            case 'M':
                end_accrual = start_accrual + relativedelta(months=length)
            case _:
                raise ValueError(f'{self.tenor[-1]} is not a valid tenor unit indicator. '
                                 f'Tenor must be in the form [int]Y or [int]M.')

        return start_accrual, end_accrual


    # Interface Methods
    def to_knot_value_pair(self) -> KnotValuePair:
        """
        Returns a KnotValuePair which represents the TermIntertestRateSwap
        for the purposes of interest rate curve calibration.

        The knot corresponds to the last exposure date of the swap legs, i.e. the end_accrual_date,
        and the value is the fixed_rate.
        """
        return KnotValuePair(knot=self.end_accrual_date, value=self.fixed_rate)

    def present_value(self, discount_curve: DiscountCurve) -> float:
        """
        Computes the present value
        """
        if self[CashflowKeys.FIXED_LEG] is None or self[CashflowKeys.FLOATING_LEG] is None:
            raise ValueError(f'Fixed leg or floating leg are None, so present value cannot be taken. '
                             f'First apply generate_floating_leg_cashflow or generate_fixed_leg_cashflow with the'
                             f' set_cashflow param set to True.')

        float_pv = discount_curve.present_value(self[CashflowKeys.FLOATING_LEG])
        fixed_pv = discount_curve.present_value(self[CashflowKeys.FIXED_LEG])

        match self.direction:
            case InterestRateSwapDirection.RECEIVER_FIXED:
                return fixed_pv - float_pv

            case InterestRateSwapDirection.PAYER_FIXED:
                return float_pv - fixed_pv

            case _:
                raise ValueError(f'Swap direction {self.direction} is not valid.')


    # Float/Fixed leg methods
    def _create_date_adjustment_function(self) -> Callable[[date], date]:
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


    def generate_floating_leg_accrual_schedule(self) -> list[SwapAccrual]:
        """
        Returns a list of SwapAccrual corresponding to all payments of the floating leg.
        """
        match self.floating_leg_payment_frequency:
            case PaymentFrequency.ANNUAL:
                increment = relativedelta(years=-1)

            case PaymentFrequency.SEMI_ANNUAL:
                increment = relativedelta(months=-6)

            case PaymentFrequency.QUARTERLY:
                increment = relativedelta(months=-3)

            case _:
                raise ValueError(f'Payment Frequency {self.floating_leg_payment_frequency} is not valid to generate a floating leg payment schedule.')

        unadjusted_accrual_dates = Scheduler.generate_dates_by_increments(start_date=self.end_accrual_date,
                                                                          end_date=self.start_accrual_date,
                                                                          increment=increment)
        unadjusted_accrual_dates.reverse()
        adjusted_accrual_dates = [self.date_adjustment_function(accrual_date) for accrual_date in unadjusted_accrual_dates]
        swap_accruals = []

        for start_accrual, end_accrual in itertools.pairwise(adjusted_accrual_dates):
            accrual = DayCountCalculator.compute_accrual_length(start_accrual,
                                                                end_accrual,
                                                                self.floating_leg_day_count_convention)

            swap_accrual = SwapAccrual(start_accrual=start_accrual,
                                       end_accrual=end_accrual,
                                       accrual_factor=accrual)

            swap_accruals.append(swap_accrual)

        return swap_accruals


    def generate_fixed_leg_accrual_schedule(self) -> list[SwapAccrual]:
        """
        Returns a list of SwapAccrual corresponding to all payments of the fixed leg.
        """

        match self.fixed_leg_payment_frequency:
            case PaymentFrequency.ANNUAL:
                increment = relativedelta(years=-1)

            case PaymentFrequency.SEMI_ANNUAL:
                increment = relativedelta(months=-6)

            case PaymentFrequency.QUARTERLY:
                increment = relativedelta(months=-3)
            case _:
                raise ValueError(
                    f'Payment Frequency {self.fixed_leg_payment_frequency} is not valid to generate a floating leg payment schedule.')

        unadjusted_accrual_dates = Scheduler.generate_dates_by_increments(start_date=self.end_accrual_date,
                                                                          end_date=self.start_accrual_date,
                                                                          increment=increment)
        unadjusted_accrual_dates.reverse()
        adjusted_accrual_dates = [self.date_adjustment_function(accrual_date) for accrual_date in unadjusted_accrual_dates]
        fixed_leg_accruals = []

        for start_accrual, end_accrual in itertools.pairwise(adjusted_accrual_dates):
            accrual = DayCountCalculator.compute_accrual_length(start_accrual,
                                                                end_accrual,
                                                                self.fixed_leg_day_count_convention)

            fixed_leg_accruals.append(SwapAccrual(start_accrual=start_accrual,
                                                  end_accrual=end_accrual,
                                                  accrual_factor=accrual))

        return fixed_leg_accruals


    @abstractmethod
    def generate_floating_leg_cashflow(self, interest_rate: Callable[[date], float], set_cashflow: bool = True) -> Cashflow:
        """
        An abstract method which subclasses must implement, and which
        dictates how the floating leg payment Cashflow is constructed from the provided
        callable interest_rate object and whether to store the cashflow in the object for
        later use in present value calculations.
        """

    def generate_fixed_leg_cashflow(self, set_cashflow: bool = True) -> Cashflow:
        """
        Creates a cashflow of fixed leg payments based on the swap's self.fixed_rate.

        The method iterates through all accruals made by the generate_fixed_leg_accrual_schedule()
        method and calculates the payment amount defined by:
            self.fixed_rate * self.notional * accrual_factor
        for each accrual_factor in the SwapAccrual list. Returns a Cashflow containing all these payments.
        """
        fixed_leg_accruals = self.generate_fixed_leg_accrual_schedule()
        fixed_payments = []

        for accrual in fixed_leg_accruals:
            payment_amount = accrual.accrual_factor * self.notional * self.fixed_rate
            payment_date = Scheduler.calculate_settlement_date(accrual.end_accrual,
                                                               self.payment_delay,
                                                               self.holiday_calendar)

            payment = Payment(payment_date=payment_date, payment=payment_amount)
            fixed_payments.append(payment)

        fixed_cf = Cashflow(fixed_payments)

        if set_cashflow:
            self[CashflowKeys.FIXED_LEG] = fixed_cf

        return fixed_cf


    def implied_rate(self, discount_curve: DiscountCurve, interest_rate: Curve, set_cashflow: bool = False) -> float:
        """
        Calculates the implied interest rate of the swap, which is the fixed interest rate which would render
        the swap fair-valued.
        """
        float_cashflow = self.generate_floating_leg_cashflow(interest_rate=interest_rate, set_cashflow=set_cashflow)
        float_pv = discount_curve.present_value(float_cashflow)

        fixed_accruals = self.generate_fixed_leg_accrual_schedule()
        discounted_fixed_accruals = 0.0
        for accrual in fixed_accruals:
            accrual_times_notional = accrual.accrual_factor * self.notional
            payment_date = Scheduler.calculate_settlement_date(accrual.end_accrual,
                                                               self.payment_delay,
                                                               self.holiday_calendar)

            discounted_fixed_accruals += discount_curve(payment_date) * accrual_times_notional

        return float_pv / discounted_fixed_accruals



class TermInterestRateSwap(FixedToFloatInterestRateSwap):

    def generate_floating_leg_cashflow(self, interest_rate: Callable[[date], float], set_cashflow: bool = True) -> Cashflow:
        """
        Creates a cashflow of floating leg payments from the provided interest rate callable.

        The interest_rate argument is assumed to be a general callable object which maps dates to
        floats representing interest rates. These could be either forward rates from a curve
        or short rates from a short rate model.

        The method iterates through all accruals made by the generate_fixed_leg_accrual_schedule()
        method and calculates the payment amount defined by:
            interest_rate(fixing_date) * self.notional * accrual_factor
        for each fixing_date and accrual_factor in the SwapAccrual list. Returns a Cashflow containing all these payments.
        """

        floating_leg_accruals = self.generate_floating_leg_accrual_schedule()
        floating_payments = []

        for accrual in floating_leg_accruals:
            fixing_date = Scheduler.calculate_settlement_date(accrual.start_accrual,
                                                              self.fixing_date_for_accrual_period,
                                                              self.holiday_calendar)

            interest_rate_fixing = interest_rate(fixing_date)

            payment_date = Scheduler.calculate_settlement_date(accrual.end_accrual,
                                                               self.payment_delay,
                                                               self.holiday_calendar)

            payment_amount = accrual.accrual_factor * self.notional * interest_rate_fixing
            payment = Payment(payment_date=payment_date, payment=payment_amount)
            floating_payments.append(payment)

        floating_cf = Cashflow(floating_payments)

        if set_cashflow:
            self[CashflowKeys.FLOATING_LEG] = floating_cf

        return floating_cf


class OvernightIndexSwap(FixedToFloatInterestRateSwap):

    def calculate_annualized_rate(self, interest_rate: Callable[[date], float], start_date: date, end_date: date) -> float:
        """
        Returns the annualized rate of accrual from compounding the provided interest_rate over all bsuiness days
        between (inclusive) start_date and end_date. The annualized rate is defined by
        Annualized Rate =
        (prod_{i=1}^{n} (1 + Rate(date_{i}) * Accrual(date_{i}, date_{i+1})) - 1) / sum_{i=1}^{n} Accrual(date_{i}, date_{i+1})

        The accruals Accrual(date_{i}, date_{i+1}) are assumed to use the DayCountConvention Actual over 360.
        """
        #TODO: Determine what to do about potentially missing start/end dates.
        accrual_dates = Scheduler.generate_business_days(start_date=start_date,
                                                         end_date=end_date,
                                                         holiday_calendar=self.holiday_calendar)

        total_accrued_growth = 1.0
        sum_accruals = 0.0

        for start_date, end_date in itertools.pairwise(accrual_dates):
            accrual = DayCountCalculator.compute_accrual_length(start_date=start_date,
                                                                end_date=end_date,
                                                                dcc=DayCountConvention.ACTUAL_OVER_360)
            sum_accruals += accrual
            total_accrued_growth *= (1 + interest_rate(start_date) * accrual)

        annualized_rate = (total_accrued_growth - 1.0) / sum_accruals
        return annualized_rate


    def generate_floating_leg_cashflow(self, interest_rate: Callable[[date], float], set_cashflow: bool = True) -> Cashflow:
        """
        Creates a cashflow of floating leg payments from the provided interest rate callable.

        The interest_rate argument is assumed to be a general callable object which maps dates to
        floats representing interest rates. These could be either forward rates from a curve
        or short rates from a short rate model.

        The method iterates through all accruals made by the generate_fixed_leg_accrual_schedule()
        method and calculates the payment amount defined by:
            interest_rate(fixing_date) * self.notional * accrual_factor
        for each fixing_date and accrual_factor in the SwapAccrual list. Returns a Cashflow containing all these payments.
        """

        floating_leg_accruals = self.generate_floating_leg_accrual_schedule()
        floating_payments = []

        for accrual in floating_leg_accruals:

            payment_date = Scheduler.calculate_settlement_date(accrual.end_accrual,
                                                               self.payment_delay,
                                                               self.holiday_calendar)

            annualized_rate = self.calculate_annualized_rate(interest_rate=interest_rate,
                                                             start_date=accrual.start_accrual,
                                                             end_date=accrual.end_accrual)

            payment_amount = accrual.accrual_factor * self.notional * annualized_rate
            payment = Payment(payment_date=payment_date, payment=payment_amount)
            floating_payments.append(payment)

        floating_cf = Cashflow(floating_payments)

        if set_cashflow:
            self[CashflowKeys.FLOATING_LEG] = floating_cf

        return floating_cf



#---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from fixedIncome.src.scheduling_tools.holidays import US_FEDERAL_HOLIDAYS
    from fixedIncome.src.curves.curve_enumerations import InterpolationMethod
    from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention

    test_libor_swap = TermInterestRateSwap(
        float_index=CurveIndex.LIBOR_3M,
        direction=InterestRateSwapDirection.RECEIVER_FIXED,
        fixed_rate=0.055,
        notional=1_000_000,
        purchase_date=date(2024, 1, 1),
        settlement_convention=SettlementConvention.T_PLUS_TWO_BUSINESS,
        tenor='10Y',
        floating_leg_payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        fixed_leg_payment_frequency=PaymentFrequency.QUARTERLY,
        floating_leg_day_count_convention=DayCountConvention.ACTUAL_OVER_360,
        fixed_leg_day_count_convention=DayCountConvention.THIRTY_OVER_THREESIXTY,
        holiday_calendar=US_FEDERAL_HOLIDAYS,
        payment_delay=SettlementConvention.T_PLUS_ZERO_BUSINESS,
        business_day_adjustment=BusinessDayAdjustment.MODIFIED_FOLLOWING
    )

    test_libor_swap.generate_fixed_leg_accrual_schedule()
    test_libor_swap.generate_floating_leg_accrual_schedule()

    start_val = KnotValuePair(date(2024, 1, 1), 0.05)
    end_val = KnotValuePair(date(2034, 1, 1), 0.10)
    test_curve = Curve([start_val, end_val],
                       interpolation_method=InterpolationMethod.LINEAR,
                       interpolation_day_count_convention=DayCountConvention.ACTUAL_OVER_360)

    test_cashflow = test_libor_swap.generate_floating_leg_cashflow(test_curve)
    print(list(test_cashflow))

    # SOFR Swap Example
    test_sofr_swap = OvernightIndexSwap(
        float_index=CurveIndex.SOFR,
        direction=InterestRateSwapDirection.RECEIVER_FIXED,
        fixed_rate=0.055,
        notional=1_000_000,
        purchase_date=date(2024, 1, 1),
        settlement_convention=SettlementConvention.T_PLUS_TWO_BUSINESS,
        tenor='10Y',
        floating_leg_payment_frequency=PaymentFrequency.ANNUAL,
        fixed_leg_payment_frequency=PaymentFrequency.ANNUAL,
        floating_leg_day_count_convention=DayCountConvention.ACTUAL_OVER_360,
        fixed_leg_day_count_convention=DayCountConvention.ACTUAL_OVER_360,
        holiday_calendar=US_FEDERAL_HOLIDAYS,
        payment_delay=SettlementConvention.T_PLUS_TWO_BUSINESS,
        business_day_adjustment=BusinessDayAdjustment.MODIFIED_FOLLOWING
    )

