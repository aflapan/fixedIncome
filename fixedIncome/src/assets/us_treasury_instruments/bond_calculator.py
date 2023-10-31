


class BondCalculator:

    # --------------------------------------------------------------------
    # Pricing Methods

    # TODO: Re-work these
    def calculate_ex_post_given_reinvestment_rates(self, reinvestment_rates: pd.Series) -> float:
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
            return 2 * diff_in_years + int(math.ceil(diff_in_months /6))

        powers = pd.Series([num_accrual_periods_left(date) for date in self.payment_schedule.index],
                           index=self.payment_schedule.index, name='Number of Acrual Periods')

        growth_factors = (1 + reinvestment_rates / (2 * 100)) ** powers
        return (self.payment_schedule * growth_factors).sum()

    def calculate_spot_rate_from_ex_post(self, ex_post :float) -> float:
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
        root_minus_one = ratio_of_final_to_pric e* *( 1 /num_accrual_periods) - 1  # Step (2)

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


