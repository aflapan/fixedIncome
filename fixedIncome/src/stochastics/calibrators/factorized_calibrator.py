"""
This calibrator is designed to factorize the model parameters into those
which effect the yield volatilities and those which effect the shape
of the yield curve.

We first fit those parameters to the provided yield volatilities, and then
keeping those fixed we fit the remaining parmeters to best approximate the
shape of the yield curve.

See Rebonato *Bond Pricing and Yield Curve Modeling* Section 16.6.4 and 16.6.5.
"""

import math

from typing import Iterable, NamedTuple
from datetime import date
import numpy as np
from scipy.optimize import minimize
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.vasicek_model import MultivariateVasicekModel, VasicekModel, CalibrationParamterType
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.doubly_mean_reverting_vasicek import DoublyMeanRevertingVasicek


class CalibrationInput(NamedTuple):
    maturity_date: date
    value: float


ModelToParamDict = {
    VasicekModel: {
        CalibrationParamterType.VOLATILITY: [],
        CalibrationParamterType.SHAPE: [],
        CalibrationParamterType.STATE_VARIABLES: []
    },
    DoublyMeanRevertingVasicek: {
        CalibrationParamterType.VOLATILITY: [],
        CalibrationParamterType.SHAPE: [],
        CalibrationParamterType.STATE_VARIABLES: []
    },

}


class ModelFactory:
    @classmethod
    def create_model(cls, param_dict):
        return cls.from_volatility_shape_parameters(param_dict)




class VolatilityFactorizedCalibrator:
    """ A class which takes an instance of MultivariateVasicekModel or VasicekModel and fits the model parameters
    to the provided inputs.
    """

    def __init__(self, volatility_inputs: Iterable[CalibrationInput], price_inputs: Iterable[CalibrationInput]) -> None:
        self.volatility_inputs = sorted(list(volatility_inputs), key=lambda calib_input: calib_input.maturity_date)
        self.price_inputs = sorted(list(price_inputs), key=lambda calib_input: calib_input.maturity_date)
        self.model_factory = ModelFactory()


    def calibrate_volatility_parameters(self,  model: DoublyMeanRevertingVasicek | VasicekModel) -> None:
        model_vol_params = model.parameter_calibration_dict[CalibrationParamterType.VOLATILITY]

        vol_accruals = [DayCountCalculator.compute_accrual_length(start_date=model.start_date_time,
                                                              end_date=vol_input.maturity_date,
                                                              dcc=model.day_count_convention)
                    for vol_input in self.volatility_inputs]

        def volatility_loss_function(volatility_params: np.array) -> float:
            assert len(model_vol_params) == len(volatility_params)
            for index, param in enumerate(model_vol_params):
                model.parameter_calibration_dict[CalibrationParamterType.VOLATILITY][index] =\
                    volatility_params[index]

            new_model = DoublyMeanRevertingVasicek.from_volatility_shape_parameters(model.parameter_calibration_dict,
                                                                                    start_datetime=model.start_date_time,
                                                                                    end_datetime=model.end_date_time,
                                                                                    dt=model.dt)

            predicted_volatilities = (new_model.yield_volatility(vol_input.maturity_date)
                                      for vol_input in self.volatility_inputs)

            error = math.sqrt(sum((pred_vol - actual_vol.value)**2
                       for accrual, pred_vol, actual_vol in zip(vol_accruals, predicted_volatilities, self.volatility_inputs)))
            return error

        initial_volatility_params = np.array([0.01 for _ in range(len(model_vol_params))])

        vol_min_result = minimize(fun=volatility_loss_function,
                                  #method='Nelder-Mead',
                                  x0=initial_volatility_params,
                                  tol=1E-5,
                                  bounds=[(0.0, 1.0) for _ in range(len(initial_volatility_params))],
                                  options={'disp': True}
                                  )
        return vol_min_result


    def calibrate_shape_state_variable_parameters(self,  model: DoublyMeanRevertingVasicek | VasicekModel) -> None:
        model_shape_params = model.parameter_calibration_dict[CalibrationParamterType.SHAPE]
        model_state_params = model.parameter_calibration_dict[CalibrationParamterType.STATE_VARIABLES]

        price_accruals = [DayCountCalculator.compute_accrual_length(start_date=model.start_date_time,
                                                              end_date=price_input.maturity_date,
                                                              dcc=model.day_count_convention)
                    for price_input in self.price_inputs]

        def price_loss_function(price_and_state_params: np.array) -> float:
            assert (len(model_shape_params) + len(model_state_params)) == len(price_and_state_params)

            for index, param in enumerate(price_and_state_params[:len(model_shape_params)]):
                model.parameter_calibration_dict[CalibrationParamterType.SHAPE][index] =\
                    price_and_state_params[index]

            for index, param in enumerate(price_and_state_params[len(model_shape_params):]):
                model.parameter_calibration_dict[CalibrationParamterType.STATE_VARIABLES][index] = \
                    price_and_state_params[index+len(model_shape_params)]

            new_model = DoublyMeanRevertingVasicek.from_volatility_shape_parameters(model.parameter_calibration_dict,
                                                                                    start_datetime=model.start_date_time,
                                                                                    end_datetime=model.end_date_time,
                                                                                    dt=model.dt)

            predicted_prices = (new_model.zero_coupon_bond_price(price_input.maturity_date)
                                      for price_input in self.price_inputs)

            error = math.sqrt(sum((pred_price - actual_price.value)**2
                       for accrual, pred_price, actual_price in zip(price_accruals, predicted_prices, self.price_inputs)))
            return error

        initial_shape_state_params = np.array([0.01 for _ in range(len(model_shape_params) + len(model_state_params))])

        price_min_result = minimize(fun=price_loss_function,
                                  #method='Nelder-Mead',
                                  x0=initial_shape_state_params,
                                  tol=1E-5,
                                  bounds=[(0.0, 1.0) for _ in range(len(initial_shape_state_params))],
                                  options={'disp': True}
                                  )
        return price_min_result

    def calibrate(self, model: MultivariateVasicekModel | VasicekModel) -> None:
        """ Performs the factorized calibration by first fitting the volatility parameters
        and then afterwards fitting the shape parameters.
        """

        vol_min_result = self.calibrate_volatility_parameters(model)

        vol_calibrated_model = DoublyMeanRevertingVasicek.from_volatility_shape_parameters(
            model.parameter_calibration_dict,
            start_datetime=model.start_date_time,
            end_datetime=model.end_date_time,
            dt=model.dt)

        price_min_result = self.calibrate_shape_state_variable_parameters(vol_calibrated_model)

        calibrated_model = DoublyMeanRevertingVasicek.from_volatility_shape_parameters(
            vol_calibrated_model.parameter_calibration_dict,
            start_datetime=vol_calibrated_model.start_date_time,
            end_datetime=vol_calibrated_model.end_date_time,
            dt=model.dt)

        return calibrated_model


if __name__ == '__main__':
    import math
    from datetime import datetime, time
    from dateutil.relativedelta import relativedelta
    import matplotlib.pyplot as plt

    from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator, DayCountConvention
    from fixedIncome.src.scheduling_tools.scheduler import Scheduler
    from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
    from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.doubly_mean_reverting_vasicek import DoublyMeanRevertingVasicek
    # Setup
    start_date = date(2025, 1, 1)

    maturity_dates = [
        date(2026, 1, 1),
        date(2027, 1, 1),
        date(2028, 1, 1),
        date(2030, 1, 1),
        date(2035, 1, 1),
        date(2040, 1, 1)
    ]

    accruals = [DayCountCalculator.compute_accrual_length(start_date=start_date,
                                                          end_date=maturity_date,
                                                          dcc=DayCountConvention.ACTUAL_OVER_ACTUAL)
                for maturity_date in maturity_dates]

    yields = [0.05, 0.06, 0.065, 0.0625, 0.06, 0.0575]


    volatility_inputs = [
        CalibrationInput(maturity_date=date(2026, 1, 1), value=0.0075),
        CalibrationInput(maturity_date=date(2027, 1, 1), value=0.009),
        CalibrationInput(maturity_date=date(2028, 1, 1), value=0.01),
        CalibrationInput(maturity_date=date(2030, 1, 1), value=0.0125),
        CalibrationInput(maturity_date=date(2035, 1, 1), value=0.0110),
        CalibrationInput(maturity_date=date(2040, 1, 1), value=0.00925)
    ]

    price_inputs = [
        CalibrationInput(maturity_date=maturity_date, value=math.exp(-accrual*yield_to_maturity))
        for maturity_date, accrual, yield_to_maturity in zip(maturity_dates, accruals, yields)
    ]


    def calibrate_single_variable():
        end_time = datetime(2041, 10, 15, 0, 0, 0, 0)
        brownian_motion = BrownianMotion(start_date_time=datetime.combine(start_date, time.min),
                                         end_date_time=datetime.combine(end_time, time.min),
                                         dimension=1)
        brownian_motion.generate_path(dt=relativedelta(hours=1), seed=1, set_path=True)

        dates = Scheduler.generate_dates_by_increments(start_date=datetime.combine(start_date, time.min),
                                                       end_date=datetime.combine(end_time, time.min),
                                                       increment=relativedelta(days=1),
                                                       max_dates=1_000_000)

        vm = VasicekModel(reversion_level=0.04,
                          reversion_speed=0.5,
                          volatility=0.02,
                          brownian_motion=brownian_motion)

        admissible_dates = [date_obj for date_obj in dates if date_obj <= vm.end_date_time]

        calibrator = VolatilityFactorizedCalibrator(volatility_inputs=volatility_inputs,
                                                    price_inputs=price_inputs)

        calibrator.calibrate_volatility_parameters(model=vm)


        plt.figure(figsize=(13, 6))
        plt.title('Zero-Coupon Bond Yield Volatilities and Calibrated Zero Coupon Yield Volatility Curve for the Vasicek Model')
        plt.plot([vol_input.maturity_date for vol_input in volatility_inputs],
                 [vol_input.value for vol_input in volatility_inputs])
        plt.plot(admissible_dates[1:], [vm.yield_volatility(date_obj) for date_obj in admissible_dates[1:]])
        plt.legend(['Yield Volatilities', 'Volatility Curve'], frameon=False)
        plt.show()

        plt.figure(figsize=(13, 6))
        plt.title('Zero-Coupon Bond Prices and Calibrated Zero Coupon Bond Price Curve for the Vasicek Model')
        plt.plot([price_input.maturity_date for price_input in price_inputs],
                 [price_input.value for price_input in price_inputs])
        plt.plot(admissible_dates[1:], [vm.zero_coupon_bond_price(date_obj) for date_obj in admissible_dates[1:]])
        plt.legend(['Bond Prices', 'Price Curve'], frameon=False)
        plt.show()

        plt.figure(figsize=(13, 6))
        plt.ylabel('Yield (%)')
        plt.xlabel('Maturity Date')
        plt.title('Zero-Coupon Bond Yields and Calibrated Yield Curve for the Vasicek Model')
        plt.plot(maturity_dates, [yield_to_maturity*100 for yield_to_maturity in yields])
        plt.plot(admissible_dates[1:], [100 * vm.zero_coupon_yield(date_obj) for date_obj in admissible_dates[1:]])
        plt.legend(['Bond Yields', 'Calibrated Yield Curve'], frameon=False)
        plt.show()

        print('Reversion Speed:', vm.reversion_speed)
        print('Reversion Level:', vm.long_term_mean)
        print('Volatility:', vm.volatility)
        print('Starting short rate:', vm.calibrated_starting_short_rate)


    def calibrate_doulby_mean_reverting():

        start_date = datetime.combine(date(2025, 1, 1), time.min)
        end_time = datetime(2041, 10, 15, 0, 0, 0, 0)
        brownian_motion = BrownianMotion(start_date_time=datetime.combine(start_date, time.min),
                                         end_date_time=end_time,
                                         dimension=2)

        brownian_motion.generate_path(dt=relativedelta(hours=1), seed=1, set_path=True)

        dates = Scheduler.generate_dates_by_increments(start_date=start_date,
                                                       end_date=end_time,
                                                       increment=relativedelta(days=1),
                                                       max_dates=1_000_000)

        short_rate_reversion = 0.5
        long_reversion = 0.05

        short_rate_vol = 50 / 10_000
        long_vol = 25 / 10_000
        short_long_corr = 0.25

        long_term_mean = 0.05

        dmr_vm = DoublyMeanRevertingVasicek(
            short_rate_reversion_speed=short_rate_reversion,
            short_rate_volatility=short_rate_vol,
            long_term_mean=long_term_mean,
            long_term_reversion_speed=long_reversion,
            long_term_volatility=long_vol,
            short_rate_long_term_correlation=short_long_corr,
            start_datetime=start_date,
            end_datetime=end_time
        )

        admissible_dates = [date_obj for date_obj in dates if date_obj <= dmr_vm.end_date_time]


        calibrator = VolatilityFactorizedCalibrator(volatility_inputs=volatility_inputs,
                                                    price_inputs=price_inputs)


        before_calib_yields = [dmr_vm.yield_volatility(date_obj) for date_obj in admissible_dates[1:]]


        # volatility calibration

        def calibrate_shape_state_variable_parameters(self, model: DoublyMeanRevertingVasicek | VasicekModel) -> None:
            model_shape_params = model.parameter_calibration_dict[CalibrationParamterType.SHAPE]
            model_state_params = model.parameter_calibration_dict[CalibrationParamterType.STATE_VARIABLES]

            price_accruals = [DayCountCalculator.compute_accrual_length(start_date=model.start_date_time,
                                                                        end_date=price_input.maturity_date,
                                                                        dcc=model.day_count_convention)
                              for price_input in self.price_inputs]

            def price_loss_function(price_and_state_params: np.array) -> float:
                assert (len(model_shape_params) + len(model_state_params)) == len(price_and_state_params)

                for index, param in enumerate(price_and_state_params[:len(model_shape_params)]):
                    model.parameter_calibration_dict[CalibrationParamterType.SHAPE][index] = \
                        price_and_state_params[index]

                for index, param in enumerate(price_and_state_params[len(model_shape_params):]):
                    model.parameter_calibration_dict[CalibrationParamterType.STATE_VARIABLES][index] = \
                        price_and_state_params[index + len(model_shape_params)]

                new_model = DoublyMeanRevertingVasicek.from_volatility_shape_parameters(
                    model.parameter_calibration_dict,
                    start_datetime=model.start_date_time,
                    end_datetime=model.end_date_time,
                    dt=model.dt)

                predicted_prices = (new_model.zero_coupon_bond_price(price_input.maturity_date)
                                    for price_input in self.price_inputs)

                error = math.sqrt(sum((pred_price - actual_price.value) ** 2
                                      for accrual, pred_price, actual_price in
                                      zip(price_accruals, predicted_prices, self.price_inputs)))
                return error

            initial_shape_state_params = np.array(
                [0.01 for _ in range(len(model_shape_params) + len(model_state_params))])

            price_min_result = minimize(fun=price_loss_function,
                                        # method='Nelder-Mead',
                                        x0=initial_shape_state_params,
                                        tol=1E-5,
                                        bounds=[(0.0, 1.0) for _ in range(len(initial_shape_state_params))],
                                        options={'disp': True}
                                        )
            return price_min_result





        calibrated_model = calibrator.calibrate(dmr_vm)

        calibrated_model = DoublyMeanRevertingVasicek.from_volatility_shape_parameters(
            calibrated_model.parameter_calibration_dict,
            start_datetime=calibrated_model.start_date_time,
            end_datetime=calibrated_model.end_date_time,
            dt=calibrated_model.dt)

        calibrated_model.generate_path(starting_state_space_values=calibrated_model.parameter_calibration_dict[CalibrationParamterType.STATE_VARIABLES],
                                       set_path=True, seed=1)


        plt.figure(figsize=(13, 6))
        plt.title(
            'Zero-Coupon Bond Yield Volatilities and Calibrated Zero Coupon Yield Volatility Curve for the Doubly-Mean Reverting Vasicek Model')
        plt.plot([vol_input.maturity_date for vol_input in volatility_inputs],
                 [vol_input.value for vol_input in volatility_inputs])
        plt.plot(admissible_dates[1:], before_calib_yields)
        plt.plot(admissible_dates[1:], [calibrated_model.yield_volatility(date_obj) for date_obj in admissible_dates[1:]])
        plt.legend(['Yield Volatilities', 'Original Volatiltiy Curve', 'Calibrated Volatility Curve'], frameon=False)
        plt.show()

        plt.figure(figsize=(13, 6))
        plt.title('Zero-Coupon Bond Prices and Calibrated Zero Coupon Bond Price Curve for the Doubly-Mean Reverting Vasicek Model')
        plt.plot([price_input.maturity_date for price_input in price_inputs],
                 [price_input.value for price_input in price_inputs])
        plt.plot(admissible_dates[1:], [calibrated_model.zero_coupon_bond_price(date_obj) for date_obj in admissible_dates[1:]])
        plt.legend(['Bond Prices', 'Price Curve'], frameon=False)
        plt.show()

        print('Starting state space variables are:', calibrated_model.state_variables_diffusion_process.path[:,0])
        print('Starting state space variables dict are:', calibrated_model.parameter_calibration_dict[CalibrationParamterType.STATE_VARIABLES])



    calibrate_doulby_mean_reverting()
