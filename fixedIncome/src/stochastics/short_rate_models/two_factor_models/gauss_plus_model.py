"""
This script contains the Vasicek Model for the short rate.
Reference, *Fixed Income Securities, 4th Ed.* by Tuckman and Serrat, page 205.

Unit tests are contained in
fixedIncome.tests.test_stochastics.test_short_rate_models.test_one_factor_models.test_vasicek_model.py
"""
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.base_short_rate_model import ShortRateModel, DriftDiffusionPair


def gauss_plus_rate_drift_diffusion(rate_reversion_scale: float) -> DriftDiffusionPair:
    """
    Function to auto-generate the drift and diffusion functions for the rate component
    of the Gauss+  interest rate model. The model has the SDE
    dr = -a_r * (m_t - r_t) dt where
    a_r is a positive float representing the mean reversion scaling,
    m_t is the medium term factor, and
    r_t is the volatility.
    """
    def drift_fxcn(time: float, rate: float, medium_term_factor: float) -> float:
        return -rate_reversion_scale * (medium_term_factor - rate)

    def diffusion_fxcn(*args, **kwargs) -> float:
        return 0.0

    return DriftDiffusionPair(drift=drift_fxcn, diffusion=diffusion_fxcn)

def gauss_plus_medium_term_factor_drift_diffusion(medium_reversion_scale: float, medium_volatiltiy: float) -> DriftDiffusionPair:
    """

    """
    def drift_fxcn() -> float:
        pass

    def diffusion_fxcn() -> float:
        pass

    return DriftDiffusionPair(drift=drift_fxcn, diffusion=diffusion_fxcn)