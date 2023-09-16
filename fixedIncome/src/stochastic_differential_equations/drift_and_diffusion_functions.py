"""
This module contains a collection of useful drift and diffusion functions
"""

from typing import NamedTuple, Callable

class DriftDiffusionPair(NamedTuple):
    drift: Callable[[float, float], float]
    diffusion: Callable[[float, float], float]

#------------------------------------------------------------------
# Brownian Motions

def brownian_motion_with_drift():
    pass


def geometric_brownian_motion(drift: float, volatility: float) -> DriftDiffusionPair:
    def drift_fxcn(time: float, current_value: float) -> float:
        return drift * current_value

    def diffusion_fxcn(time: float, current_value: float) -> float:
        return volatility * current_value

    return DriftDiffusionPair(drift=drift_fxcn, diffusion=diffusion_fxcn)


#------------------------------------------------------------------
# Interest Rate Models
def vasicek_model(long_term_mean: float, reversion_scale: float, volatility: float) -> DriftDiffusionPair:
    """
    Function to auto-generate the drift and diffusion functions for the Vasicek interest rate
    model. The model has the SDE dr = a * (m - r) dt + sigma dWt where
    a is a positive float representing the mean reversion scaling,
    m is the long-term mean for the interest rate, and
    sigma is the volatility.

    Reference, *Fixed Income Securities, 4th Ed.* by Tuckman and Serrat, page 205.
    """
    def drift_fxcn(time: float, current_value: float) -> float:
        return reversion_scale * (long_term_mean - current_value)

    def diffusion_fxcn(time: float, current_value: float) -> float:
        return volatility

    return DriftDiffusionPair(drift=drift_fxcn, diffusion=diffusion_fxcn)
