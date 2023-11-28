
from fixedIncome.src.stochastics.brownian_motion import datetime_to_path_call, BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.base_short_rate_model import DriftDiffusionPair

def geometric_brownian_motion_drift_diffusion(drift: float, volatility: float) -> DriftDiffusionPair:
    """
    Produces the drift-diffusion named tuple of functions in the SDE for geometric brownian motion:
    dX_t = mu * X_t dt + sigma * X_t dW_t.
    """

    def drift_fxcn(time: float, current_value: float) -> float:
        return drift * current_value

    def diffusion_fxcn(time: float, current_value: float) -> float:
        return volatility * current_value

    return DriftDiffusionPair(drift=drift_fxcn, diffusion=diffusion_fxcn)

class GeometricBrownianMotion:
    def __init__(self):
        pass



