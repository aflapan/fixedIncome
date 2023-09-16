import numpy as np
from typing import Callable
import time
import matplotlib.pyplot as plt


def time_this(fxcn):
    def timed_fxcn(*args, **kwargs):
        start = time.perf_counter()
        result = fxcn(*args, **kwargs)
        end = time.perf_counter()
        print(f'The elapsed time was {end-start} seconds')
        return result

    return timed_fxcn



@time_this
def default_bernoulli_paths(hazard_rate_fxcn: Callable[[float], float], time_to_maturity: float, dt: float = 0.01, num_paths :int = 1_000) -> np.array:
    """
    A function to create sample paths using monte-carlo
    """
    default_times = np.empty((num_paths,))
    default_times[:] = np.nan
    for i in range(num_paths):
        time = dt
        default_has_occured = False
        while time < time_to_maturity:
            prob_default_next_interval = hazard_rate_fxcn(time) * dt
            default_has_occured = (np.random.uniform(0, 1, 1) <= prob_default_next_interval).item()
            if default_has_occured:
                default_times[i] = time
                break
            else:
                time += dt

    return default_times



def generate_cox_model_sample_paths() -> np.array:
    pass





def main():
    RATE, NUM_TRIALS = 0.02, 1_000_000
    hazard_fxcn = lambda val: RATE
    default_times = default_bernoulli_paths(hazard_fxcn, time_to_maturity=5, dt=1/250)

    plt.plot(figsize=(10, 6))
    plt.title(f'Estimated Time to Default for Hazard Rate {RATE}\nAcross {NUM_TRIALS} Monte Carlo Simulations')
    plt.hist(default_times, bins=100, density=True)
    plt.xlabel('Time to Default in Years')
    plt.show()


if __name__ == '__main__':
    main()