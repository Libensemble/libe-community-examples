from typing import Optional, Tuple
import time

from numpy import random


# TODO (wardlt): Should we add a "restart" cost?
def run_timesteps(state: int, rng: Optional[random.RandomState], num_steps: int, threshold: float,
                 timestep_time: Optional[float]) -> Tuple[int, random.RandomState]:
    """Run a certain number of timesteps

    Args:
        state: Number of times an event has happened
        rng: Random number generator
        num_steps: Number of timesteps to perform
        threshold: Probability of an event occuring
        timestep_time: Minimum time per timestep (s)
    Returns:
        - Cumulative number of times event has happened
        - Random number generator state
    """

    if timestep_time is not None:
        time.sleep(timestep_time * num_steps)

    # Make a new rng if none provided
    if not rng:
        rng = random.default_rng()

    # Determine how many times the event happens
    state += rng.binomial(num_steps, p=threshold)
    return state, rng


def evaluate_state(state: int, target_count: int, delay: Optional[float] = None) -> bool:
    """Determine whether we should stop a trajectory

    Args:
        state: Number of times an event has happened
        target_count: Target number of events
        delay: Minimum run time (s)
    Returns:
         Whether we should continue this simulation or start a new one
    """

    if delay is not None:
        time.sleep(delay)

    return state >= target_count
