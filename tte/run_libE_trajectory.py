import numpy as np
from numpy import random

from libensemble.libE import libE
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.tools import parse_args, add_unique_random_streams

from tte import run_timesteps, evaluate_state


def timesteps_simf_wrap(In, persis_info, sim_specs, libE_info):

    in_state = In["state"][0]
    stop_sim = In["stop_sim"][0]

    rng = persis_info["rand_stream"]
    num_steps = sim_specs["user"]["num_steps"]
    threshold = sim_specs["user"]["threshold"]
    timestep_time = sim_specs["user"]["timestep_time"]

    out_state, persis_info["rand_stream"] = run_timesteps(
        in_state,
        rng,
        num_steps,
        threshold,
        timestep_time,
    )

    Out = np.zeros(1, dtype=sim_specs["out"])
    Out["state"] = out_state
    print(out_state)

    return Out, persis_info


def evaluate_genf_wrap(In, persis_info, gen_specs, libE_info):

    in_state = In["state"]

    if len(in_state):
        in_state = In["state"][0]
    else:
        in_state = 0

    stop_sim = evaluate_state(
        in_state, gen_specs["user"]["target_count"], gen_specs["user"]["delay"]
    )

    Out = np.zeros(1, dtype=gen_specs["out"])
    Out["stop_sim"] = stop_sim

    return Out, persis_info


if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()

    sim_specs = {
        "sim_f": timesteps_simf_wrap,
        "in": ["state", "stop_sim"],
        "out": [("state", int)],
        "user": {
            "timestep_time": 0.01,
            "num_steps": 10,
            "threshold": 0.01,
        },
    }

    gen_specs = {
        "gen_f": evaluate_genf_wrap,
        "in": ["state"],
        "out": [("stop_sim", bool)],
        "user": {
            "target_count": 8,
            "delay": 0.1,
        },
    }

    exit_criteria = {"sim_max": 100}
    persis_info = add_unique_random_streams({}, nworkers + 1)

    H, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs
    )
