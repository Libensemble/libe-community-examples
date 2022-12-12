""" A toy representation of the emittance minimization problem.

The following functions are included:

 - the ParMOO compatible simulation function accelerator_sim_model(x)
 - the ParMOO compatible objective function emittance(x, s, der=0)
 - the ParMOO compatible objective function bunch_length(x, s, der=0)

"""

import numpy as np

# Define the parmoo input keys for later use
DES_NAMES = ["foc_strength", "lens_shift"]

# Define input bounds and scale/shift terms
lb = np.array([0.5e6, -0.5e-2])
ub = np.array([0.9e6, 0.5e-2])

__shifts__ = lb.copy()
__scales__ = np.zeros(2)
__scales__ = ub - lb

def accelerator_sim_model(H, persis_info, sim_specs, _):
    " libE compatible stand-in function for the real accelerator simulation. "

    # Get size of libE batch and initialize an output struct
    batch = len(H)
    H_o = np.zeros(batch, dtype=sim_specs['out'])
    for i in range(batch):
        # Extract the parmoo named-inputs into a numpy ndarray in [0,1]
        xx = np.zeros(len(DES_NAMES))
        for j, name in enumerate(DES_NAMES):
            xx[j] = (H[name][i] - __shifts__[j]) / __scales__[j]
        # Initialize empty outputs
        H_o['sim out'][i] = np.zeros(4)
        # Assign made-up outputs
        H_o['sim out'][i] = np.linalg.norm(xx - np.array([0.2, 0.2])) ** 2 + 1
        H_o['sim out'][i] = np.linalg.norm(xx - np.array([0.2, 0.8])) ** 2 + 1
        H_o['sim out'][i] = xx[0] * xx[1]
        H_o['sim out'][i] = np.linalg.norm(xx - np.array([0.8, 0.7])) ** 2 + 1
    return H_o, persis_info

def emittance(x, s, der=0):
    """ A ParMOO objective that calculates the emittance of the
    accelerator_sim_model.

    Args:
        x (numpy structured array): contains 2 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "sim out"
            with 4 float-valued fields, as output by the
            accelerator_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["sim out"][0] = s["sim out"][1]
        res["sim out"][1] = s["sim out"][0]
        res["sim out"][2] = -2.0 * s["sim out"][2]
        return res
    else:
        return s["sim out"][0] * s["sim out"][1] - (s["sim out"][2] ** 2)

def bunch_length(x, s, der=0):
    """ A ParMOO objective that calculates the bunch length of the
    accelerator_sim_model.

    Args:
        x (numpy structured array): contains 2 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "sim out"
            with 4 float-valued fields, as output by the
            accelerator_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["sim out"][3] = 1.0
        return res
    else:
        return s["sim out"][3]


if __name__ == "__main__":
    """ Driver code that checks the output is reasonable for a dummy input """

    # Do 100 random tests
    for i in range(100):
        # Draw a random sample from the bounding box
        x = np.random.random_sample(2) * __scales__ + __shifts__
        # Define as parmoo expects
        des_type = [(f"{name}", float) for name in DES_NAMES]
        xx = np.zeros(1, dtype=des_type)[0]
        Ho, persis_info = np.zeros(1, dtype=[("sim out", float, 4)])[0]
        for j in range(len(DES_NAMES)):
            xx[DES_NAMES[j]] = x[j]
        Ho["sim out"][:] = accelerator_sim_model(xx)
        assert(emittance(xx, Ho) >= 0)
