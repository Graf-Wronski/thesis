from pathlib import Path

import numpy as np

from grecco_sim.controller.casadi_model import CasadiModel

log_path = Path(".") / "logs"


def write_solver_parameters_to_file(x_lb, x_ub, g_lb, g_ub, p):

    if not log_path.exists():
        log_path.mkdir()

    with open(log_path / "optimization.txt", 'w') as f:
        f.write("Optimization Problem Parameters:\n\n")

        f.write("Lower bounds on x (lbx):\n")
        f.write(f"{x_lb}\n\n")

        f.write("Upper bounds on x (ubx):\n")
        f.write(f"{x_ub}\n\n")

        f.write("Lower bounds on g (lbg):\n")
        f.write(f"{g_lb}\n\n")

        f.write("Upper bounds on g (ubg):\n")
        f.write(f"{g_ub}\n\n")

        f.write("Parameter vector (p):\n")
        f.write(f"{p}\n")


def log_mathematical_model(model: CasadiModel, parameters: dict, now: int,
                           sys_id: str):


    if not log_path.exists():
        log_path.mkdir()

    lbx, ubx = model.state_bounds
    lbg, ubg = model.constraint_bounds

    tag = sum([np.sum(y) for y in parameters.values()])

    with open(log_path / f"optimization_{sys_id}_{now}_{tag}.txt", 'w') as f:
        f.write("Optimization Problem Parameters:\n\n")

        f.write("Lower bounds on x (lbx):\n")
        f.write(f"{lbx}\n\n")

        f.write("Upper bounds on x (ubx):\n")
        f.write(f"{ubx}\n\n")

        f.write("Lower bounds on g (lbg):\n")
        f.write(f"{lbg}\n\n")

        f.write("Upper bounds on g (ubg):\n")
        f.write(f"{ubg}\n\n")

        f.write("Parameters p:\n")
        for key, val in parameters.items():
            f.write(f"{key}: {val}\n")
