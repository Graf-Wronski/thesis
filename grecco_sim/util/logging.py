from pathlib import Path

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