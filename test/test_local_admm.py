from grecco_sim.local_problems import local_admm
from grecco_sim.util import type_defs


def test_local_admm() -> bool:
    """Test if the local ADMM controller an solver tools work

    :return: True if the test passes
    :rtype: bool
    """

    sys_pars = type_defs.SysParsPVBat(
        "test_system", 0.25, 0.3, 0.1, 0.5, 5., 5.
    )
    opt_pars = type_defs.OptParameters(solver_name = "osqp", rho = 2., mu = 10., horizon=10, alpha=1.)

    solver = local_admm.LocalADMMSolver(
        horizon=10,
        sys_id="ag_test",
        sys_parameters=sys_pars,
        controller_pars=opt_pars,
        signal_lengths=[1]
    )


if __name__ == "__main__":
    test_local_admm()


