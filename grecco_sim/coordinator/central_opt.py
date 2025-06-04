"""
This module provides a centralized controller with the goal of limiting power to a peak value.

"""
import numpy as np

from grecco_sim.coordinator import coordinator

from grecco_sim.util import signals
from grecco_sim.local_problems import local_solver_pv_bat
from grecco_sim.util import type_defs
from grecco_sim.util import helper


class CentralOptimizationCoordinator(coordinator.Coordinator):
    """
    This controller is a completely centralized controller for the case of limiting grid power to a certain value
    It solves an OCP at every timestep and passes the controls back to the connected nodes.
    """

    coord_name = "Central Control"

    def __init__(self, controller_pars: type_defs.OptParameters, sys_ids:list[str], grid: type_defs.GridDescription):
        super().__init__(controller_pars.horizon)
        self.current_signal_time = -1

        self.grid = grid
        self.opt_pars = controller_pars
        
        self.constraint_value = 1e10

    def has_converged(self, time_index):
        return time_index == self.current_signal_time

    def get_signals(self, futures: dict[str, type_defs.Schedule]) -> dict[str, signals.Signal]:

        current_horizon = list(futures.values())[0].horizon

        flex_futures, inflex_futures, flex_sum, inflex_sum = helper.get_flex_and_inflex(futures)

        if (inflex_sum > self.grid.p_lim).any() and True:
            print(inflex_sum)

        flex_controllers = {
            sys_id: 
            local_solver_pv_bat.CasadiSolverLocalPVBatVectorized(
                current_horizon, sys_id, flex_futures[sys_id]._meta["_model_pars"], self.opt_pars
            )
            for sys_id in flex_futures
        }

        nlp_solver = _combine(flex_controllers, inflex_sum, self.grid, self.opt_pars)

        # # Check current soc to be in limits
        # soc_init = np.array([flex_futures[sys_id]._meta["state"]["soc"] for sys_id in flex_futures])
        # for soc in soc_init:
        #     assert 0. <= soc <= 1., f"Given initial SoC must be in [0., 1.] but is {soc}"


        par_values = {sys_id: {
            "x_init": futures[sys_id]._meta["state"]["soc"],
            "lam_a": [0.] * current_horizon,
            "fc_p_unc": futures[sys_id]._meta["fc"]} for sys_id in flex_futures
        }

        res = _solve(nlp_solver, par_values)
        if False:
            import matplotlib.pyplot as plt
            plt.plot(inflex_sum)
            flex_sol = np.array([ag_res["yk"] for ag_res in res.values()]).sum(axis=0)
            plt.plot(flex_sol)
            plt.plot(flex_sol + inflex_sum)
            plt.show()

        self.constraint_value = np.abs(np.array([ag_res["yk"] for ag_res in res.values()]).sum(axis=1)).sum() / self.horizon

        self.current_signal_time = list(futures.values())[0]._meta["state"]["t"]

        ret_flex = {sys_id: signals.DirectControlSignal(res[sys_id]["uk"]) for sys_id in res}
        ret_inflex = {sys_id: signals.DirectControlSignal(np.zeros(current_horizon)) for sys_id in inflex_futures}

        return {**ret_flex, **ret_inflex}


def _combine(
    single_controllers: dict[str, local_solver_pv_bat.CasadiSolverLocalPVBatVectorized],
    inflex_sum: np.ndarray,
    grid_descr: type_defs.GridDescription,
    opt_pars: type_defs.OptParameters,
) -> mycas.MyNLPSolver:

    states = []
    constraints = []
    pars = {}
    obj = 0.0

    grid_combined = 0.

    for ag_tag in single_controllers:
        (_obj, _states, _constraints, _pars), _grid_var = \
                single_controllers[ag_tag].get_central_problem_contribution()

        grid_combined += _grid_var.sx

        obj += _obj
        states += _states
        constraints += _constraints

        pars.update({f"{ag_tag}_{par_name}": _pars[par_name] for par_name in _pars})

    coeff_slack = 100.

    slack_ub = mycas.MySX("s_g_ub", 0., np.infty, horizon=len(inflex_sum))
    slack_lb = mycas.MySX("s_g_lb", 0., np.infty, horizon=len(inflex_sum))
    states += [slack_ub, slack_lb]

    constraints += [
        # Upper bound of grid
        mycas.MyConstr(
            grid_combined + inflex_sum - slack_ub.sx, -grid_descr.p_lim, grid_descr.p_lim, "Grid upper"
        ),
        # lower bound
        mycas.MyConstr(
            grid_combined + inflex_sum + slack_lb.sx, -grid_descr.p_lim, grid_descr.p_lim, "Grid lower"
        )
    ]
    obj += mycas.dot(slack_ub.sx, np.ones(slack_ub.horizon)*coeff_slack)
    obj += mycas.dot(slack_lb.sx, np.ones(slack_lb.horizon)*coeff_slack)

    return mycas.MyNLPSolver(obj, states, constraints, pars, solver=opt_pars.solver_name)


def _solve(
    solver: mycas.MyNLPSolver,
    par_sets: dict[str, dict]
    ):

    par_values = {
        f"{ag_tag}_{par_name}": par_sets[ag_tag][par_name]
        for ag_tag in par_sets for par_name in par_sets[ag_tag]
    }

    solver.solve(par_values)

    ret_dict = {}
    for ag_tag in par_sets:
        ret_uk = solver.opt_vector(f"p_ch_a_{ag_tag}") - solver.opt_vector(f"p_dch_a_{ag_tag}")

        ret_yg = solver.opt_vector(f"y_g_a_{ag_tag}")
        xterm = solver.opt_vector(f"x_a_{ag_tag}")[-1]

        ret_dict[ag_tag] = {"uk": ret_uk, "yk": ret_yg, "xterm": xterm}
    
    return ret_dict
