from typing import Any, Dict

import numpy as np

from grecco_sim.controller import mycas
from grecco_sim.util import sig_types
from grecco_sim.util import sig_types
from grecco_sim.local_problems import local_solver_pv_bat, local_solver_thermal_system
from grecco_sim.util import type_defs


class LocalSolverGradientDescent(object):
    """
    This solver extends a local_solver with the ADMM reference power term.

    This class should be generic to a range of different systems.
    """

    def __init__(
        self,
        horizon: int,
        sys_id: str,
        sys_parameters: type_defs.SysParsPVBat,
        controller_pars: type_defs.OptParameters,
    ):

        self.horizon = horizon
        self.sys_id = sys_id

        self.system_type = sys_parameters.system

        self.nlp_solver = self._create_problem(sys_parameters, controller_pars)

    def _create_problem(self, sys_pars: type_defs.SysPars, opt_pars: type_defs.OptParameters) -> mycas.MyNLPSolver:
        """
        Take the local problem and add the reference power parameter.
        Previous signals is cropped and assumed to start at the current time index.
        """
        match self.system_type:
            case "pv_bat":
                if not sys_pars.on_off:
                    single_controller = local_solver_pv_bat.CasadiSolverLocalPVBatVectorized(
                        self.horizon, self.sys_id, sys_pars, opt_pars
                    )
                else:
                    raise ValueError(
                        "When reactivating OnOff algorithm. Implement vectorized version for on/off w/o lambda"
                    )
                    single_controller = local_solver_pv_bat.OnOffPVBatSolver(self.horizon, self.sys_id, sys_pars)
            case "heatpump":
                single_controller = local_solver_thermal_system.SolverThermalSystem(
                    self.horizon, self.sys_id, sys_pars, opt_pars
                )
            case _:
                raise ValueError(f"Unmatched system type {sys_pars.system}")

        (plain_obj, states, constraints, pars), grid_var = single_controller.get_central_problem_contribution()

        pars["lam_a"] = mycas.MyPar(f"lam_a_{self.sys_id}", self.horizon)

        # Add cost term from Lagrangian augmentation.
        obj = 0
        for k in range(self.horizon):
            obj += pars["lam_a"].sx[k] * grid_var.sx[k]

        # user_functions = {"local_objective": plain_obj}

        return mycas.MyNLPSolver(
            obj + plain_obj,
            states,
            constraints,
            pars,
            solver=opt_pars.solver_name,
            # user_functions=user_functions,  # relic from second order. Might be useful
        )

    def solve(self, state: dict[str, Any], forecast: type_defs.Forecast, signal: sig_types.FirstOrderSignal):
        """
        Trigger solving the optimization problem for scheduling.

        The parameterization is depending on the controlled system's type.
        Maybe, in the future, this could be more generic with the OCP factory (local_solver_XXX)
        providing a method how to extract parameters from the state.
        """

        match self.system_type:
            case "pv_bat":
                par_values = {
                    "x_init": [state["soc"]],
                    "fc_p_unc": forecast.fc_res_load,
                    "fc_pv_prod": -np.clip(forecast.fc_res_load, a_min=None, a_max=0),
                }
            case "heatpump":
                par_values = {
                    "temp_init": state["temp"],
                    "temp_outside": forecast.add_fc["temp_outside"],
                    "solar_heat_gain": forecast.add_fc["solar_heat_gain"],
                    "fc_p_unc": forecast.fc_res_load,
                }
            case _:
                raise ValueError(f"Unmatched system type {self.system_type}")

        par_values["lam_a"] = signal.mul_lambda

        self.nlp_solver.solve(par_values)

    def get_u(self) -> np.ndarray:
        """Return schedule just for controlled flexibility"""
        match self.system_type:
            case "pv_bat":
                return self.nlp_solver.opt_vector(f"p_ch_a_{self.sys_id}") - self.nlp_solver.opt_vector(
                    f"p_dch_a_{self.sys_id}"
                )
            case "heatpump":
                return self.nlp_solver.opt_vector(f"p_el_hp_{self.sys_id}")
            case _:
                raise ValueError(f"No return implemented for system type {self.system_type}.")

    def get_yg(self):
        """Return schedule for schedule of grid power."""
        return self.nlp_solver.opt_vector(f"y_g_a_{self.sys_id}")


_created_solvers: Dict[str, LocalSolverGradientDescent] = {}


def get_solver(
    horizon: int,
    sys_id: str,
    sys_parameters: type_defs.SysParsPVBat,
    controller_pars: type_defs.OptParameters,
) -> LocalSolverGradientDescent:
    """Make sure that a solver with a certain configuration is reused."""

    solver_hash = f"{sys_id}_{horizon}"

    if solver_hash in _created_solvers:
        return _created_solvers[solver_hash]
    else:
        _created_solvers[solver_hash] = LocalSolverGradientDescent(horizon, sys_id, sys_parameters, controller_pars)

    return _created_solvers[solver_hash]
