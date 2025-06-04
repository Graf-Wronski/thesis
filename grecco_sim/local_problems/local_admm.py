from typing import Any, List, Dict
from collections import Counter
import warnings

import numpy as np

from grecco_sim.util import signals
from grecco_sim.local_problems import local_solver_base, local_solver_pv_bat, local_solver_thermal_system
from grecco_sim.util import type_defs


class LocalADMMSolver(object):
    """
    This solver extends a local_solver with the ADMM reference power term
    """

    def __init__(
        self,
        horizon: int,
        sys_id: str,
        sys_parameters: type_defs.SysParsPVBat,
        controller_pars: type_defs.OptParameters,
        signal_lengths: List[int]
    ):

        self.horizon = horizon
        self.sys_id = sys_id
        self.system_type = sys_parameters.system

        self.nlp_solver = self._create_problem(sys_parameters, controller_pars, signal_lengths)

    def _add_previous_cost_contributions(
        self,
        grid_var,
        opt_pars: type_defs.OptParameters,
        signal_lengths: List[int],
    ):
        pars = {}
        obj = 0

        for signal_len, count in sorted(Counter(signal_lengths).items()):
            for number in range(count):
                pars[f"lam_{signal_len}_{number}"] = mycas.MyPar(f"lam_{signal_len}_{number}", signal_len)
                pars[f"ref_grid_power_{signal_len}_{number}"] = mycas.MyPar(f"ref_grid_power_{signal_len}_{number}", signal_len)

                obj += mycas.dot(pars[f"lam_{signal_len}_{number}"].sx, grid_var.sx[:signal_len])
                obj += (
                    opt_pars.rho
                    / 2.0
                    * mycas.dot(grid_var.sx[:signal_len]- pars[f"ref_grid_power_{signal_len}_{number}"].sx,
                                grid_var.sx[:signal_len]- pars[f"ref_grid_power_{signal_len}_{number}"].sx)
                )

        return obj, pars

    def _create_problem(
        self, sys_pars: type_defs.SysPars, opt_pars: type_defs.OptParameters, signal_lengths: List[int]
    ) -> mycas.MyNLPSolver:
        """
        Take the local problem and add the reference power parameter.
        Previous signals is cropped and assumed to start at the current time index.
        """
        match self.system_type:
            case "bat":
                if not sys_pars.on_off:
                    single_controller:local_solver_base.LocalSolverBase = local_solver_pv_bat.CasadiSolverLocalPVBatVectorized(
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
                raise ValueError(f"Unmatched system type for second order solver: {self.system_type}")

        (plain_obj, states, constraints, pars), grid_var = single_controller.get_central_problem_contribution()

        obj_prev, pars_prev = self._add_previous_cost_contributions(grid_var, opt_pars, signal_lengths)
        plain_obj += obj_prev
        pars.update(pars_prev)

        pars["ref_grid_power"] = mycas.MyPar(
            f"ref_grid_power_{self.sys_id}", self.horizon
        )
        pars["lam_a"] = mycas.MyPar(f"lam_a_{self.sys_id}", self.horizon)

        # Add cost term from Lagrangian augmentation.
        obj: mycas.casadi.SX = 0
        for k in range(self.horizon):
            obj += pars["lam_a"].sx[k] * grid_var.sx[k]
            obj += (
                opt_pars.rho
                / 2.0
                * ((grid_var.sx[k] - pars["ref_grid_power"].sx[k]) ** 2)
            )

        return mycas.MyNLPSolver(
            obj + plain_obj,
            states,
            constraints,
            pars,
            solver=opt_pars.solver_name,
            user_functions=single_controller.get_user_functions(),
        )

    def solve(
        self,
        state: dict[str, Any],
        forecast: type_defs.Forecast,
        signal: signal.SecondOrderSignal,
        prev_signals: List[signal.SecondOrderSignal],
    ):
        """Fill parameter values and solve problem."""
        # FIll physical parameters
        match self.system_type:
            case "pv_bat":
                par_values = {
                    "x_init": [state["soc"]],
                    "fc_p_unc": forecast.residual_load,
                    "fc_pv_prod": -np.clip(forecast.residual_load, a_min=None, a_max=0),
                }
            case "heatpump":
                par_values = {
                    "temp_init": state["temp"],
                    "temp_outside": forecast.extra_args["temp_outside"],
                    "solar_heat_gain": forecast.extra_args["solar_heat_gain"],
                    "fc_p_unc": forecast.residual_load,
                }
            case _:
                raise ValueError(f"Unmatched system type {self.system_type}")

        par_values["lam_a"]=  signal.mul_lambda
        par_values["ref_grid_power"] = signal.res_power_set

        signals_seen: dict[int, int] = {}

        for signal in prev_signals:
            number = signals_seen.get(signals.Signal_len, 0)
            signals_seen[signals.Signal_len] = number + 1
            par_values[f"lam_{signals.Signal_len}_{number}"] = signal.mul_lambda
            par_values[f"ref_grid_power_{signals.Signal_len}_{number}"] = signal.res_power_set

        self.nlp_solver.solve(par_values)
        # print(f"Objective: {self.nlp_solver.opt_objective()}")

    def get_u(self):
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
        return self.nlp_solver.opt_vector(f"y_g_a_{self.sys_id}")

    def get_local_gradients(self):

        # Doesn't make sense: returns state bounds
        # multipliers = self.nlp_solver.opt_multipliers([f"y_g_{k}_a_{self.sys_id}" for k in range(self.horizon)])
        # grads = self.nlp_solver.opt_gradient("local_objective", f"y_g_a_{self.sys_id}")
        # TODO: This calculation is wrong. must be either grad_f or grad_s depending on which variable is active

        xf = self.nlp_solver.opt_vector(f"xf_a_{self.sys_id}")
        xs = self.nlp_solver.opt_vector(f"xs_a_{self.sys_id}")

        grads_f = self.nlp_solver.opt_gradient("local_objective", f"xf_a_{self.sys_id}")[0, :]
        grads_s = self.nlp_solver.opt_gradient("local_objective", f"xs_a_{self.sys_id}")[0, :]

        return - grads_f * (xf>xs) + grads_s * (xs>=xf)

    def get_active_constraint_jacobian(self) -> np.ndarray:
        """Get the gradient of the constraints that are active."""
        jac = []
        for constr in ["u_lb", "u_ub", "x_ub", "x_lb"]:  # xf_ub
            constr_val  = self.nlp_solver.get_custom_function_value(constr)
            constr_val = constr_val[:, 0]  # take all lines but only first column (value is scalar)

            p_ch = self.nlp_solver.opt_vector(f"p_ch_a_{self.sys_id}")
            p_dc = self.nlp_solver.opt_vector(f"p_dch_a_{self.sys_id}")

            # select the active gradient
            constr_gradient = (
                self.nlp_solver.opt_gradient(constr, f"p_ch_a_{self.sys_id}") * (p_ch >= p_dc)
                - self.nlp_solver.opt_gradient(constr, f"p_dch_a_{self.sys_id}") * (p_ch < p_dc)
            )

            jac += constr_gradient[np.where(constr_val >= 0)].tolist()

        return np.array(jac)


_created_solvers: Dict[str, LocalADMMSolver] = {}


def get_solver(
    horizon: int,
    signal_lengths: List[int],
    sys_id: str,
    sys_parameters: type_defs.SysParsPVBat,
    controller_pars: type_defs.OptParameters,
) -> LocalADMMSolver:
    """Make sure that a solver with a certain configuration is reused."""

    solver_hash = f"{sys_id}_{horizon}_" + "_".join(
        [f"{n}_{count}" for n, count in sorted(Counter(signal_lengths).items())]
    )

    if solver_hash in _created_solvers:
        return _created_solvers[solver_hash]
    else:
        _created_solvers[solver_hash] = LocalADMMSolver(
            horizon, sys_id, sys_parameters, controller_pars, signal_lengths
        )

    return _created_solvers[solver_hash]
