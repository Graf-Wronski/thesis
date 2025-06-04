"""
This module provides a centralized controller with the goal of limiting power to a peak value.

"""
import warnings
import numpy as np
import casadi

from grecco_sim.util import logger

from grecco_sim.coordinator import coord_interface



class CasadiCentralControllerPeakLimit(coord_interface.Coordinator):
    """
    This controller is a completely centralized controller for the case of limiting grid power to a certain value
    It solves an OCP at every timestep and passes the controls back to the connected nodes.
    """

    def __init__(self, peak_limit, fc_horizon):
        super().__init__(horizon=fc_horizon)
        warnings.warn("__DEPRECATED__")
        self.current_signal_time = -1

        self.peak_limit = peak_limit

    def has_converged(self, time_index):
        return time_index == self.current_signal_time

    def get_signals(self, futures):
        sys_ids = list(futures.keys())

        soc_init = np.array([futures[sys_id]["state"]["soc"] for sys_id in sys_ids])
        for soc in soc_init:
            assert 0. <= soc <= 1., f"Given initial SoC must be in [0., 1.] but is {soc}"
        fc_res_load = np.array([futures[sys_id]["fc"] for sys_id in sys_ids])

        cs_sol = CasadiCentralSolverPeakLimit(*fc_res_load.shape, self.peak_limit)

        controls = cs_sol.solve(soc_init, fc_res_load)

        self.current_signal_time = futures[sys_ids[0]]["state"]["t"]

        return {sys_id: controls[i][0] for i, sys_id in enumerate(sys_ids)}


class CasadiCentralSolverPeakLimit(object):
    def __init__(self, n_agents, horizon, p_lim=40.):
        self.horizon = horizon
        self.n_agents = n_agents
        self.p_lim = p_lim

        self.dt_h = 0.25
        self.capacity = 10.

        self.penalty_slack_grid = 1000.

        self._create_problem()

    def _get_x_next_function(self):
        x = casadi.SX.sym("x", 1)
        u = casadi.SX.sym("u", 1)
        x_next = x + u * self.dt_h / self.capacity
        return casadi.Function('xnext', [x, u], [x_next], ['x0', 'u'], ["x_next"])

    def _create_problem(self):
        c_sup = 0.2
        u_lb = -5.
        u_ub = 5.
        x_lb = 0.
        x_ub = 1.

        a_very_high_number = 270099
        sys_ids = range(self.n_agents)

        f_x_next = self._get_x_next_function()

        states = []
        constraints = []

        self.pars = {}

        list_xk = [[mycas.MySX(f"x_a_{a}_k_0", x_lb, x_ub)] for a in sys_ids]
        states += [list_xk[a][-1] for a in sys_ids]

        self.pars.update({f"x_init_a_{a}": mycas.MyPar(f"x_init_a_{a}") for a in sys_ids})
        constraints += [mycas.MyConstr(list_xk[a][0].sx - self.pars[f"x_init_a_{a}"].sx, 0., 0.) for a in sys_ids]

        # Forecast of uncontrollable load on site
        self.pars.update({f"fc_p_unc_a_{a}": mycas.MyPar(f"fc_p_unc_a_{a}", self.horizon) for a in sys_ids})

        obj = 0
        list_grid_power_overall = [0] * self.horizon

        # Formulate the NLP
        for a in sys_ids:
            for k in range(self.horizon):
                # New NLP variable for the control
                uk = mycas.MySX(f"u_a_{a}_k_{k}", u_lb, u_ub, discrete=False)
                states += [uk]

                res_power = uk.sx + self.pars[f"fc_p_unc_a_{a}"].sx[k]
                obj += res_power * c_sup

                list_grid_power_overall[k] += res_power

                # ============== Local variable and Constraints
                xk = mycas.MySX(f"x_a_{a}_k_{k+1}", x_lb, x_ub)
                states += [xk]
                list_xk[a] += [xk]

                # Constrain state evolution
                x_plus = f_x_next(x0=list_xk[a][-2].sx, u=uk.sx)["x_next"]
                constraints += [mycas.MyConstr(x_plus - xk.sx, 0., 0.)]

        for k in range(self.horizon):
            slack_grid_k = mycas.MySX(f"slack_grid_k_{k}", 0., a_very_high_number)
            states += [slack_grid_k]
            constraints += [mycas.MyConstr(
                list_grid_power_overall[k], -a_very_high_number, self.p_lim)]

            obj += slack_grid_k.sx * self.penalty_slack_grid

        # =================== Transform to casadi input ==============================
        # Concatenate decision variables and constraint terms
        w = casadi.vertcat(*[x.sx for x in states])
        # discrete = [x.discrete for x in states]

        g = casadi.vertcat(*[g.expr for g in constraints])
        p = casadi.vertcat(*[self.pars[par_name].sx for par_name in self.pars])

        # Create an NLP solver
        nlp_prob = {'f': obj, 'x': w, 'g': g, 'p': p}
        self.nlp_solver = casadi.nlpsol('nlp_solver', 'ipopt', nlp_prob)

        self.state_names = [x.name for x in states]

        self.x_lb = [x.lb for x in states]
        self.x_ub = [x.ub for x in states]
        self.g_lb = [g.lb for g in constraints]
        self.g_ub = [g.ub for g in constraints]

        self.init_guess = casadi.vertcat(*([0] * len(states)))

    @logger.suppress_output
    # @logger.default_printing
    def _pure_casadi_solve(self, init_guess, parameters):
        sol = self.nlp_solver(
            x0=init_guess,
            lbx=self.x_lb, ubx=self.x_ub, lbg=self.g_lb, ubg=self.g_ub, p=parameters
        )
        return sol

    def solve(self, soc_init, fc_res_load):
        par_values = {f"x_init_a_{a}": soc_init[a] for a in range(self.n_agents)}
        par_values.update({f"fc_p_unc_a_{a}": fc_res_load[a] for a in range(self.n_agents)})

        parameters = casadi.vertcat(*[par_values[par_name] for par_name in self.pars])
        sol = self._pure_casadi_solve(self.init_guess, parameters)
        x = sol["x"]

        def get(var_name):
            idx = self.state_names.index(var_name)
            return float(x[idx])

        # print([f"{key}| {get(key):0.02f}" for key in self.state_names])

        list_u_opt = []
        for a in range(self.n_agents):
            list_u_opt += [np.array([get(f"u_a_{a}_k_{k}") for k in range(self.horizon)])]

        # print([get(f"slack_grid_k_{k}") for k in range(self.horizon)])

        return list_u_opt
