"""
This module provides a centralized controller with the goal of limiting power to a peak value.

"""
import warnings
import numpy as np
import matplotlib.pyplot as plt

from grecco_sim.coordinator import coord_interface

from grecco_sim.util import style


class CasadiCentralControllerPeakLimitDiscrete(coord_interface.Coordinator):
    """
    This controller is a completely centralized controller for the case of limiting grid power to a certain value
    It solves an OCP at every timestep and passes the controls back to the connected nodes.
    """

    def __init__(self, peak_limit, fc_horizon):
        super().__init__(horizon=fc_horizon)
        warnings.warn("__DEPRECATED__")
        self.current_signal_time = -1
        self.dt_h = 0.25

        self.peak_limit = peak_limit

    def has_converged(self, time_index):
        return time_index == self.current_signal_time

    def get_signals(self, futures):

        sys_ids = list(futures.keys())
        fc_res_load = {sys_id: np.array(futures[sys_id]["fc"]) for sys_id in sys_ids}
        horizon = len(fc_res_load[sys_ids[0]])

        print(f"get_signal at k = {futures[sys_ids[0]]['state']['k']}")
        print(futures)

        idx_init = {sys_id: 0 for sys_id in sys_ids}
        soc_init = {sys_id: futures[sys_id]["state"]["soc"] for sys_id in sys_ids}

        idx_term = {sys_id: 0 for sys_id in sys_ids}
        soc_target = {sys_id: futures[sys_id]["state"]["soc"] for sys_id in sys_ids}

        for sys_id in sys_ids:
            if futures[sys_id]["state"]["ev_connected"]:

                assert 0. <= soc_init[sys_id] <= 1., f"Given initial SoC must be in [0., 1.] but is {sys_id}"

                idx_term[sys_id] = int(np.floor(futures[sys_id]["state"]["remaining_time_h"] / self.dt_h))
                soc_target[sys_id] = futures[sys_id]["state"]["target_soc"]

        cs_sol = CasadiCentralSolverPeakLimit(len(sys_ids), horizon, self.peak_limit)
        controls = cs_sol.solve(soc_init, soc_target, idx_init, idx_term, fc_res_load)

        self.current_signal_time = futures[sys_ids[0]]["state"]["t"]

        print(f"Controls: {controls}")
        return controls


class CasadiCentralSolverPeakLimit(object):

    F_MAX = 270099

    def __init__(self, n_agents, horizon, p_lim=40.):
        self.horizon = horizon
        self.n_agents = n_agents
        self.p_lim = p_lim

        self.dt_h = 0.25
        self.capacity = 40.
        self.p_nom_ch = 11.
        self.eff = 0.93

        self.penalty_slack_grid = 1000.

        self.c_sup = 0.2
        self.c_feed = 0.1

        self.x_lb = 0.
        self.x_ub = 1.

    def _ev_subproblem(self, a, idx_init, idx_term, x_init, x_term):

        f_x_next = mycas.function_x_next_ev_on_off(self.p_nom_ch, self.dt_h, self.capacity, self.eff)
        constraints, states, list_xk = [], [], []
        list_grid_power_overall = [0] * self.horizon

        obj = 0

        for k in range(self.horizon):

            if idx_init <= k < idx_term:
                # Defining x_[k+1] in time step k has the advantage that uk can be related to the other powers at k
                if idx_init == k:
                    # Declare x_0 at the connection time point
                    xk = mycas.MySX(f"x_a_{a}_k_{idx_init}", x_init, x_init)
                    states += [xk]
                    list_xk = [xk]

                # Define state in next step and relate to current time steps init soc via charging power uk
                xk_plus1 = mycas.MySX(f"x_a_{a}_k_{k + 1}", self.x_lb, self.x_ub)
                states += [xk_plus1]
                list_xk += [xk_plus1]

                uk = mycas.MySX(f"u_a_{a}_k_{k}", 0., 1, discrete=True)
                states += [uk]
                constraints += [
                    mycas.MyConstr(f_x_next(x0=list_xk[-2].sx, u=uk.sx)["x_next"] - xk_plus1.sx, 0., 0.)
                    # mycas.MyConstr(list_xk[-2].sx + uk.sx * self.p_nom_ch * 1000. - xk_plus1.sx, 0, 0)
                ]

            # Exclude system with an empty charging process (k > 0)
            elif k == idx_term and k > 0:
                constraints += [mycas.MyConstr(list_xk[-1].sx - x_term, 0, self.F_MAX)]
                uk = mycas.MySX(f"u_a_{a}_k_{k}", 0, 1., discrete=False)
                states += [uk]
            else:
                uk = mycas.MySX(f"u_a_{a}_k_{k}", 0, 1., discrete=False)
                states += [uk]

            xfk = mycas.MySX(f"xf_a_{a}_k_{k}", 0, self.F_MAX)
            states += [xfk]
            xsk = mycas.MySX(f"xs_a_{a}_k_{k}", 0, self.F_MAX)
            states += [xsk]

            # Supply - Feedin = load_fc + charging_power
            constraints += [
                mycas.MyConstr(xsk.sx - xfk.sx - self.pars[f"fc_p_unc_a_{a}"].sx[k] - self.p_nom_ch * uk.sx, 0, 0)
            ]

            obj += xsk.sx * self.c_sup - xfk.sx * self.c_feed
            list_grid_power_overall[k] = xsk.sx - xfk.sx

        return constraints, states, list_grid_power_overall, obj

    def _create_problem(self, idx_init, idx_term, x_init, x_term):
        """
        :param k_init: dict sys_id -> first index at which the EV is connected (starting with x_init_a
        :param k_term:
        :return:
        """

        sys_ids = list(idx_init.keys())

        states = []
        constraints = []

        self.pars = {}

        # Forecast of uncontrollable load on site
        self.pars.update({f"fc_p_unc_a_{a}": mycas.MyPar(f"fc_p_unc_a_{a}", self.horizon) for a in sys_ids})

        obj = 0
        list_grid_power_overall = [0] * self.horizon

        # Formulate the NLP
        for a in sys_ids:
            # Formulate the independent subproblems.
            constraints_a, states_a, list_grid_power_overall_a, obj_a = self._ev_subproblem(
                a, idx_init[a], idx_term[a], x_init[a], x_term[a])

            # print(states_a)

            constraints += constraints_a
            states += states_a
            list_grid_power_overall = [list_grid_power_overall_a[i] + list_grid_power_overall[i]
                                       for i in range(len(list_grid_power_overall))]
            obj += obj_a

        # Constraint on peak_limit grid power
        for k in range(self.horizon):
            slack_grid_k = mycas.MySX(f"slack_grid_k_{k}", 0., self.F_MAX)
            states += [slack_grid_k]
            constraints += [mycas.MyConstr(
                list_grid_power_overall[k] - slack_grid_k.sx, -self.F_MAX, self.p_lim)]

            obj += slack_grid_k.sx * self.penalty_slack_grid

        self.nlp_solver = mycas.MyNLPSolver(obj, states, constraints, self.pars)

    def solve(self, soc_init: dict[str: float], soc_target: dict[str: float], idx_init: dict, idx_term: dict,
              fc_res_load: dict[str: np.ndarray]):
        self._create_problem(idx_init, idx_term, soc_init, soc_target)

        par_values = {f"fc_p_unc_a_{a}": fc_res_load[a] for a in soc_init}
        print("========")
        self.nlp_solver.solve(par_values)
        print("=========")

        ret_uk = {
            a: self.nlp_solver.opt_vars([f"u_a_{a}_k_0"]) if idx_init[a] == 0 else 0
            for a in soc_init
        }

        fig, ax = style.styled_plot()
        for a in ret_uk:
            if idx_init[a] == idx_term[a]:
                continue

            print(self.nlp_solver.opt_vars([f"u_a_{a}_k_{k}" for k in np.arange(idx_init[a], idx_term[a])]))
            ax.plot(
                np.arange(idx_init[a], idx_term[a]),
                self.nlp_solver.opt_vars([f"u_a_{a}_k_{k}" for k in np.arange(idx_init[a], idx_term[a])]),
                label=f"uk_{a}")

            ax.plot(
                np.arange(idx_init[a], idx_term[a] + 1),
                self.nlp_solver.opt_vars([f"x_a_{a}_k_{k}" for k in np.arange(idx_init[a], idx_term[a] + 1)]),
                label=f"xk_{a}")
            print(np.arange(idx_init[a], idx_term[a] + 1))

        print(soc_init)
        print(soc_target)
        print(idx_init)
        print(idx_term)
        print(ret_uk)

        ax.legend()
        plt.show()

        return ret_uk


