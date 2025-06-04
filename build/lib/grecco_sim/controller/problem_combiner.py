"""
DEPRECATED
"""
import warnings
import numpy as np

from grecco_sim.util import type_defs
from grecco_sim.util import style
from grecco_sim.coordinators import coord_interface
from grecco_sim.controller import mycas
from grecco_sim.local_problems import local_solver_pv_bat


class ProblemCombiner(coord_interface.CoordinatorInterface):

    def __init__(self, controller_parameters: type_defs.OptParameters, horizon):
        super().__init__(horizon)

        self.nlp_solver = None
        self.list_ag_tags = []
        self._solver_name = controller_parameters.solver_name

    def has_converged(self, time_index):
        warnings.warn("To be implemented")

    def get_signals(self, futures):
        warnings.warn("To be implemented")

    def combine(self, single_controllers: dict[str, local_solver_pv_bat.CasadiSolverLocalPVBatVectorized], p_lim: float):

        states = []
        constraints = []
        pars = {}
        obj = 0.

        grid_combined = 0.

        self.list_ag_tags = list(single_controllers.keys())

        for ag_tag in single_controllers:
            (_obj, _states, _constraints, _pars), _grid_var = \
                single_controllers[ag_tag].get_central_problem_contribution()
            
            grid_combined += _grid_var.sx

            obj += _obj
            states += _states
            constraints += _constraints

            pars.update({f"{ag_tag}_{par_name}": _pars[par_name] for par_name in _pars})

        coeff_slack = 100.

        slack = mycas.MySX("s_g", 0., mycas.A_VERY_HIGH_NUMBER, horizon=self.horizon)
        states += [slack]
        for k in range(self.horizon):
            constraints += [mycas.MyConstr(grid_combined[k] - slack.sx[k], 0., p_lim)]
            obj += slack.sx[k] * coeff_slack

        self.nlp_solver = mycas.MyNLPSolver(obj, states, constraints, pars, solver=self._solver_name)

    def solve(self, par_values):
        par_values = {f"{ag_tag}_{par_name}": par_values[ag_tag][par_name]
                      for ag_tag in par_values for par_name in par_values[ag_tag]}

        self.nlp_solver.solve(par_values)

        ret_dict = {}
        for ag_tag in self.list_ag_tags:
            ret_uk = self.nlp_solver.opt_vector(
                f"p_ch_a_{ag_tag}"
            ) - self.nlp_solver.opt_vector(f"p_dch_a_{ag_tag}")

            ret_yg = self.nlp_solver.opt_vector(f"y_g_a_{ag_tag}")
            xterm = self.nlp_solver.opt_vector(f"x_a_{ag_tag}")[-1]

            ret_dict[ag_tag] = {"uk": ret_uk, "yk": ret_yg, "xterm": xterm}

            if False:
                ret_x = self.nlp_solver.opt_vars([f"x_k_{k + 1}_a_{ag_tag}" for k in range(self.horizon)])

                ret_xs = self.nlp_solver.opt_vars([f"xs_k_{k}_a_{ag_tag}" for k in range(self.horizon)])
                ret_xf = self.nlp_solver.opt_vars([f"xf_k_{k}_a_{ag_tag}" for k in range(self.horizon)])

                fig, ax = style.styled_plot(figsize="landscape")
                # ax.plot(ret_x, label=ag_tag)
                ax.plot(ret_yg, label="yg", drawstyle="steps-post")
                ax.plot(ret_xs, label="xs", drawstyle="steps-post")
                ax.plot(ret_xf, label="xf", drawstyle="steps-post")
                ax.legend(title=f"X_N: {ret_x[-1]: 0.03f}")
                fig.tight_layout()

        return ret_dict

