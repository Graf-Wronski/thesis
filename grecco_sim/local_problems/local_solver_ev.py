import warnings
import matplotlib.pyplot as plt


from grecco_sim.util import style


class CasadiSolverLocalEV(object):
    def __init__(self, horizon):
        self.dt_h = 0.25
        self.capacity = 40.
        self.horizon = horizon

        self.p_nom_ch = 11.
        self.eff = 0.9

        self._create_problem()

    def _create_problem(self):
        c_sup = 0.2
        c_feed = 0.1

        x_lb = 0.
        x_ub = 1.

        a_very_high_number = 270099

        f_x_next = mycas.function_x_next_ev_on_off(self.p_nom_ch, self.dt_h, self.capacity, self.eff)

        states = []
        constraints = []

        self.pars = {}

        list_xk = [mycas.MySX(f"x_k_0", x_lb, x_ub)]
        states += [list_xk[-1]]

        self.pars["x_init"] = mycas.MyPar("p_x_init")
        constraints += [mycas.MyConstr(list_xk[0].sx - self.pars["x_init"].sx, 0, 0)]
        self.pars["x_term"] = mycas.MyPar("p_x_term")

        # Lagrange multiplier
        self.pars["lam_a"] = mycas.MyPar("lam_a", self.horizon)
        # Forecast of uncontrollable load on site
        self.pars["fc_p_unc"] = mycas.MyPar("fc_p_unc", self.horizon)

        self.pars["ev_connected"] = mycas.MyPar("ev_connected", self.horizon)

        obj = 0

        # Formulate the NLP
        for k in range(self.horizon):
            # New NLP variable for the control
            # uk = mycas.MySX(f"u_k_{k}", 0, self.pars["ev_connected"].sx[k], discrete=True)
            uk = mycas.MySX(f"u_k_{k}", 0, 1., discrete=True)
            states += [uk]

            constraints += [mycas.MyConstr(self.pars["ev_connected"].sx[k] - uk.sx, 0., 1.)]

            xfk = mycas.MySX(f"xf_k_{k}", 0, a_very_high_number)
            states += [xfk]
            xsk = mycas.MySX(f"xs_k_{k}", 0, a_very_high_number)
            states += [xsk]

            # Supply - Feedin = load_fc + charging_power
            constraints += [mycas.MyConstr(xsk.sx - xfk.sx - self.pars["fc_p_unc"].sx[k] - self.p_nom_ch * uk.sx, 0, 0)]

            obj += xsk.sx * c_sup - xfk.sx * c_feed

            y_g = mycas.MySX(f"y_g_{k}", -a_very_high_number, a_very_high_number)
            states += [y_g]

            obj += self.pars["lam_a"].sx[k] * y_g.sx
            # Constrain the coupling variable to be larger than supply - feed-in (works only for shaving peak load)
            constraints += [mycas.MyConstr(y_g.sx - xsk.sx + xfk.sx, 0, 0)]

            # ============== Local variable and Constraints
            # The naming here is a little bad: Like this, there are two variables named x_k_0
            # (initial one and after 1 time step)
            xk = mycas.MySX(f"x_k_{k+1}", x_lb, x_ub)
            states += [xk]
            list_xk += [xk]

            # Constrain state evolution
            x_plus = f_x_next(x0=list_xk[-2].sx, u=uk.sx)["x_next"]
            constraints += [mycas.MyConstr(x_plus - xk.sx, 0., 0.)]

        constraints += [mycas.MyConstr(list_xk[-1].sx - self.pars["x_term"].sx, 0, a_very_high_number)]

        # =================== Transform to casadi input ==============================

        self.nlp_solver = mycas.MyNLPSolver(obj, states, constraints, self.pars, solver="gurobi")

    def solve(self, soc_init, soc_target, fc_res_load, signal, ev_connected):
        par_values = {
            "x_init": [soc_init],
            "x_term": [soc_target],
            "lam_a": signal["lambda"],
            "fc_p_unc": fc_res_load,
            "ev_connected": ev_connected
        }

        # print(par_values)

        self.nlp_solver.solve(par_values)
        ret_uk = self.nlp_solver.opt_vars([f"u_k_{k}" for k in range(self.horizon)])
        ret_yg = self.nlp_solver.opt_vars([f"y_g_{k}" for k in range(self.horizon)])

        # fig, ax = style.styled_plot()
        # ax.plot(ret_uk, label="uk")
        # ax.plot(ret_yg, label="yg")
        # ax.plot(fc_res_load, label="fc_res")
        # ax.legend()
        # plt.show()

        return ret_uk, ret_yg


def test_problem():
    horizon = 20
    soc_init = 0.2
    soc_target = 0.8
    fc_res_load = [0.1 * i for i in range(horizon)]
    lam = [0.] * horizon
    ev_connected = [0] + [1] * (horizon - 5) + [0, 0, 0, 0]

    agent = CasadiSolverLocalEV(horizon)
    uk, yk = agent.solve(soc_init, soc_target, fc_res_load, {"lambda": lam}, ev_connected)

    xk = agent.nlp_solver.opt_vars([f"x_k_{k}" for k in range(horizon + 1)])

    fig, ax = style.styled_plot()
    ax.plot(list(range(horizon)), uk, label="uk", drawstyle="steps-post")
    ax.plot(list(range(horizon)), yk, label="yk", drawstyle="steps-post")
    ax.plot(list(range(horizon + 1)), xk, label="SoC", drawstyle="steps-post")
    ax.legend()

    plt.show()


if __name__ == '__main__':
    test_problem()
