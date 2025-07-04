from casadi import casadi

from grecco_sim.controller import casadi_model
from grecco_sim.util import configs


class CentralOptimizationModel(casadi_model.CasadiModel):
    def __init__(
            self,
            horizon: int,
            opt_pars: configs.OptimizerConfiguration,
            ems_configs: dict[str, configs.EMSConfiguration],
            now: int):

        super().__init__(horizon, opt_pars, ems_configs, now)

    @property
    def p_trafo(self) -> casadi.SX:
        p_trafo = casadi.SX(0)
        for sys_id in self.sys_ids:
            p_trafo += self.consumption[sys_id]
            p_trafo -= self.generation[sys_id]
        return p_trafo

    def build_objective(self) -> casadi.SX:
        return self.opt_pars.alpha * casadi.dot(self.p_trafo, self.p_trafo)


class LocalOptimizationProblem(casadi_model.CasadiModel):
    def __init__(
            self,
            horizon: int,
            opt_pars: configs.OptimizerConfiguration,
            sys_id: str,
            ems_config: configs.EMSConfiguration,
            market_config: configs.MarketConfiguration,
            now: int):

        self.sys_id = sys_id
        self.config = ems_config
        self.market_config = market_config

        # Superclass expects dict with configs.
        ems_configs = {sys_id: ems_config}
        super().__init__(horizon, opt_pars, ems_configs, now)

    def build_objective(self) -> casadi.SX:
        c_supply = self.market_config.c_supply
        c_feed_in = self.market_config.c_feed_in

        self.objective += c_supply * casadi.sum1(self.consumption[self.sys_id])
        self.objective -= c_feed_in * casadi.sum1(self.generation[self.sys_id])

        var_name = f"congestion_penalty_at_{self.sys_id}"
        lam_congestion = self.build_parameter(var_name, self.horizon)

        if self.config.bat:
            p_bat = self.p_bat[self.sys_id]
            self.objective += casadi.dot(lam_congestion, p_bat)
        if self.config.hp:
            p_heatpump = self.p_heatpump[self.sys_id]
            self.objective += casadi.dot(lam_congestion, p_heatpump)
        if self.config.ev_charger:
            p_ev = self.p_ev[self.sys_id]
            self.objective += casadi.dot(lam_congestion, p_ev)

        return self.objective

    def p_grid(self) -> casadi.SX:
        p_grid = casadi.SX(0)
        for sys_id in self.sys_ids:
            p_grid += self.consumption[sys_id]
            p_grid -= self.generation[sys_id]
        return p_grid

