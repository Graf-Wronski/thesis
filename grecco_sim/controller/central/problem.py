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


