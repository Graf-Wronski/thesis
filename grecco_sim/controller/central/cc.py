import numpy as np
from casadi import casadi

from grecco_sim.util import configs, build, type_defs
from .problem import CentralOptimizationModel

from typing import Any


class CentralController:
    def __init__(
            self,
            opt_config: configs.OptimizerConfiguration,
            ems_configs: dict[str, configs.EMSConfiguration]):

        self.now = 0
        self.config = opt_config

        # Horizon shrinks if time window is too small.
        self.horizon = opt_config.horizon
        self.ems_configs = ems_configs

        self.mathematical_model = CentralOptimizationModel(
            horizon=self.config.horizon,
            opt_pars=self.config,
            ems_configs=ems_configs,
            now=self.now)

        problem = self.mathematical_model.problem
        discrete = self.mathematical_model.discrete
        self.solver = build.solver(self.config, problem, discrete)

    def step(self):
        self.now += 1
        self.mathematical_model.step()

    @property
    def sys_ids(self) -> list[str]:
        return self.mathematical_model.sys_ids

    def get_schedules(
            self,
            state: dict,
            forecast: Any,
            horizon: int):

        # Update model if remaining time steps are less than horizon.
        if horizon < self.horizon:
            self.horizon = horizon
            self.mathematical_model = CentralOptimizationModel(
                horizon=self.horizon,
                opt_pars=self.config,
                ems_configs=self.ems_configs,
                now=self.now)
            problem = self.mathematical_model.problem
            discrete = self.mathematical_model.discrete
            self.solver = build.solver(self.config, problem, discrete)

        # Extract relevant information from state.
        params = self.state_to_params(state, forecast)

        # Solve.
        solution = self.solver(**params)

        # Extract node schedules from solver result.
        schedules = self.solution_to_schedules(solution)

        return schedules

    def state_to_params(
            self,
            state: dict,
            forecast: Any) -> dict[str, np.ndarray]:

        # ToDo: Conflict between global weather and individual load forecasts.

        p = dict()

        p[f"temp_outside"] = forecast.temp_outisde
        p[f"solar_irradiance"] = forecast.solar_irradiance

        for ems in self.ems_configs.values():
            sys_id = ems.sys_id

            p[f"p_baseload_at_{sys_id}"] = forecast.nodes[sys_id].baseload

            if ems.pv:
                p[f"inflexible_pv_at_{sys_id}"] = forecast.nodes[sys_id].pv

            if ems.bat:
                p[f"soc_init_at_{sys_id}"] = state[sys_id]["bat_soc"]

            if ems.hp:
                p[f"temp_init_at_{sys_id}"] = state[sys_id]["hp_temp_in"]

            if ems.ev_requests:
                lower_lims, upper_lims = build.cummulative_ev_lims(
                    now=self.now,
                    horizon=len(forecast.solar_irradiance),
                    config=ems.ev_charger,
                    request_list=ems.ev_requests)

                p[f"ev_cum_upper_limit_at_{sys_id}"] = upper_lims
                p[f"ev_cum_lower_limit_at_{sys_id}"] = lower_lims

        p = casadi.vertcat(*[p[param_name] for param_name in
                             self.mathematical_model.parameters])

        x_lb, x_ub = self.mathematical_model.state_bounds
        g_lb, g_ub = self.mathematical_model.constraint_bounds

        return {"p": p, "lbx": x_lb, "ubx": x_ub, "lbg": g_lb, "ubg": g_ub}

    def solution_to_schedules(self, solution):
        schedules = dict()

        def get_solution_vals(var: str) -> np.ndarray:
            """ Extract variable from solution vector """
            start, end = self.mathematical_model.state_vector_segments[var]
            return np.array(solution["x"][start:end]).reshape(self.horizon)

        for ems_config in self.ems_configs.values():

            p_grid, p_battery, p_heatpump = None, None, None

            p_grid = get_solution_vals(f"p_grid_at_{ems_config.sys_id}")

            if ems_config.bat:
                p_battery = get_solution_vals(f"p_bat_at_{ems_config.sys_id}")

            if ems_config.hp:
                p_heatpump = get_solution_vals(f"p_heatpump_at_"
                                               f"{ems_config.sys_id}")

            if ems_config.ev_charger:
                raise NotImplementedError

            schedule = type_defs.Schedule(
                p_grid=p_grid,
                p_bat=p_battery,
                p_hp=p_heatpump)

            schedules[ems_config.sys_id] = schedule

        return schedules

