from typing import Any

import numpy as np
import casadi
import pandas as pd

from grecco_sim.controller.central.problem import LocalOptimizationProblem
from grecco_sim.simulator import forecaster
from grecco_sim.util import type_defs, signals, configs, build, logging
from grecco_sim.util.console import suppress_stdout
from grecco_sim.util.type_defs import Schedule


class MultiUnitController:
    def __init__(
        self,
        sim_config: configs.SimulationConfiguration,
        ems_config: configs.EMSConfiguration,
        timeseries: pd.DataFrame,
    ):
        """Control (potentially) multiple units of different types."""

        self.now = 0
        self.dth = sim_config.dt_h
        self.sys_id = ems_config.sys_id
        self.opt_pars = sim_config.optimizer_config
        self.sys_config = ems_config
        # ToDo: MuC actually does not need time-series data in init.
        self.unit_ts_data = timeseries

        # ToDo: This should be adjustable in script.
        self.market_config = sim_config.market_config

        with suppress_stdout():
            self.mathematical_model = LocalOptimizationProblem(
                horizon=self.opt_pars.horizon,
                sys_id=self.sys_id,
                ems_config=self.sys_config,
                opt_pars=self.opt_pars,
                market_config=self.market_config,
                now=self.now,
            )

        p = self.mathematical_model.problem

        # Usage:
        with suppress_stdout():
            self.solver = build.solver(self.opt_pars, p)

    def step(self):
        # ToDo: step is required for charging processes. A clean solution
        #   would probably use preprocessing instead of step.
        self.now += 1
        self.mathematical_model.step()

    def get_schedule(
        self,
        signal: signals.Signal,
        forecast: forecaster.NodeForecast,
        state: dict[str, Any],
    ) -> type_defs.Schedule:
        """Determine planed power consumption as sum over controlled units.

        Args:
            state: State of the underlying households physical system.
            forecast: ...
            signal: Control signal as set by coordinator. Holds information
                such as target power (control) or pricing factor (mul_lambda).

        Returns:
            Schedule: ...
        """

        # Update mathematical model if signal is too short.
        if len(signal) < self.opt_pars.horizon:
            self.mathematical_model = LocalOptimizationProblem(
                horizon=len(signal),
                sys_id=self.sys_id,
                ems_config=self.sys_config,
                opt_pars=self.opt_pars,
                market_config=self.market_config,
                now=self.now,
            )

        p = self.mathematical_model.problem
        with suppress_stdout():
            self.solver = build.solver(self.opt_pars, p)

        p = dict()

        p[f"p_baseload_at_{self.sys_id}"] = forecast.baseload

        if signal.is_empty:
            p[f"congestion_penalty_at_{self.sys_id}"] = np.zeros(len(forecast))
        else:
            p[f"congestion_penalty_at_{self.sys_id}"] = signal.mul_lambda

        p[f"solar_irradiance"] = forecast.solar_irradiance
        p[f"temp_outside"] = forecast.temp_outisde

        if self.sys_config.pv:
            p[f"inflexible_pv_at_{self.sys_id}"] = forecast.pv

        if self.sys_config.bat:
            p[f"soc_init_at_{self.sys_id}"] = state["bat_soc"]

        if self.sys_config.hp:
            p[f"temp_init_at_{self.sys_id}"] = state["hp_temp_in"]

        if self.sys_config.ev_charger:
            lower_lims, upper_lims = build.cummulative_ev_lims(
                now=self.now,
                horizon=len(signal),
                config=self.sys_config.ev_charger,
                request_list=self.sys_config.ev_requests,
            )

            p[f"ev_cum_upper_limit_at_{self.sys_id}"] = upper_lims
            p[f"ev_cum_lower_limit_at_{self.sys_id}"] = lower_lims

        p = casadi.vertcat(
            *[p[param_name] for param_name in self.mathematical_model.parameters]
        )

        x_lb, x_ub = self.mathematical_model.state_bounds
        g_lb, g_ub = self.mathematical_model.constraint_bounds

        with suppress_stdout():
            solution = self.solver(lbx=x_lb, ubx=x_ub, lbg=g_lb, ubg=g_ub, p=p)

        p_grid, p_battery, p_heatpump, p_ev = None, None, None, None

        start, end = self.mathematical_model.state_vector_segments[
            f"p_grid_at_{self.sys_id}"
        ]
        p_grid = np.array(solution["x"][start:end]).reshape(len(signal))

        if self.sys_config.bat:
            start, end = self.mathematical_model.state_vector_segments[
                f"p_bat_at_{self.sys_id}"
            ]
            p_battery = np.array(solution["x"][start:end]).reshape(len(signal))

        if self.sys_config.hp:
            start, end = self.mathematical_model.state_vector_segments[
                f"p_heatpump_at_{self.sys_id}"
            ]
            p_heatpump = np.array(solution["x"][start:end]).reshape(len(signal))

        if self.sys_config.ev_charger:
            start, end = self.mathematical_model.state_vector_segments[
                f"p_ev_at_{self.sys_id}"
            ]
            p_ev = np.array(solution["x"][start:end]).reshape(len(signal))

        schedule = type_defs.Schedule(
            p_grid=p_grid, p_bat=p_battery, p_hp=p_heatpump, p_ev=p_ev
        )

        return schedule
