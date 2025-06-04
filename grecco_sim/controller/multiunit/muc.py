import sys, os
from typing import Any, Optional

import numpy as np
import casadi
import pandas as pd

from grecco_sim.controller.local_control import LocalControllerBase
from grecco_sim.controller.multiunit.problem import MultiUnitModel

from grecco_sim.simulator import forecaster
from grecco_sim.util import type_defs, signals, configs, build, console
from grecco_sim.util.type_defs import Schedule


class MultiUnitController:
    def __init__(
            self,
            sim_config: configs.SimulationConfiguration,
            ems_config: configs.EMSConfiguration,
            timeseries: pd.DataFrame):

        """ Control (potentially) multiple units of different types. """

        self.sys_id = ems_config.sys_id
        self.opt_pars = sim_config.optimizer_config
        self.sys_config = ems_config
        # ToDo: MuC actually does not need time-series data in init.
        self.unit_ts_data = timeseries

        # ToDo: This should be adjustable in script.
        self.market_config = sim_config.market_config

        self.mathematical_model = MultiUnitModel(
            horizon=self.opt_pars.horizon,
            sys_config=self.sys_config,
            sys_id=self.sys_id,
            opt_pars=self.opt_pars,
            market_config=self.market_config)

        p = self.mathematical_model.lp
        self.solver = build.solver(self.opt_pars, p)

    def get_schedule(
            self,
            signal: signals.Signal,
            forecast: forecaster.NodeForecast,
            state: dict[str, Any]) -> type_defs.Schedule:

        """ Determine planed power consumption as sum over controlled units.

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
            self.mathematical_model = MultiUnitModel(
                horizon=len(signal),
                sys_config=self.sys_config,
                sys_id=self.sys_id,
                opt_pars=self.opt_pars,
                market_config=self.market_config)

            lp = self.mathematical_model.lp
            self.solver = build.solver(self.opt_pars, lp)

        p = dict()

        p[f"inflexible_load_at_{self.sys_id}"] = forecast.baseload

        if signal.is_empty:
            p[f"congestion_penalty_at_{self.sys_id}"] = np.zeros(len(forecast))
        else:
            p[f"congestion_penalty_at_{self.sys_id}"] = signal.mul_lambda

        p[f"solar_irradiance_at_{self.sys_id}"] = forecast.solar_irradiance

        if self.sys_config.pv:
            p[f"inflexible_pv_at_{self.sys_id}"] = forecast.pv

        if self.sys_config.bat:
            p[f"soc_init_at_{self.sys_id}"] = state["bat_soc"]

        if self.sys_config.hp:
            p[f"temp_outside_at_{self.sys_id}"] = forecast.temp_outisde
            p[f"temp_init_at_{self.sys_id}"] = state["hp_temp_in"]

        p = casadi.vertcat(*[p[param_name] for param_name in
                            self.mathematical_model.parameters])

        x_lb, x_ub = self.mathematical_model.state_bounds
        g_lb, g_ub = self.mathematical_model.constraint_bounds

        solution = self.solver(lbx=x_lb, ubx=x_ub, lbg=g_lb, ubg=g_ub, p=p)

        p_grid, p_battery, p_heatpump = None, None, None

        start, end = self.mathematical_model.state_vector_segments[
            f"p_grid_at_{self.sys_id}"]
        p_grid = np.array(solution["x"][start:end]).reshape(len(signal))

        if self.sys_config.bat:
            start, end = self.mathematical_model.state_vector_segments[
                f"p_bat_at_{self.sys_id}"]
            p_battery = np.array(solution["x"][start:end]).reshape(len(signal))

        if self.sys_config.hp:
            start, end = self.mathematical_model.state_vector_segments[
                f"p_heatpump_at_{self.sys_id}"]
            p_heatpump = np.array(solution["x"][start:end]).reshape(len(signal))

        schedule = type_defs.Schedule(
            p_grid=p_grid,
            p_bat=p_battery,
            p_hp=p_heatpump)

        return schedule
