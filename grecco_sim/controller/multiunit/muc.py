from typing import Any

import numpy as np
import casadi
import pandas as pd

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

        self.now = 0
        self.dth = sim_config.dt_h
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
            market_config=self.market_config,
            now=self.now)

        p = self.mathematical_model.lp
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
                market_config=self.market_config,
                now=self.now)

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

        if self.sys_config.ev_requests:
            lower_lims, upper_lims = self.get_cummulative_ev_lims(
                requests=requests,
                horizon=len(forecast))
            p[f"ev_cum_upper_limit_at_{self.sys_id}"] = upper_lims
            p[f"ev_cum_lower_limit_at_{self.sys_id}"] = lower_lims

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

    def get_cummulative_ev_lims(
            self,
            horizon: int,
            request_list: list[configs.ChargingRequest]) -> (
                tuple)[np.ndarray, np.ndarray]:

        """ Calculate the cummulative upper limits for WLS given requests. """

        lower_limits = np.zeros(horizon)
        upper_limits = np.zeros(horizon)

        # Filter requests relevant to the regarded timeframe.
        requests = [r for r in request_list
                    if r.start_step <= self.now + horizon
                    and r.end_step >= self.now]

        max_kwh_by_t = self.sys_config.ev_charger.p_lim_effective * self.dth

        for i in range(horizon):
            # Get the request that is active at that point in time.
            active_requests = [r for r in requests
                               if r.active_at(i + self.self.now)]

            if len(active_requests) > 1:
                msg = "Charger can handle only one request per time step."
                raise NotImplementedError(msg)

            elif len(active_requests) == 0 or active_requests[0].capacity == 0:
                if i == 0:
                    # Keep limits at 0.
                    continue
                else:
                    lower_limits[i] = lower_limits[i - 1]
                    upper_limits[i] = upper_limits[i - 1]

            else:
                # We have an active request with positive capacity.
                r = active_requests[0]

                # If a request has capacity x and in the future y can be
                # delivered, then this time step has to deliver at least x - y.
                steps_left = r.end_step - (self.now + i)
                debt = r.capacity - steps_left * max_kwh_by_t
                lower_limits[i] = lower_limits[i - 1] + max(debt, 0)

                start = max(r.start_step, self.now)
                start_value = 0 if start == self.now else upper_limits[start - 1]
                before = upper_limits[i - 1]
                max_add = min(max_kwh_by_t, r.capacity + start_value - before)
                max_add = max(max_add, 0)
                upper_limits[i] = upper_limits[i - 1] + max_add

        return lower_limits, upper_limits
