from typing import Optional

import numpy as np
from casadi import casadi

from grecco_sim.util import configs


class CasadiModel:
    def __init__(
        self,
        horizon: int,
        opt_pars: configs.OptimizerConfiguration,
        # market_config: configs.MarketConfiguration,
        ems_configs: dict[str, configs.EMSConfiguration],
        now: int,
    ):

        self.now = now

        self.ems_configs = ems_configs

        self.horizon = horizon
        self.opt_pars = opt_pars

        # Each 'state'
        self.states = []
        self.parameters = {}
        self.constraints = []
        self.objective = casadi.SX(0.0)

        self.consumption = dict()
        self.generation = dict()
        self.c_supply = dict()

        # Dicts allow reference to loads.
        self.p_baseload = dict()
        self.p_inflex_pv = dict()
        self.p_bat = dict()
        self.p_bat_charge = dict()
        self.p_bat_discharge = dict()
        self.p_heatpump = dict()
        self.p_ev = dict()
        self.p_grid = dict()

        for sys_id in self.sys_ids:
            self.consumption[sys_id] = casadi.SX(np.zeros((horizon, 1)))
            self.generation[sys_id] = casadi.SX(np.zeros((horizon, 1)))
            self.c_supply[sys_id] = casadi.SX(np.zeros((horizon, 1)))

        # Solar irradiation and temperature do not vary over controlled area.
        var_name = f"solar_irradiance"
        self.solar_irradiance = self.build_parameter(var_name, self.horizon)
        var_name = f"temp_outside"
        self.temperature_outside = self.build_parameter(var_name, self.horizon)

        # For each node:
        # Build parameters. (Will be filled with forecast / state during run)
        for ems in self.ems_configs.values():
            self.add_baseload(ems.sys_id)

            if ems.pv:
                self.add_pv(ems.sys_id, ems.pv)

            if ems.bat:
                self.add_battery(ems.sys_id, ems.bat)

            if ems.hp:
                self.add_heatpump(ems.sys_id, ems.hp)

            if ems.ev_charger:
                self.add_ev(ems.sys_id, ems.ev_charger)

        # Use constraints to determine feed in and load dynamic.
        for sys_id in self.sys_ids:
            var_name = f"p_feed_in_at_{sys_id}"
            p_feed_in = self.build_state(var_name, bounds=(0.0, np.infty))

            var_name = f"p_consume_at_{sys_id}"
            p_consume = self.build_state(var_name, bounds=(0.0, np.infty))

            var_name = f"p_grid_at_{sys_id}"
            p_grid = self.build_state(var_name)

            self.set_value(p_feed_in, self.generation[sys_id])
            self.set_value(p_consume, self.consumption[sys_id])
            self.set_value(p_grid, p_consume - p_feed_in)
            self.p_grid[sys_id] = p_grid

        # Add top-level objective to slack objectives (see e.g. heat pump).
        self.objective += self.build_objective()

    def step(self):
        self.now += 1

    def build_objective(self) -> casadi.SX:
        raise NotImplementedError("Optimization objective.")

    @property
    def sys_ids(self) -> list[str]:
        return [ems_config.sys_id for ems_config in self.ems_configs.values()]

    def add_baseload(self, sys_id: str) -> None:
        """Base parameters."""

        var_name = f"p_baseload_at_{sys_id}"
        p_baseload = self.build_parameter(var_name, self.horizon)
        self.p_baseload[sys_id] = p_baseload
        self.consumption[sys_id] += p_baseload

    def add_pv(self, sys_id: str, config: configs.PVConfig) -> None:
        var_name = f"inflexible_pv_at_{sys_id}"
        p_inflex_pv = self.build_parameter(var_name, self.horizon)
        self.p_inflex_pv[sys_id] = p_inflex_pv
        self.generation[sys_id] += p_inflex_pv

    def add_battery(self, sys_id: str, config: configs.StorageConfig) -> None:

        bat_soc = self.build_state(
            f"soc_bat_at_{sys_id}",
            bounds=(config.x_lb, config.x_ub),
            horizon=self.horizon + 1,
        )  # ,
        initial_soc = self.build_parameter(f"soc_init_at_{sys_id}")
        self.set_value(bat_soc[0], initial_soc)

        var_name = f"p_bat_charge_at_{sys_id}"
        charge_bounds = (0.0, config.p_lim_ac)
        p_bat_charge = self.build_state(var_name, bounds=charge_bounds)
        self.p_bat_charge[sys_id] = p_bat_charge

        discharge_bounds = (0.0, config.p_lim_dc)
        var_name = f"p_bat_discharge_at_{sys_id}"
        p_bat_discharge = self.build_state(var_name, bounds=discharge_bounds)
        self.p_bat_discharge[sys_id] = p_bat_discharge

        var_name = f"p_bat_at_{sys_id}"
        p_bat = self.build_state(var_name)
        self.p_bat[sys_id] = p_bat

        self.set_value(p_bat, p_bat_charge - p_bat_discharge)

        self.consumption[sys_id] += p_bat_charge
        self.generation[sys_id] += config.eff * p_bat_discharge

        # Battery state evolution.
        for k in range(self.horizon):
            energy = config.eff * p_bat[k] * config.dt_h
            delta_soc = energy / config.capacity
            self.set_value(bat_soc[k + 1], bat_soc[k] + delta_soc)

    def add_heatpump(self, sys_id: str, config: configs.HeatPumpConfig):

        var_name = f"p_heatpump_at_{sys_id}"
        p_heatpump = self.build_state(var_name, bounds=(0.0, config.p_max))
        self.p_heatpump[sys_id] = p_heatpump
        self.consumption[sys_id] += p_heatpump

        var_name = f"temperature_at_{sys_id}"
        temperature = self.build_state(var_name, horizon=self.horizon + 1)

        # Slack variable ensures feasible solution of temperature constraints.
        var_name = f"temperature_slack_at_{sys_id}"
        slack_temperature = self.build_state(
            var_name, bounds=(0, np.infty), horizon=self.horizon + 1
        )

        self.add_constraint(
            "minimal_temperature",
            temperature + slack_temperature,
            bounds=(config.temp_min_heat, np.infty),
        )

        self.add_constraint(
            "maximal_temperature",
            temperature - slack_temperature,
            bounds=(-np.infty, config.temp_max_heat),
        )

        # Add a penalty for slack relaxation.
        penalty_weight = np.ones(self.horizon + 1)
        penalty_weight *= self.opt_pars.slack_penalty_thermal

        slack_penalty = casadi.dot(slack_temperature, penalty_weight)
        self.objective += slack_penalty

        # Introduce a discrete 'on-off' variable for heat pumps.
        if config.heat_pump_model == "discrete":
            use_heatpump = self.build_state(
                f"use_heatpump_at_{sys_id}", bounds=(0.0, 1.0), is_discrete=True
            )
            self.set_value(p_heatpump, use_heatpump * casadi.SX(config.p_max))

        temp_init = self.build_parameter(f"temp_init_at_{sys_id}")
        self.set_value(temperature[0], temp_init)

        # Temperature evolution.
        for k in range(self.horizon):
            # ToDo: Explanations for this formulas.
            t_in_s = config.dt_h * 60 * 60

            absorption = config.absorbance * config.irradiance_area / 1000.0
            solar_heat_gain = self.solar_irradiance[k] * absorption

            delta_temp = self.temperature_outside[k] - temperature[k]
            thermal_diffusion = config.heat_rate * delta_temp

            hp_heating = p_heatpump[k] * config.cop

            heat_gain = solar_heat_gain + thermal_diffusion + hp_heating
            delta_temperature = (heat_gain / config.thermal_mass) * t_in_s

            next_temperature = temperature[k] + delta_temperature
            self.set_value(temperature[k + 1], next_temperature)

    def add_ev(self, sys_id: str, config: configs.ChargerAndEVConfig):

        var_name = f"ev_cum_upper_limit_at_{sys_id}"
        upper_limits = self.build_parameter(var_name, horizon=self.horizon)

        var_name = f"ev_cum_lower_limit_at_{sys_id}"
        lower_limits = self.build_parameter(var_name, horizon=self.horizon)

        var_name = f"p_ev_at_{sys_id}"
        p_ev = self.build_state(var_name, bounds=(0.0, config.p_inv))
        self.p_ev[sys_id] = p_ev
        var_name = f"p_ev_effective_at_{sys_id}"

        self.consumption[sys_id] += p_ev
        p_ev_effective = self.build_state(var_name, bounds=(0.0, config.p_inv))
        self.set_value(p_ev_effective, casadi.times(config.eff, p_ev))

        # Finns idea: Set lower limits from behind.
        # Calculate: How much demand can I fulfill in future and set lower
        # limit accordingly.

        var_name = f"slack_ev_at_{sys_id}"
        slack_ev = self.build_state(
            var_name, bounds=(0, np.infty), horizon=self.horizon
        )

        penalty_weight = np.ones(self.horizon)
        penalty_weight *= self.opt_pars.slack_penalty_ev

        slack_penalty = casadi.dot(slack_ev, penalty_weight)
        self.objective += slack_penalty

        for i in range(self.horizon):
            self.add_constraint(
                name="ev_lower_charging_limits",
                sx=casadi.sum(p_ev_effective[0 : i + 1]) - lower_limits[i] + slack_ev,
                bounds=(0, np.infty),
            )

            self.add_constraint(
                name="ev_upper_charging_limits",
                sx=casadi.sum(p_ev_effective[0 : i + 1]) - upper_limits[i] - slack_ev,
                bounds=(-np.infty, 0),
            )

    @property
    def discrete(self) -> list[bool]:
        """Casadi requires a list of bools to describe discrete states."""
        discrete = []
        for state in self.states:
            discrete.extend(state["horizon"] * [state["is_discrete"]])
        return discrete

    @property
    def problem(self) -> dict[str, casadi.SX]:
        problem = dict()

        # States.
        problem["x"] = casadi.vertcat(*[state["sx"] for state in self.states])

        # Objective.
        problem["f"] = self.objective

        # (Equality) constraints.
        problem["g"] = casadi.vertcat(*[c["sx"] for c in self.constraints])

        # Parameters.
        problem["p"] = casadi.vertcat(*[val for val in self.parameters.values()])

        return problem

    def build_state(
        self,
        name: str,
        bounds: tuple[float, float] = (-np.infty, np.infty),
        is_discrete: bool = False,
        horizon: Optional[int] = None,
        initial_value: Optional[float] = None,
    ) -> casadi.SX:

        if horizon is None:
            horizon = self.horizon

        symbolic_expression = casadi.SX.sym(name, horizon)
        self.states.append(
            {
                "name": name,
                "lower_bound": bounds[0],
                "upper_bound": bounds[1],
                "is_discrete": is_discrete,
                "horizon": horizon,
                "sx": symbolic_expression,
                "initial_value": initial_value,
            }
        )

        return symbolic_expression

    def add_constraint(
        self,
        name: str,
        sx: casadi.SX,
        bounds: tuple[float, float] = (-np.infty, np.infty),
    ) -> None:

        self.constraints.append(
            {"name": name, "sx": sx, "lower_bound": bounds[0], "upper_bound": bounds[1]}
        )

    def set_value(self, variable: casadi.SX, value: casadi.SX):
        """By constraining the difference to be zero, we enforce a value."""

        name = f"Set variable {variable.name} to {value.name}."
        self.add_constraint(name, variable - value, bounds=(0.0, 0.0))

    def build_parameter(self, name: str, horizon: int = 1) -> casadi.SX:

        symbolic_expression = casadi.SX.sym(name, horizon)
        self.parameters[name] = symbolic_expression

        return symbolic_expression

    @property
    def state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower_bounds, upper_bounds = [], []

        for state in self.states:
            for _ in range(state["horizon"]):
                lower_bounds.append(state["lower_bound"])
                upper_bounds.append(state["upper_bound"])

        return np.array(lower_bounds), np.array(upper_bounds)

    @property
    def state_vector_segments(self) -> dict[str, tuple[int, int]]:
        """Interpret state vector, e.g. in solution.

        Based on an original GrECCo idea (mycas.state_toc).

        Returns:
            dict: Keys are state names. Value are lower and upper index of
                vector segment.
        """

        segmentation = dict()
        start_index, end_index = 0, 0

        for state in self.states:

            for _ in range(state["horizon"]):
                end_index += 1

            segmentation[state["name"]] = start_index, end_index

            # Next state starts at end index.
            start_index = end_index

        return segmentation

    @property
    def constraint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower_bounds, upper_bounds = [], []

        for constraint in self.constraints:
            for _ in range(constraint["sx"].shape[0]):
                lower_bounds.append(constraint["lower_bound"])
                upper_bounds.append(constraint["upper_bound"])

        return np.array(lower_bounds), np.array(upper_bounds)
