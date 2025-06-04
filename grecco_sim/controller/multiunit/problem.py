from typing import Optional

import numpy as np
import casadi
from grecco_sim.util import configs


class MultiUnitModel:
    def __init__(
            self,
            horizon: int,
            sys_config: configs.EMSConfiguration,
            sys_id: str,
            opt_pars: configs.OptimizerConfiguration,
            market_config: configs.MarketConfiguration):

        self.horizon = horizon
        self.market_config = market_config
        self.sys_config = sys_config
        self.sys_id = sys_id
        self.opt_pars = opt_pars

        # Each 'state'
        self.states = []
        self.parameters = {}
        self.constraints = []
        self.objective = casadi.SX(0.)

        self.consumption = casadi.SX(np.zeros((horizon, 1)))
        self.generation = casadi.SX(np.zeros((horizon, 1)))

        # Base equations.
        var_name = f"inflexible_load_at_{sys_id}"
        p_inflex_load = self.build_parameter(var_name, self.horizon)
        self.consumption += p_inflex_load

        var_name = f"solar_irradiance_at_{sys_id}"
        solar_irradiance = self.build_parameter(var_name, self.horizon)

        var_name = f"p_feed_in_at_{sys_id}"
        p_feed_in = self.build_state(var_name, bounds=(0., np.infty))

        var_name = f"p_consume_at_{sys_id}"
        p_consume = self.build_state(var_name, bounds=(0., np.infty))

        var_name = f"p_grid_at_{sys_id}"
        p_grid = self.build_state(var_name)

        var_name = f"congestion_penalty_at_{sys_id}"
        lam_congestion = self.build_parameter(var_name, self.horizon)

        if self.sys_config.pv:
            self.add_pv()

        if self.sys_config.bat:
            self.add_battery(lam_congestion)

        if self.sys_config.hp:
            self.add_heatpump(solar_irradiance, lam_congestion)

        if self.sys_config.ev:
            self.add_ev()

        # Central objective is to minimize costs (= maximize profits).
        self.objective += market_config.c_supply * casadi.sum1(p_consume)
        self.objective -= market_config.c_feed_in * casadi.sum1(p_feed_in)

        # Use constraints to determine feed in and load dynamic.
        for k in range(self.horizon):
            self.set_value(p_grid[k], p_consume[k] - p_feed_in[k])
            self.set_value(p_feed_in[k], self.generation[k])
            self.set_value(p_consume[k], self.consumption[k])

    def add_pv(self):
        # ToDo: PV should be calculated by solar irradiation and system size.
        var_name = f"inflexible_pv_at_{self.sys_id}"
        p_inflex_pv = self.build_parameter(var_name, self.horizon)
        self.generation += p_inflex_pv

    def add_battery(self, lam_congestion: np.ndarray):
        # ToDo: p_bat_charge and p_bat_discharge are both constrained by p_inv?
        #   I overtook this from GrECCo - but does it make sense?

        config = self.sys_config.bat
        bat_soc = self.build_state(
            f"soc_bat_at_{self.sys_id}",
            bounds=(config.x_lb, config.x_ub),
            horizon=self.horizon+1)  # ,
        initial_soc = self.build_parameter(f"soc_init_at_{self.sys_id}")
        self.set_value(bat_soc[0], initial_soc)

        var_name = f"p_bat_charge_at_{self.sys_id}"
        p_bat_charge = self.build_state(var_name, bounds=(0., config.p_inv))

        var_name = f"p_bat_discharge_at_{self.sys_id}"
        p_bat_discharge = self.build_state(var_name, bounds=(0., config.p_inv))

        var_name = f"p_bat_at_{self.sys_id}"
        p_bat = self.build_state(var_name)

        # Discharged power is discharge power minus efficiency losses.
        bat_efficiency = config.eff * np.ones(self.horizon)
        p_bat_discharged = casadi.dot(bat_efficiency, p_bat_discharge)

        self.set_value(p_bat, p_bat_charge - p_bat_discharged)

        self.consumption += p_bat_charge
        self.generation += p_bat_discharged

        # Battery state evolution.
        for k in range(self.horizon):
            # p = (p_bat_charge[k] * config.eff - p_bat_discharge[k] * (1 /
            # config.eff))
            p = p_bat[k] * config.dt_h
            delta_soc = (p * config.eff / config.capacity)
            self.set_value(bat_soc[k + 1], bat_soc[k] + delta_soc)

        self.objective += casadi.dot(lam_congestion, p_bat)


    def add_heatpump(self, solar_irradiance: casadi.SX, lam_congestion):
        config = self.sys_config.hp

        var_name = f"temp_outside_at_{self.sys_id}"
        temperature_outside = self.build_parameter(var_name, self.horizon)

        var_name = f"p_heatpump_at_{self.sys_id}"
        p_heatpump = self.build_state(var_name, bounds=(0., config.p_max))
        self.consumption += p_heatpump

        var_name = f"temperature_at_{self.sys_id}"
        temperature = self.build_state(var_name, horizon=self.horizon + 1)

        # Slack variable ensures feasible solution of temperature constraints.
        var_name = f"temperature_slack_at_{self.sys_id}"
        slack_temperature = self.build_state(
            var_name,
            bounds=(0, np.infty),
            horizon=self.horizon + 1)

        self.add_constraint(
            "minimal_temperature",
            temperature + slack_temperature,
            bounds=(config.temp_min_heat, np.infty))

        self.add_constraint(
            "maximal_temperature",
            temperature - slack_temperature,
            bounds=(-np.infty, config.temp_max_heat))

        # Add a penalty for slack relaxation.
        penalty_weight = np.ones(self.horizon + 1)
        penalty_weight *= self.opt_pars.slack_penalty_thermal

        slack_penalty = casadi.dot(slack_temperature, penalty_weight)
        self.objective += slack_penalty

        use_heatpump = self.build_state(
            f"use_heatpump_at_{self.sys_id}",
            bounds=(0., 1.),
            is_discrete=True)
        self.set_value(p_heatpump, use_heatpump * casadi.SX(config.p_max))

        temp_init = self.build_parameter(f"temp_init_at_{self.sys_id}")
        self.set_value(temperature[0], temp_init)

        # Temperature evolution.
        for k in range(self.horizon):
            # ToDo: Explanations for this formulas.

            absorption = config.absorbance * config.irradiance_area / 1000.
            solar_heat_gain = solar_irradiance[k] * absorption

            delta_temp = temperature_outside[k] - temperature[k]
            thermal_diffusion = config.heat_rate * delta_temp

            hp_heating = p_heatpump[k] * config.cop

            heat_gain = solar_heat_gain + thermal_diffusion + hp_heating
            t_in_s = config.dt_h * 60 * 60
            delta_temperature = (heat_gain / config.thermal_mass) * t_in_s

            next_temperature = temperature[k] + delta_temperature
            self.set_value(temperature[k + 1], next_temperature)

        self.objective += casadi.dot(lam_congestion, p_heatpump)


    def add_ev(self):
        # ToDo: EV currently not regarded.
        var_name = "p_ev"
        p_ev = self.build_state(var_name, bounds=(0., 0.))
        self.consumption += p_ev
        raise NotImplementedError("EV planned for multi-unit controller.")


    def build_state(
            self,
            name: str,
            bounds: tuple[float, float] = (-np.infty, np.infty),
            is_discrete: bool = False,
            horizon: Optional[int] = None,
            initial_value: Optional[float] = None) -> casadi.SX:

        if horizon is None:
            horizon = self.horizon

        symbolic_expression = casadi.SX.sym(name, horizon)
        self.states.append(
            {"name": name,
             "lower_bound": bounds[0],
             "upper_bound": bounds[1],
             "is_discrete": is_discrete,
             "horizon": horizon,
             "sx": symbolic_expression,
             "initial_value": initial_value})

        return symbolic_expression

    def add_constraint(
            self,
            name: str,
            sx: casadi.SX,
            bounds: tuple[float, float] = (-np.infty, np.infty)) -> None:

       self.constraints.append(
           {"name": name,
            "sx": sx,
            "lower_bound": bounds[0],
            "upper_bound": bounds[1]})

    def set_value(self, variable: casadi.SX, value: casadi.SX):
        """ By constraining the difference to be zero, we enforce a value. """

        name = f"Set variable {variable.name} to {value.name}."
        self.add_constraint(name, variable - value, bounds=(0., 0.))

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
        """ Interpret state vector, e.g. in solution.

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

    @property
    def lp(self) -> dict[str, casadi.SX]:
        lp = dict()

        # States.
        lp["x"] = casadi.vertcat(*[state['sx'] for state in self.states])

        # Objective.
        lp["f"] = self.objective

        # (Equality) constraints.
        lp["g"] = casadi.vertcat(*[c['sx'] for c in self.constraints])

        # Parameters.
        lp["p"] = casadi.vertcat(*[val for val in self.parameters.values()])

        return lp