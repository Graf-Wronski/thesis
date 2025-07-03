import dataclasses
import datetime
import os
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclasses.dataclass
class MarketConfiguration:
    c_supply: float = 0.3  # Costs associated with consuming energy (€/kWh).
    c_feed_in: float = 0.1  # Reward associated with generating energy (€/kWh).
    max_market_iterations: int = 1 # Limits exchange between units and
    # coordinator.

    def as_dict(self):
        as_dict = dict()
        as_dict['c_supply'] = self.c_supply
        as_dict['c_feed_in'] = self.c_feed_in
        as_dict['max_market_iterations'] = self.max_market_iterations
        return as_dict

@dataclasses.dataclass
class OptimizerConfiguration:
    """ Define solver and respective parameters. """

    solver_name: str  # Results of different solvers can vary substantially.
    horizon: int  # Number of time steps the optimizer regards.
    forecast_type: str = "perfect"

    # First order.
    alpha: float = 0.  # Step size in first order methods.

    # Second order.
    rho: float = 0.  # Step size in ADMM and second order.
    mu: float = 0.  # Slack constraint in second order algorithm.

    slack_penalty_thermal: float = 500.  # Penalty for constraint violations.

    # Gurobi.
    gurobi_version: str = "110"  # Tested on gurobi version 110.

    def as_dict(self):
        as_dict = dict()
        as_dict['solver_name'] = self.solver_name
        as_dict['horizon'] = self.horizon
        as_dict['forecast_type'] = self.forecast_type
        as_dict['alpha'] = self.alpha
        as_dict['rho'] = self.rho
        as_dict['mu'] = self.mu
        as_dict['slack_penalty_thermal'] = self.slack_penalty_thermal
        return as_dict

    def __post_init__(self):
        """ Set solver specific variables. """

        if self.solver_name == "gurobi":
            license_path = os.getenv("GUROBI_LICENSE_FILE")

            if license_path is None:
                msg = (
                    "Please set GUROBI_LICENSE_FILE. In conda you can do so"
                    "using\n conda env config vars set "
                    "GUROBI_LICENSE_FILE=<path/to/gurobi.lic>. \n "
                    "Reactivate your environment afterwards.")
                raise ValueError(msg)

            if not Path(license_path).exists():
                msg = f"GUROBI_LICENSE_FILE set incorrectly ({license_path})."
                raise FileNotFoundError(msg)

            os.environ["GUROBI_VERSION"] = self.gurobi_version


@dataclasses.dataclass
class SimulationConfiguration:
    """ Parameterization of a simulation."""

    grid_data_path: Path
    weather_data_path: Path

    # Horizon of the simulation. Loaded data is cropped. To start_time + 15min * horizon
    n_time_steps: int
    # Intended start time of the simulation
    start_time: datetime.datetime

    # Configuration for local optimizers.
    optimizer_config: OptimizerConfiguration

    coordinator_name: str  # out of [central, distributed, none, admm]

    sim_tag: str  # unique identifier of simulation

    # Market configuration has information about supply and feed-in tariffs.
    market_config: MarketConfiguration

    # Path to store simulation output (e.g. time series, analysis results)
    output_dir: Path = Path(__file__).parents[2] / "results" / "default"

    # Simulation time step in datetime timedelta
    step_size: datetime.timedelta = datetime.timedelta(minutes=15)

    # Use previous signals in scheduling to augment local objective
    use_previous_signals: bool = False

    # Plotting parameters
    plot: bool = True
    show: bool = True

    use_pv: bool = True

    # Decide which flexibilities are to be used during Simulation.
    use_heatpumps: bool = False  # Heatpumps are exported from PyPSA grid.
    use_batteries: bool = False  # Batteries are exported from PyPSA grid.
    use_ev: bool = False
    # EVs need additional data.
    f_name = "test_charging_sessions_2023.csv"
    charging_process_path: Path = Path(
        f"/home/carl-wanninger/data/ev/{f_name}")
    ev_capacity_data_path: Optional[Path] = None


    # State space for heat pumps: discrete ("on-off") or continious
    heat_pump_model: str = "discrete"

    def __post_init__(self):
        # Write output in result directory if no absolute path is given.
        if not self.output_dir.is_absolute():
            p = Path(__file__).parents[2] / "results" / self.output_dir
            self.output_file_dir = p

        if isinstance(self.start_time, str):
            self.start_time = datetime.datetime.fromisoformat(self.start_time)

        if self.start_time.tzinfo is None:
            raise ValueError("Specify a time zone for simulation range.")

        # Verify data paths.
        if not self.grid_data_path.exists():
            msg = f"Network data at: {self.grid_data_path}."
            raise FileNotFoundError(msg)
        if not self.weather_data_path.exists():
            msg = f"Weather data at: {self.weather_data_path}."
            raise FileNotFoundError(msg)

    @property
    def dt_h(self) -> float:
        """ Simulation time step in hours as float. """
        return self.step_size.total_seconds() / 3600.

    @property
    def time_index(self) -> pd.DatetimeIndex:
        """ Build time index of simulation steps from inputs. """
        return pd.date_range(
            start=self.start_time,
            freq=self.step_size,
            periods=self.n_time_steps)

    @property
    def plot_dir(self) -> Path:
        return self.output_dir / "plots"

    def as_dict(self) -> dict:
        """ Configuration as dictionary for plotting. """

        as_dict = dict()
        as_dict["sim_tag"] = self.sim_tag

        as_dict["grid_data_path"] = str(self.grid_data_path)
        as_dict["weather_data_path"] = str(self.weather_data_path)
        as_dict["output_dir"] = self.output_dir

        as_dict["n_time_steps"] = self.n_time_steps
        as_dict["start_time"] = self.start_time.isoformat()
        as_dict["step_size"] = self.step_size

        as_dict["coordinator_name"] = self.coordinator_name

        as_dict["use_pv"] = self.use_pv
        as_dict["use_heatpumps"] = self.use_heatpumps
        as_dict["use_batteries"] = self.use_batteries
        as_dict["use_ev"] = self.use_ev

        # Update configuration dictionary with sub configurations.
        as_dict.update(**self.market_config.as_dict())
        as_dict.update(**self.optimizer_config.as_dict())

        return as_dict


@dataclasses.dataclass
class UnitConfiguration:
    """Base data class for system description"""
    name: str

    # Supply and feed in price
    # ToDo: Can units without nodes participate in the market?
    market_config: MarketConfiguration

    # Simulation time step
    dt_h: float

    @property
    def unit_type(self) -> str:
        raise NotImplementedError("Implement 'unit_type'.")


@dataclasses.dataclass
class BaseloadConfig(UnitConfiguration):
    """Parameter class describing a plain household"""
    unit_type: str = "load"


@dataclasses.dataclass
class PVConfig(UnitConfiguration):
    """Parameter class describing a PV unit."""
    unit_type: str = "pv"


@dataclasses.dataclass
class StorageConfig(UnitConfiguration):
    """Parameter class describing a Battery unit."""

    init_soc: float

    # Battery parameters
    capacity: float  # Battery capacity in kWh.
    p_inv: float  # ToDo: What is this - and is it used correctly?

    eff: float = 0.80  # Efficiency of battery charging (Originally: Inverse
    # for discharging.).
    p_lim_dc: float = 10.  # ToDo: Where is this used?
    p_lim_ac: float = 10.  # ToDo: Where is this used?
    on_off: bool = False

    x_lb: float = 0.1  # Minimal SOC for battery.
    x_ub: float = 1.0  # Maximal SOC for battery.

    unit_type: str = "bat"

    def __post_init__(self):
        if self.p_inv <= 0.:
            raise ValueError("Storage power must be greater than 0.")
        if self.p_lim_ac <= 0.:
            raise ValueError("Charging limit must be greater than 0.")


@dataclasses.dataclass
class HeatPumpConfig(UnitConfiguration):
    """Parameter class describing a Heat Pump system."""
    initial_temp: float = 20.0

    # Household parameters          # TODO: Validate data.  This has to be variable among type of households
    thermal_mass: float = 16500  # in kJ/ K
    heat_rate: float = 0.05  # In kW/K  # coefficient determining heat transfer through building hull
    absorbance: float = 0.1  # Percentage of absorbance of solar irradiation
    irradiance_area: float = 10.0  # in m2

    # Heat pump model can either be "on-off" or "variable-speed"
    heat_pump_model: str = "on-off"

    # Parameters used for the control of the heat pump
    temp_min_heat: float = 19.0  # Minimal desired building temperature (°C).
    temp_max_heat: float = 23.0  # Maximal desired building temperature (°C).
    temp_min_cold: float = 22.0  # 22
    temp_max_cold: float = 26.0  # 26

    heat_pump_type: str = None
    temp_lower_bound: float = 17.0  # 17
    temp_upper_bound: float = 25.0  # 25
    p_max: float = 2.0  # (Nominal) operating power. ATM: Either maximum or 0.
    cop: float = 0.6  # 2.5 Coefficient of performance: Heat per load (J/kWh).

    unit_type: str = "heatpump"


@dataclasses.dataclass
class ChargerAndEVConfig(UnitConfiguration):
    """ Since we match charger to vehicle 1:1, ChargerConfig has EV data. """
    ev_name: str

    # Battery parameters
    capacity: float
    p_inv: float

    # Arguments with defaults (eff from SynPro Data, TBC)
    eff: float = 0.93
    p_lim_ac: float = 11.

    # Bounds for state of charge
    x_lb: float = 0.1
    x_ub: float = 1.

    unit_type: str = "ev"

    @property
    def p_lim_effective(self) -> float:
        return self.eff * self.p_lim_ac

@dataclasses.dataclass
class ChargingRequest:
    """ Request capacity between two time steps. """
    start_step: int
    end_step: int
    capacity: float

    def active_at(self, step: int):
        """ Check if a request is active at a given time step. """
        return (self.start_step <= step) and (step <= self.end_step)

    def __post_init__(self):
        if self.end_step < self.start_step:
            raise ValueError("End step must be later than start step.")

@dataclasses.dataclass
class EMSConfiguration:
    """ Unit configurations associated with an EMS. """

    sys_id: str
    horizon: int
    dt_h: float

    market: MarketConfiguration

    baseload: Optional[BaseloadConfig] = None
    pv: Optional[PVConfig] = None
    bat: Optional[StorageConfig] = None
    hp: Optional[HeatPumpConfig] = None
    ev_charger: Optional[ChargerAndEVConfig] = None
    ev_requests: Optional[list[ChargingRequest]] = None

    @property
    def is_inflexible(self):
        """ EMS is inflexible if it has no flexible unit configuration. """
        return (self.bat is None and
                self.hp is None and
                self.ev_charger is None)

    def __post_init__(self):
        if not ((self.ev_charger is None) == (self.ev_requests is None)):
            msg = (f"EV Charger {self.ev_charger} and f{self.ev_requests}"
                   f"inconsistent.")
            raise ValueError(msg)

