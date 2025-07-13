import dataclasses
import datetime
import os
from pathlib import Path
from typing import Optional, Literal

from grecco_sim.graph.utils.format import Format
import pandas as pd


@dataclasses.dataclass
class MarketConfiguration:
    c_supply: float = 0.66  # Costs associated with consuming energy (€/kWh).
    c_feed_in: float = 0.33  # Reward associated with generating energy (
    # €/kWh).
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
    alpha: float = 1.

    # Penalty for constraint violations: Only affects relaxed constraints.
    slack_penalty_thermal: float = 500.
    slack_penalty_ev: float = 500.

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

    # Configuration for local optimizers.
    optimizer_config: OptimizerConfiguration

    coordinator_name: str
    sim_tag: str  # unique identifier of simulation

    # Market configuration has information about supply and feed-in tariffs.
    market_config: MarketConfiguration

    # Coordinator temporal resolution.
    temporal_resolution: Literal["cubic"] = "cubic"

    # Path to store simulation output (e.g. time series, analysis results)
    output_dir: Path = Path(__file__).parents[2] / "results" / "default"

    # Capacity limits.
    feeder_lim: Optional[float] = None
    transformer_lim: Optional[float] = None

    # Either [specify step_size, n_time_steps and step_size] or time_index.
    step_size: Optional[datetime.timedelta] = None
    n_time_steps: Optional[int] = None
    start_time: Optional[datetime.datetime] = None
    time_index: Optional[pd.DatetimeIndex] = None

    # Use previous signals in scheduling to augment local objective
    use_previous_signals: bool = False

    # Plotting parameters
    plot: bool = True
    show: bool = True

    use_pv: bool = True

    # Decide which flexibilities are to be used during Simulation.
    use_heatpumps: bool = True  # Heatpumps are exported from PyPSA grid.
    use_batteries: bool = True  # Batteries are exported from PyPSA grid.
    use_ev: bool = True
    # EVs need additional data.
    f_name = "test_charging_sessions_2023.csv"
    charging_request_path: Path = Format().data_root / "ev" / f_name
    ev_capacity_data_path: Optional[Path] = None


    # State space for heat pumps: discrete ("on-off") or continious
    heat_pump_model: Literal["discrete", "continous"] = "discrete"

    def __post_init__(self):
        # Write output in result directory if no absolute path is given.
        if not self.output_dir.is_absolute():
            p = Path(__file__).parents[2] / "results" / self.output_dir
            self.output_file_dir = p

        time_index_given = self.time_index is not None
        time_triple_given = all(x is not None for x in (self.start_time,
                                                        self.step_size,
                                                        self.n_time_steps))

        msg = ("Specify either time_index\n"
               "or start_time, step_size and n_time_steps.\n"
               "Do not specify both.")

        # Enforce mutual exclusivity
        if time_index_given and time_triple_given:
            raise ValueError(msg)
        elif not time_index_given and not time_triple_given:
            raise ValueError(msg)

        if time_triple_given:
            self.time_index = pd.date_range(
                start=self.start_time,
                freq=self.step_size,
                periods=self.n_time_steps)

        if time_index_given:
            time_steps = self.time_index.to_series().diff().dropna()

            if not time_steps.nunique() == 1:
                msg = "TimeIndex must have uniform stepsize."
                raise ValueError(msg)

            step_size = time_steps.iloc[0]
            if step_size != pd.Timedelta(minutes=15):
                raise NotImplementedError

            self.start_time = self.time_index[0]
            self.step_size = step_size
            self.n_time_steps = len(self.time_index)

    @property
    def dt_h(self) -> float:
        """ Simulation time step in hours as float. """
        return self.step_size.total_seconds() / 3600.


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
    p_inv: float  # DC limit of inverter.

    eff: float = 0.95  # Efficiency of battery inverter (charging/discharging).
    capacity: float = 5.  # Battery capacity in kWh.

    init_soc: float = 0.1

    x_lb: float = 0.1  # Minimal SOC for battery.
    x_ub: float = 1.0  # Maximal SOC for battery.

    unit_type: str = "bat"

    @property
    def p_lim_ac(self) -> float:
        return self.p_inv

    @property
    def p_lim_dc(self) -> float:
        return self.eff * self.p_lim_ac


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
    p_inv: float = 11.0 # kW
    eff: float = 0.93  # eff from SynPro Data

    # Bounds for state of charge
    x_lb: float = 0.1
    x_ub: float = 1.

    unit_type: str = "ev"

    @property
    def p_lim_effective(self) -> float:
        return self.eff * self.p_inv

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

