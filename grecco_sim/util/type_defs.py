"""Provide data type definitions."""
import datetime
import json
import dataclasses
import os
import warnings
from typing import Union, List, Dict, Optional
from pathlib import Path

import numpy as np
import pypsa


class EnhancedJSONEncoder(json.JSONEncoder):
    """Provide a way to serialize data classes."""
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, Path):
            return str(o)
        elif isinstance(o, datetime.timedelta):
            return o.total_seconds()
        elif isinstance(o, datetime.datetime):
            return o.isoformat()
        
        return super().default(o)

@dataclasses.dataclass
class RunParameters:
    """ Parameterization of a simulation run."""

    # Horizon of the simulation. Loaded data is cropped. To start_time + 15min * horizon
    sim_horizon: int
    # Intended start time of the simulation
    start_time: datetime.datetime

    # Maximum number of iterations before breaking the market with suboptimal result
    max_market_iterations: int
    # Parameters specifying which algorithm type is run
    coordination_mechanism: str  # out of [central, distributed, none, admm]
    scenario: Dict  # Scenario description. Should at least include a key: 'name' as identififer

    sim_tag: str # unique identifier of simulation

    # The community optimization problem occuring in a certain time step can be saved to a file.
    inspection: Union[List, None] = None
    # path to store simulation output (e.g. time series, analysis results)
    output_file_dir: Path = Path(__file__).parent.parent.parent / "results" / "default"

    # Simulation time step in datetime timedelta
    dt: datetime.timedelta = datetime.timedelta(minutes=15)

    # Use previous signals in scheduling to augment local objective
    use_prev_signals: bool = False
    
    # Plotting parameters
    plot: bool = False
    show: bool = False

    # Debug settings
    profile_run: bool = False  # switch if profiling is requested when running sim.

    def __post_init__(self):
        self.output_file_dir = Path(self.output_file_dir)

        if not self.output_file_dir.is_absolute():
            self.output_file_dir = Path(__file__).parent.parent.parent / "results" / self.output_file_dir

        if isinstance(self.start_time, str):
            self.start_time = datetime.datetime.fromisoformat(self.start_time)
        assert self.start_time.tzinfo is not None, "Specify a time zone for simulation range."
    
    @property
    def dt_h(self) -> float:
        """ Simulation time step in hours as float. """
        return self.dt.total_seconds() / 3600.


@dataclasses.dataclass
class Forecast(object):
    """Datatype for forecast communication."""

    baseload: np.ndarray
    residual_load: np.ndarray

    pv: Optional[np.ndarray] = None
    temp_outside: Optional[np.ndarray] = None
    solar_irradiance: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.residual_load)

ALLOWED_FLEX_TYPES = ["inflexible", "continuous", "discrete"]


@dataclasses.dataclass
class Schedule(object):
    """This is the message, a local agents sends to the coordinator."""

    p_grid: np.ndarray
    p_bat: Optional[np.ndarray] = None
    p_hp: Optional[np.ndarray] = None
    u: np.ndarray = dataclasses.field(default_factory=lambda: np.array([])) # not necessary

    grads: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    # For ALADIN second order method use a jacobian matrix to avoid going against local constraints.
    jacobian: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))

    def __len__(self):
        return self.p_grid.shape[0]

    def __post_init__(self):
        pass  # could be self.validate()

    @property
    def now(self):
        """ Scheduled values for control variables in the next time step. """

        now =  dict()

        now["p_grid"] = self.p_grid[0]

        if self.p_bat is not None:
            now["p_bat"] = self.p_bat[0]

        if self.p_hp is not None:
            now["p_hp"] = self.p_hp[0]

        return now

    def validate(self):
        """Validate that communicated data is suitable for central coordinator."""
        if not (self.p_hp is None) and (self.p_hp < 0).any():
            msg = f"Solver gave negative heatpump power. {self.p_hp}."
            warnings.warn(msg)
            self.p_hp[self.p_hp < 0] = 0.


@dataclasses.dataclass
class GridDescription:
    """This class is used to describe the grid for optimization."""
    # First guess model is just a large grid with one constraint
    p_lim: float
    network: pypsa.Network
    feeder: List[List[str]]  # For each feeder a list of grid_node ids.

@dataclasses.dataclass
class OptParameters:
    """This data class encodes parameters for optimization solvers"""
    solver_name: str

    # coefficient for step size in ADMM and second order 
    rho: float
    # Coefficient for constraint slack in second order algorithm
    mu: float

    # Coefficient for step size in first oder methods
    alpha: float

    horizon: int

    fc_type: str = "perfect"

    slack_penalty_thermal: float = 500.

    gurobi_version: str = "100"  # GrECCo tested on gurobi version 100.

    def __post_init__(self):
        """ Set solver specific variables. """

        if self.solver_name == "gurobi":

            license_path = os.getenv("GUROBI_LICENSE_FILE")

            if license_path is None:
                msg = ("Please set GUROBI_LICENSE_FILE. In conda you can do so"
                       "using\n conda env config vars set "
                       "GUROBI_LICENSE_FILE=<path/to/gurobi.lic>. \n "
                       "Reactivate your environment afterwards.")
                raise ValueError(msg)

            if not Path(license_path).exists():
                msg = f"GUROBI_LICENSE_FILE set incorrectly ({license_path})."
                raise FileNotFoundError(msg)

            os.environ["GUROBI_VERSION"] = self.gurobi_version



@dataclasses.dataclass
class SysPars:
    """Base data class for system description"""
    name: str
    # Simulation time step
    dt_h: float

    # Supply and feed in price
    c_sup: float
    c_feed: float

    @property
    def system(self) -> str:
        raise NotImplementedError("Implement 'system' (system name).")


@dataclasses.dataclass
class SysParsLoad(SysPars):
    """Parameter class describing a plain household"""
    system: str = "load"

@dataclasses.dataclass
class SysParsPV(SysPars):
    """Parameter class describing a PV System"""
    system: str = "pv"


@dataclasses.dataclass
class SysParsPVBat(SysPars):
    """Parameter class describing a Battery system."""

    init_soc: float

    # Battery parameters
    capacity: float # Battery capacity in kWh.
    p_inv: float  # ToDo: What is this - and is it used correctly?

    eff: float = 0.80  # Efficiency of battery charging (Originally: Inverse
    # for discharging.).
    p_lim_dc: float = 10.  # ToDo: Where is this used?
    p_lim_ac: float = 10.  # ToDo: Where is this used?
    on_off: bool = False

    x_lb: float = 0.1  # Minimal SOC for battery.
    x_ub: float = 1.0  # Maximal SOC for battery.

    system: str = "bat"


@dataclasses.dataclass
class SysParsHeatPump(SysPars):
    """Parameter class describing a Heat Pump system."""
    initial_temp: float = 20.0

    # Household parameters          # TODO: Validate data.  This has to be variable among type of households
    thermal_mass: float = 16500   # in kJ/ K
    heat_rate: float = 0.05 # In kW/K  # coefficient determining heat transfer through building hull
    absorbance: float = 0.1  # Percentage of absorbance of solar irradiation
    irradiance_area: float = 10.0 # in m2

    # Heat pump model can either be "on-off" or "variable-speed"
    heat_pump_model: str = "on-off"
    
    # Parameters used for the control of the heat pump
    temp_min_heat: float = 18.0  # Minimal desired building temperature (°C).
    temp_max_heat: float = 23.0  # Maximal desired building temperature (°C).
    temp_min_cold: float = 22.0  # 22
    temp_max_cold: float = 26.0  # 26

    heat_pump_type: str = None
    temp_lower_bound: float = 17.0  # 17
    temp_upper_bound: float = 25.0  # 25
    p_max: float = 2.0  # (Nominal) operating power. ATM: Either maximum or 0.
    cop: float = 0.6  # 2.5 Coefficient of performance: Heat per load (J/kWh).

    system: str = "heatpump"

@dataclasses.dataclass
class SysParsEV(SysPars):
    """Parameter class describing a EV system."""
    # Charging parameters
    init_soc: float
    target_soc: float

    # Battery parameters
    capacity: float
    p_inv: float

    # Arguments with defaults (eff from SynPro Data, TBC)
    eff: float = 0.93
    p_lim_dc:float = 10.
    p_lim_ac:float = 11.

    # Bounds for state of charge
    x_lb: float = 0.1
    x_ub: float = 1.
    
    system: str = "ev"

@dataclasses.dataclass
class BuildingConfiguration:
    absorbance: float = 0.0  # Solar absorbance of building envelope.
    irradiance_area: float = 0.0  # Area that absorbs solar irradiation (m²).
    heat_rate: float = 0.0  # Temperature diffusion rate of envelope.
    thermal_mass: float = 0.0  # Heat storage capacity of building (J/°C).