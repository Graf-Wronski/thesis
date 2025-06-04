"""Provide data type definitions."""
import datetime
import json
import dataclasses
from typing import Union, List, Dict
from pathlib import Path

import numpy as np


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
    # Forecast of residual generation. Is relevant for all households.
    fc_res_load: np.ndarray

    # Additional time series e.g. weather only relevant to certrain systems.
    add_fc: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    @property
    def fc_len(self):
        return len(self.fc_res_load)

ALLOWED_FLEX_TYPES = ["inflexible", "continuous", "discrete"]


@dataclasses.dataclass
class LocalFuture(object):
    """This is the message, a local agents sends to the coordinator."""
    yg: np.ndarray
    u: np.ndarray = dataclasses.field(default_factory=lambda: np.array([])) # not necessary

    grads: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    # For ALADIN second order method use a jacobian matrix to avoid going against local constraints.
    jacobian: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))

    _meta: Dict = dataclasses.field(default_factory=dict)

    flex_type: str = "inflexible"

    @property
    def horizon(self):
        """Return horizon (= future length)."""
        return self._meta["fc"].fc_len

    @property
    def k(self):
        """Return current time step where future starts."""
        return self._meta["state"]["k"]

    def validate(self):
        """Validate that communicated data is suitable for central coordinator."""
        if self.grads is not None:
            assert self.grads.shape == self.yg.shape, "Grid power and gradients must have the same shape."

        if self.jacobian is not None:
            assert self.jacobian.shape[1] == self.yg.shape[1], "Grid power horizon and jacobian horizon must be the same."

        if self.u is not None:
            assert self.u.shape == self.yg.shape, "Grid power and battery power must have the same shape."

        assert (
            self.flex_type in ALLOWED_FLEX_TYPES
        ), f"flex_type (is: '{self.flex_type}') should be in {ALLOWED_FLEX_TYPES}"


@dataclasses.dataclass
class GridDescription:
    """This class is used to describe the grid for optimization."""
    # First guess model is just a large grid with one constraint
    p_lim: float


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

    slack_penalty_thermal = 500.


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
    def system(self):
        """
        Access system name via base class.

        The actual name has to be defined in each inheriting class.
        """
        return self._system

@dataclasses.dataclass
class SysParsLoad(SysPars):
    """Parameter class describing a plain household"""
    _system: str = "load"


@dataclasses.dataclass
class SysParsPVBat(SysPars):
    """Parameter class describing a PV Battery system."""

    init_soc: float

    # Battery parameters
    capacity: float
    p_inv: float

    # Arguments with defaults
    eff: float = 0.9
    p_lim_dc:float = 10.
    on_off: bool = False

    # Bounds for state of charge
    x_lb: float = 0.
    x_ub: float = 1.
    
    _system: str = "pv_bat"


@dataclasses.dataclass
class SysParsHeatPump(SysPars):
    """Parameter class describing a Heat Pump system."""
    initial_temp: float = 20

    # Household parameters          # TODO: Validate data.  This has to be variable among type of households
    thermal_mass: float = 16500   # in kJ/ K
    heat_rate: float = 0.05 # In kW/K  # coefficient determining heat transfer through building hull
    absorbance: float = 0.1  # Percentage of absorbance of solar irradiation
    irradiance_area: float = 10 # in m2

    # Heat pump model can either be "on-off" or "variable-speed"
    heat_pump_model: str = "on-off"
    
    # Parameters used for the control of the heat pump
    temp_min_heat: float = 18
    temp_max_heat: float = 23
    temp_min_cold: float = 22
    temp_max_cold: float = 26

    heat_pump_type: str = None
    temp_lower_bound: float = 17
    temp_upper_bound: float = 25
    p_max:float = 2.0  # electric nominal power
    cop: float = 2.5

    _system: str = "heatpump"

