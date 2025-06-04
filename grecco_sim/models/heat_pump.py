"""This module provides a model for a heat pump thermal system."""
import abc
from enum import Enum
import pathlib

import numpy as np
import pandas as pd

from grecco_sim.util import configs
from grecco_sim.models import model



class OpMode(Enum):
    """Enum type for modes of heat pump operation (should be somewhat translatable to SGready)"""

    MUST_HEAT = 1
    MUST_COOL = 2
    MUST_OFF = 3
    FREE = 4

class HeatPumpBase(abc.ABC):
    """
    Abstract Base class for heat pump models.

    This class defines an interface for specific heat pump implementations.
    """

    p_in: float = 0  # electric input power
    q_out: float = 0  # thermal output

    @abc.abstractmethod
    def set_operation(self, mode: OpMode, control: float) -> None:
        """Set the operation mode of the heat pump.
        
        Parameters:
            mode: set mode (mandatory heating/cooling or free operation)
            control: set a power for control (the specific implementation can transform this (ON/OFF))
        """



class HeatPumpOnOff(HeatPumpBase):
    """
    This model class represents a simple on-off heat pump model. So the heatpump can either be off and deliver no heat/cold 
    or be on and deliver a constant amount of heat/cold. The heat pump is controlled by a simple on-off controller.

    Parameters: heat_pump_type - select from database.
                To each type, refrigerant, pressure levels and compressor power limits are defined
    """

    def __init__(self, parameters):

        # Input parameters
        self.p_max = parameters.p_max
        self.cop = parameters.cop

        self.p_in = 0.
        self.q_out = 0.

    def set_operation(self, mode: OpMode, control: float) -> None:
        
        assert isinstance(mode, OpMode), "Clean up old stuff!"

        """if mode == OpMode.MUST_HEAT:
            self.p_in = self.p_max
            self.q_out = self.p_max * self.cop
        elif mode == OpMode.MUST_OFF:
            self.p_in = 0.
            self.q_out = 0.
        else:
        # Free operation as controlled.
        if control > 0.: #heating
            self.p_in = control # self.p_max
            self.q_out = control # * self.cop
        elif control <0.: #cooling if permitted
            self.p_in = self.p_max
            self.q_out = -self.p_max * self.cop
        else:
            self.p_in = 0.
            self.q_out = 0.
            
        """
        self.p_in = control  # self.p_max
        self.q_out = self.p_in * self.cop


class ThermalSystem(model.Model):
    """
    This class model represents the thermal system of the household and interacts with the heat pump model
    defined in class Heat_Pump_Thermodynamic and the EMS configured in the controllers
    """

    @property
    def model_type(self) -> str:
        return "hp"

    def __init__(
            self,
            sys_id: str,
            horizon: int,
            dt_h: float,
            config: configs.HeatPumpConfig,
            ts_data: pd.DataFrame):

        super().__init__(sys_id, horizon, dt_h)

        self.data = ts_data
        self.config = config

        # Log the temperature within the building.
        self.temp_in = self.build_param(self.config.initial_temp)

        # ToDo: Mode might be deprecated. Discuss this with Àlvaro and Rebecca.
        self.mode = self.build_param()

        # ToDo: Would be better to handle forecasts on EMS level.
        # Forecasts for outside temperature are given to heat pump.
        temp_out = ts_data["Outside Temperature"].values
        self.temp_out = self.build_param(temp_out)

        solar_irradiance = ts_data["Solar Irradiance"].values
        self.solar_irradiance = self.build_param(solar_irradiance)

        # Store received signals, consumed power and produced heat.
        self.control = self.build_param()
        self.p = self.build_param()
        self.q_hp = self.build_param()

        # Adding building parameters.
        self.absorbance = self.config.absorbance
        self.irradiance_area = self.config.irradiance_area

        self.heat_pump: HeatPumpBase

        # Import the heat pump database
        hp_db_path = pathlib.Path(__file__).parent.absolute()
        hp_db_path = hp_db_path.parent.parent
        hp_db_path = (hp_db_path / "data" / "heat_pump_database" /
                      "heat_pump_database_short_version.csv")
        heat_pump_info = pd.read_csv(hp_db_path, sep=";", index_col= 0)

        self.heat_pump_type = self.config.heat_pump_type

        # Either on-off or variable speed
        self.heat_pump_model = self.config.heat_pump_model

        if self.heat_pump_type is not None:  # Assigning a model if known
            # If the model's name does not exist in the heat pump database, retrieve data from the default model
            if self.heat_pump_type in heat_pump_info.index:
                self.heat_pump_params = heat_pump_info.loc[self.heat_pump_type]
                
            else:
                raise ValueError(
                    f"ERROR: Entered model {self.heat_pump_type} is not in "
                    "the database. Select from {heat_pump_info.index.values}")

            # Define the heat pump model
            if self.heat_pump_model == "on-off":
                self.heat_pump = HeatPumpOnOff(self.heat_pump_params)
            else:
                msg = (f"ERROR: Entered model {self.heat_pump_model} is not "
                       f"valid. Select from 'on-off' or 'variable-speed'")
                raise ValueError(msg)
        else:
            self.heat_pump = HeatPumpOnOff(self.config)  # Initializing onoff when information is limited

    @property
    def p_model(self) -> np.ndarray:
        return self.p

    def apply_control(self, control: dict):
        """ Apply control to the Thermal System. """

        if control["p_hp"] is None:
            msg = f"Invalid control. {control} misses heatpump."
            raise ValueError(msg)

        hp_control = control["p_hp"]

        if self.heat_pump_model == "on-off":
            if hp_control > self.heat_pump.p_max / 2:
                hp_control = self.heat_pump.p_max
            else:
                hp_control = 0.

        self.control[self.t] = hp_control
        self._evolve(hp_control)

    def _get_mode(self) -> OpMode:
        """Make sure that temperature bounds cannot be violated."""

        temp = self.temp_in[self.t]
        if temp < self.config.temp_min_heat:
            return OpMode.MUST_HEAT
        elif temp > self.config.temp_max_heat:
            return OpMode.MUST_OFF
        else:
            return OpMode.FREE

        
        # Please reintegrate if you want to model cooling
        # But the way it was integrated just didn't use the on/off external control
        # if self.mode[self.k - 1] == 1:
        #     if self.temp[self.k] > self.temp_max_heat:
        #         self.mode[self.k] = 0
        #     else:
        #         self.mode[self.k] = 1
        # elif self.mode[self.k - 1] == -1:
        #     if self.temp[self.k] < self.temp_min_cold:
        #         self.mode[self.k] = 0
        #     else:
        #         self.mode[self.k] = -1
        # elif self.mode[self.k - 1] == 0:
        #     if self.temp[self.k] < self.temp_min_heat:
        #         self.mode[self.k] = 1
        #     elif self.temp[self.k] > self.temp_max_cold:
        #         self.mode[self.k] = -1
        #     else:
        #         self.mode[self.k] = 0

    def _evolve(self, control: float):
        """
        Evolve the thermal system model by one time step using the heat pump model
        If the heat pump model is "variable-speed", the power is calculated in the set_power_and_mode function
        and the adjusted schedule self.p_in, a room temperature profile (self.temp) is calculated
        """

        # Determine mode and heat/power of HP
        mode = self._get_mode()
            
        self.heat_pump.set_operation(mode, control)

        # Evolve mode evolution
        self.p[self.t] = self.heat_pump.p_in
        self.q_hp[self.t] = self.heat_pump.q_out

        delta_temp = self.temp_out[self.t] - self.temp_in[self.t]
        heat_transfer = self.config.heat_rate * delta_temp

        irradiation_heat = (self.solar_irradiance[self.t] * self.absorbance *
                            self.irradiance_area / 1000)  # In kW
        heat_gain = heat_transfer + self.heat_pump.q_out + irradiation_heat
        temp_gain =  ((1 / self.config.thermal_mass) * heat_gain * 3600 *
                      self.dt_h)

        self.temp_in[self.t + 1] = self.temp_in[self.t] + temp_gain

    @property
    def state_history(self) -> dict[str, np.ndarray]:

        state_history = dict()
        state_history["temp_in"] = self.temp_in
        state_history["mode"] = self.mode
        state_history["p"] = self.p
        state_history["q_hp"] = self.q_hp
        state_history["control"] = self.control

        return state_history