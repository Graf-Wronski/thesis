"""
This module provides an interface and a few simple implementations to a forecast provider.

The golar of such a class is to provide a forecast using a certain method.
A simple implementation would e.g. be the PerfectForesightForecast.
Here, the get_fc method will just return the actual power time series.
"""

import abc
import numpy as np

from grecco_sim.models import household
from grecco_sim.util import type_defs


class ForecastProvider(abc.ABC):
    """Abstract interface for forecast provider."""

    def __init__(self, household_model: household.Household):
        """Initialize object.

        The forecast ist always specific to a certain household.
        """
        self.hh = household_model

    @abc.abstractmethod
    def get_fc(self, sim_time_step: int, fc_horizon: int) -> type_defs.Forecast:
        """
        Return forecast.

        The returned forecast format is specified in the type definitions.
        All forecasts include a time series for residual load.

        Other time series are optional depending on the system type

        E.g.
        `Forecast(
            fc_res_load=[x, y, z, ...],
            add_fc={"temp_outside": [a, b, c, ...]}
        )`
        """


class PerfectForesightForecast(ForecastProvider):
    """Implementation for perfect foresight forecast."""

    def __init__(self, household_model: household.Household):
    
        """Initializing parent class is sufficient."""
        super().__init__(household_model)

        self.baseload = self.hh.baseload.p.copy()[:-1]

        # Residual load is baseload minus (optional) generation.
        self.residual_load = self.baseload

        if self.hh.pv:
            self.pv = self.hh.pv.p.copy()[:-1]
            self.residual_load -= self.pv

        self.temp_outside = self.hh.temp_out_ts
        self.solar_irradiance = self.hh.irradiance_ts

    def get_fc(self, sim_time_step: int, fc_horizon: int) -> type_defs.Forecast:
        """Return perfect forecast i.e. the realized time series."""

        fc_range = slice(sim_time_step, sim_time_step + fc_horizon)

        baseload = self.baseload[fc_range]
        residual_load = self.residual_load[fc_range]

        if "pv" in self.hh.units:
            pv = self.pv[fc_range] if "pv" in self.hh.units else None
        else:
            pv = None

        temp_outside = self.temp_outside[fc_range]
        solar_irradiance = self.solar_irradiance[fc_range]

        return type_defs.Forecast(
            baseload=baseload,
            residual_load=residual_load,
            pv=pv,
            temp_outside=temp_outside,
            solar_irradiance=solar_irradiance)


class ForecastProviderNaive(ForecastProvider):
    """
    This class generates a naive seasonal forecast by shifting the time series by a certain number of values.
    """

    def __init__(self, household_model: household.Household, shift_by: int = 96) -> \
            None:

        super().__init__(household_model)

        self.res_load = household_model.units["baseload"].load
        if household_model.pv:
            self.res_load -= household_model.units["pv"].p_gen

        self.shifted = np.concatenate(
            [
                self.res_load[-shift_by:],  # takes the last shift_by values
                self.res_load[:-shift_by],  # takes the complete array but the last shift_by values
            ]
        )

        if household_model.hp:
            self.shifted_temp_outside = np.concatenate(
                [self.hh.units["hp"].temp_out[-shift_by:], self.hh.units["hp"].temp_out[:-shift_by]]
            )
            self.shifted_solar_heat_gain = np.concatenate(
                [self.hh.units["hp"].solar_irradiance[-shift_by:], self.hh.units["hp"].solar_irradiance[:-shift_by]]
            )

    def get_fc(self, sim_time_step: int, fc_horizon: int)->type_defs.Forecast:
        """
        Return Naive seasonal forecast.

        This is hacky in some way. The current value should be input to the forecast.
        """

        fc_range = slice(sim_time_step, sim_time_step + fc_horizon)

        fc_res_load = self.shifted[fc_range]
        fc_res_load[0] = self.res_load[sim_time_step]

        fc_ret = {}
        if "hp" in self.hh.units:
            fc_ret["temp_outside"] = self.shifted_temp_outside[fc_range]
            fc_ret["temp_outside"][0] = self.hh.units["hp"].temp_out[sim_time_step]

            fc_ret["solar_heat_gain"] = self.shifted_solar_heat_gain[fc_range]
            fc_ret["solar_heat_gain"][0] = self.hh.units["hp"].solar_irradiance[sim_time_step]

        return type_defs.Forecast(
            fc_res_load,

            extra_args=fc_ret)

# ToDo: Discuss ontological difference between physical model, forecast and
#  household in the context of GrECCo.
def build_forecast_provider(
        forecast_provider_name: str,
        physical_model: household.Household) -> ForecastProvider:

    """ Build ForecastProvider object for simulation data.

    Args:
        forecast_provider_name: The desired ForecastProvider as string.
        physical_model: Forecast is specific to pv units.

    Returns:
        ForecastProvider as Household-specific object.
    """

    match forecast_provider_name:
        case "perfect":
            return PerfectForesightForecast(household_model=physical_model)
        case "naive":
            ForecastProviderNaive(household_model=physical_model)
        case _:
            msg = f"Unknown forecast provider: {forecast_provider_name}."
            raise ValueError(msg)