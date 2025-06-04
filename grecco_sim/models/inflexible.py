import numpy as np
import pandas as pd

from grecco_sim.models import model
from grecco_sim.util import configs


class Inflexible(model.Model):
    """Superclass for inflexible systems like PV and Baseload."""

    def __init__(
            self,
            sys_id: str,
            horizon: int,
            dt_h: float,
            config: configs.UnitConfiguration,
            ts_data: pd.DataFrame):

        super().__init__(sys_id, horizon, dt_h)
        self.config = config

        if not len(ts_data) == self.horizon:
            msg = (f"Length of given time series ({len(ts_data)}) "
                   f"does not match horizon length ({self.horizon})")
            raise ValueError(msg)

        self.p = self.build_param(ts_data.values)

    def apply_control(self, control: dict) -> None:
        """ Inflexible Units only iterate their inner clock. """
        pass

    @property
    def state_history(self) -> dict[str, np.ndarray]:
        return {"p": self.p}

    @property
    def model_type(self) -> str:
        raise NotImplementedError

    @property
    def p_model(self) -> np.ndarray:
        raise NotImplementedError


class PV(Inflexible):
    @property
    def model_type(self) -> str:
        return "pv"

    @property
    def p_model(self) -> np.ndarray:
        """ For inflexible load: p_grid = -p."""
        return -1 * self.p


class Baseload(Inflexible):
    @property
    def model_type(self) -> str:
        return "baseload"

    @property
    def p_model(self) -> np.ndarray:
        """ For inflexible load: p_grid = p. """
        return self.p