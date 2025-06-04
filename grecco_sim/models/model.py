import abc
from typing import Optional, Union
import math

import numpy as np


class Model(abc.ABC):

    def __init__(self, sys_id, horizon, dt_h):
        self.sys_id = sys_id
        self.horizon = horizon
        self.dt_h = dt_h

        self.t = 0

    def step(self):
        self.t += 1

    @property
    @abc.abstractmethod
    def p_model(self) -> np.ndarray:
        """ P_model is positive when power is drawn and negative else. """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_type(self) -> str:
        """ Model type is frequently used during result interpretation. """
        raise NotImplementedError

    @abc.abstractmethod
    def apply_control(self, control):
        pass

    @property
    def state(self) -> dict[str, float]:
        """ State is the current value of each parameter in state history. """
        state = dict()
        for key, val in self.state_history.items():
            state[key] = val[self.t]

        return state

    @property
    @abc.abstractmethod
    def state_history(self) -> dict[str, np.ndarray]:
        """ A history of past states for result analysis. """
        raise NotImplementedError

    def build_param(
            self,
            initial_value: Optional[Union[float, np.ndarray]] = None) \
            -> np.ndarray:

        """ A parameter has 'np.nan' as default values and length
            'self.horizon + 1' """

        param = np.full(self.horizon + 1, np.nan)

        if initial_value is None:
            return param

        elif isinstance(initial_value, float):
            param[0] = initial_value
            return param

        elif isinstance(initial_value, np.ndarray):
            param[:len(initial_value)] = initial_value
            return param

        else:
            msg = f"Unknown data type {type(initial_value)} for initial value."
            raise ValueError(msg)
