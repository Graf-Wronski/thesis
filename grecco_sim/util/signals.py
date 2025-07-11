"""Definitions around coordination signals."""

import abc
import dataclasses
import numpy as np

from grecco_sim.util import type_defs


@dataclasses.dataclass
class Signal(abc.ABC):
    """
    This class represents a message from central coordinator to local agent.

    This is the abstract base class with extending classes provided for each CM.
    """

    @abc.abstractmethod
    def validate(self) -> None:
        """ Raise an exception if the signal is not valid."""
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def is_empty(self) -> bool:
        raise NotImplementedError

    def __post_init__(self):
        self.validate()

@dataclasses.dataclass
class DirectControlSignal(Signal):
    """ Hand schedule directly over. """

    schedule: type_defs.Schedule

    @property
    def is_empty(self) -> bool:
        return True

    def __len__(self):
        return len(self.schedule)

    def validate(self) -> None:
        pass


@dataclasses.dataclass
class FirstOrderSignal(Signal):
    """First order signal aka. Time-varying grid fee."""

    mul_lambda: np.ndarray

    def __len__(self):
        return len(self.mul_lambda)

    @property
    def is_empty(self) -> bool:
        """ If no content or all content is 0. """
        return self.__len__() == 0 or not self.mul_lambda.any()

    def validate(self) -> None:
        if self.mul_lambda.ndim != 1:
            raise ValueError("Lambda must be one dimensional array.")
    def __post_init__(self):
        self.mul_lambda = np.clip(self.mul_lambda, a_min=-3, a_max=3)


@dataclasses.dataclass
class SecondOrderSignal(FirstOrderSignal):
    """Signal type used by second order algorithms."""

    res_power_set: np.ndarray
    # init_time: int  # index of first entry wrt. simulation time range

    def validate(self) -> None:
        super().validate()
        if len(self.res_power_set) != len(self.mul_lambda):
            raise ValueError("Signal is invalid due to unequals lengths")
