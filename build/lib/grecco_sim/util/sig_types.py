"""Definitions around coordination signals."""

import abc
import dataclasses
import numpy as np



@dataclasses.dataclass
class SignalType(abc.ABC):
    """
    This class represents a message from central coordinator to local agent.

    This is the abstract base class with extending classes provided for each CM.
    """

    @abc.abstractmethod
    def validate(self) -> None:
        """Raise an exception if the signal is not valid."""


class NoneSignal(SignalType):
    """Empty Signal specifying no flexibility needed."""
    def validate(self) -> None:
        pass

@dataclasses.dataclass
class DirectControlSignal(SignalType):
    """Signal type used to control flexibility directly from central controller."""

    control: np.ndarray

    @property
    def signal_len(self):
        """Return signal length which is asserted to be consistent."""
        return len(self.control)

    def validate(self) -> None:
        if self.control.ndim != 1:
            raise ValueError("Control must be one dimensional array")


@dataclasses.dataclass
class FirstOrderSignal(SignalType):
    """First order signal aka. Time-varying grid fee."""

    mul_lambda: np.ndarray

    @property
    def signal_len(self):
        """Return signal length which is asserted to be consistent."""
        return len(self.mul_lambda)

    def validate(self) -> None:
        if self.mul_lambda.ndim != 1:
            raise ValueError("Lambda must be one dimensional array")


@dataclasses.dataclass
class SecondOrderSignal(FirstOrderSignal):
    """Signal type used by second order algorithms."""

    res_power_set: np.ndarray
    # init_time: int  # index of first entry wrt. simulation time range

    def validate(self) -> None:
        super().validate()
        if len(self.res_power_set) != len(self.mul_lambda):
            raise ValueError("Signal is invalid due to unequals lengths")
