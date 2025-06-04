
import abc


class Model(abc.ABC):
    """
    The Model interface
    """

    def __init__(self, sys_id, horizon, dt_h):
        self.sys_id = sys_id
        self.horizon = horizon
        self.dt_h = dt_h

        self.k = 0

    @abc.abstractmethod
    def apply_control(self, control):
        pass

    @abc.abstractmethod
    def get_state(self):
        pass

    @abc.abstractmethod
    def get_output(self):
        pass
