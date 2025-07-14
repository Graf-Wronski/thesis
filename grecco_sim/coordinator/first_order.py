import numpy as np
from typing import Dict, Any, Callable

from grecco_sim.coordinator import coordinator
from grecco_sim.util import signals, type_defs, build


class GridFeeCoordinator(coordinator.Coordinator):
    """Base class for fee-based coordination methods."""

    def __init__(
            self,
            grecco_sim: Any,
            temporal_resolution: Callable[[np.ndarray], np.ndarray]):

        super().__init__(grecco_sim)
        self.temporal_resolution = temporal_resolution

    @property
    def trafo_p_lim_kw(self) -> float:

        trafo = [x for x in self.sim_grid.capacities.keys()
                 if ("transformer" in x.lower() or "trafo" in x.lower())]

        if len(trafo) != 1:
            msg = "Exactly one transformer is assumed."
            raise NotImplementedError(msg)

        return self.sim_grid.capacities[trafo[0]] * 1000

    @property
    def max_market_iterations(self) -> int:
        return self.sim_config.market_config.max_market_iterations

    def get_signals(self, sim_state: dict[str, dict]) \
            -> dict[str, signals.Signal]:

        raise NotImplementedError

    def has_converged(self, time_index: int) -> bool:
        raise NotImplementedError

    @staticmethod
    def default_signal(signal_length: int) -> signals.FirstOrderSignal:
        """ Default signal is used when no schedules are available. """
        return signals.FirstOrderSignal(np.zeros(signal_length))

    @staticmethod
    def get_grid_fees(
            realization_grid: dict[str, float],
            signals: dict[str, signals.FirstOrderSignal]
        ) -> Dict[str, float]:

        # ToDo: Should grid_fee also be applied to inflexible load?
        grid_fees = dict()
        for sys_id in signals.keys():
            signal = signals[sys_id]
            grid_fee = signal.mul_lambda[0] * realization_grid[sys_id]
            grid_fees[sys_id] = grid_fee

        return grid_fees


class CoordinatorDailyGridFee(GridFeeCoordinator):
    """
    Coordinator for a static grid fee. (Maybe even just two steps HT/NT)
    """

    def get_signals(self, sim_state: dict[str, dict]) \
            -> dict[str, signals.FirstOrderSignal]:

            signals = self.market_clearing()

            return signals

    def market_clearing(self) -> dict[str, signals.FirstOrderSignal]:
        """ Interaction between nodes and coordinator.

        Nodes want to minimize their costs.
        Coordinator wants to minimize congestion. """

        # Default signal is used to create initial schedules.
        default_signal = self.default_signal(self.horizon)
        fee_signals = {str(node): default_signal for node in self.sim_nodes}

        schedules = dict()

        for node in self.sim_nodes:

            node_forecast = self.forecaster.get_node_forecast(str(node))
            schedules[str(node)] = node.get_schedule(
                forecast=node_forecast,
                coordinator_signal=fee_signals[str(node)])

        self.sim_result.log_market(fee_signals, schedules, self.t, 0)

        for k in range(1, self.max_market_iterations):
            fee_signals = self.get_fee_signals(schedules)

            if all([s.is_empty for s in fee_signals.values()]):
                break

            # Update schedules.
            for node in self.sim_nodes:
                node_forecast = self.forecaster.get_node_forecast(str(node))
                schedules[str(node)] = node.get_schedule(
                    forecast=node_forecast,
                    coordinator_signal=fee_signals[str(node)])

            self.sim_result.log_market(fee_signals, schedules, self.t, k)

        return fee_signals

    def get_fee_signals(self, schedules: Dict[str, type_defs.Schedule]) \
            -> Dict[str, signals.FirstOrderSignal]:

        p_grid = np.array([schedule.p_grid for schedule in schedules.values()])
        current_grid_power = p_grid.sum(axis=0)

        # Weight fee signal with parameter alpha.
        #weight = self.sim_config.optimizer_config.alpha
        # lam = np.ones(current_grid_power.shape) * weight
        lam = current_grid_power / self.trafo_p_lim_kw
        # lam[current_grid_power < self.trafo_p_lim] = 0.
        lam = self.temporal_resolution(lam)

        return {sys_id: signals.FirstOrderSignal(mul_lambda=lam)
                for sys_id in schedules}

    def has_converged(self, time_index) -> bool:
        return False


class CoordinatorFeederDependentGridFee(CoordinatorDailyGridFee):
    def get_fee_signals(self, schedules: Dict[str, type_defs.Schedule]) \
            -> Dict[str, signals.FirstOrderSignal]:

        # For grid fee, we require signed congestions.
        _, signed_congestion = self.get_congestion(schedules)
        signed_line_congestion = signed_congestion[self.sim_grid.n.lines.index]

        # Map lines to corresponding feeders.
        mapper = lambda x: self.sim_grid.feeder_map[x]
        feeder_congestion = signed_line_congestion.rename(columns=mapper)

        # A feeder is congested if any of its segments is congested.
        feeder_congestion = feeder_congestion.T.groupby(level=0).sum().T

        # Create signals based on feeder congestion.
        lam = dict()
        for sys_id in schedules:
            bus_name = sys_id.split("_")[-1]
            feeder = self.sim_grid.feeder_map[bus_name]
            lam[sys_id] = self.temporal_resolution(feeder_congestion[feeder])

        return {sys_id: signals.FirstOrderSignal(lam[sys_id])
                for sys_id in schedules}


class Uncoordinated(GridFeeCoordinator):
    def get_signals(self, sim_state: dict[str, dict]) \
            -> dict[str, signals.Signal]:

        return {sys_id: self.default_signal(self.horizon)
                for sys_id in sim_state}
