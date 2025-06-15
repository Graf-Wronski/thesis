import numpy as np
from typing import Dict, Any

from grecco_sim.coordinator import coordinator
from grecco_sim.util import signals, type_defs
from thesis.graph.utils import network

class GridFeeCoordinator(coordinator.Coordinator):
    """Base class for fee-based coordination methods."""

    def __init__(self, grecco_sim: Any):
        super().__init__(grecco_sim)

    @property
    def trafo_p_lim(self) -> float:

        trafo = [x for x in self.sim_grid.capacities.keys()
                 if "transformer" in x.lower()]

        if len(trafo) != 1:
            msg = "Exactly one transformer is assumed."
            raise NotImplementedError(msg)

        return self.sim_grid.capacities[trafo[0]]

    @property
    def max_market_iterations(self) -> int:
        return self.sim_config.market_config.max_market_iterations

    def get_signals( self, sim_state: dict[str, dict]) \
            -> dict[str, signals.Signal]:

        raise NotImplementedError

    def has_converged(self, time_index: int) -> bool:
        raise NotImplementedError

    @staticmethod
    def empty_signal() -> signals.FirstOrderSignal:
        """Return initial signal for first order methods."""
        return signals.FirstOrderSignal(np.array([]))

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

    @staticmethod
    def default_signal(signal_length: int) -> signals.FirstOrderSignal:
        """ Default signal is used when no schedules are available. """
        return signals.FirstOrderSignal(np.zeros(signal_length))

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
            # ToDo: Implement feeder-specific optimization.
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
        weight = self.sim_config.optimizer_config.alpha
        lam = np.ones(current_grid_power.shape) * weight
        lam[current_grid_power < self.trafo_p_lim] = 0.

        return {sys_id: signals.FirstOrderSignal(mul_lambda=lam)
                for sys_id in schedules}

    def has_converged(self, time_index) -> bool:
        return False


class CoordinatorFeederDependentGridFee(CoordinatorDailyGridFee):
    def get_fee_signals(self, schedules: Dict[str, type_defs.Schedule]) \
            -> Dict[str, signals.FirstOrderSignal]:

        # Use a network copy to avoid side effects.
        n = self.sim_grid.n.copy()
        n.set_snapshots([x for x in range(self.horizon)])

        # Replace loads with aggregates (P grid) to simplify power flow.
        for load in n.loads.index:
            n.remove("Load", name=load)
        for sys_id, schedule in schedules.items():
            bus_id = sys_id.split('_')[-1]
            identifier = {"name": f"P grid at {bus_id}", "bus": bus_id}
            n.add(class_name="Load", p_set=schedule.p_grid, **identifier)

        n.lpf(snapshots=[x for x in range(self.horizon)])

        # ToDo: This could refined for different feeder and line values.
        congested_lines = n.lines_t["p0"].copy()
        for col in congested_lines.columns:
            congested_lines[col].values[:] = 0
        congested_lines[n.lines_t["p0"] < -self.sim_grid.feeder_p_lim] = -1.0
        congested_lines[n.lines_t["p0"] > self.sim_grid.feeder_p_lim] = 1.0


        # Map lines to corresponding feeders.
        mapper = lambda x: self.sim_grid.feeder_map[x]
        feeder_congestion = congested_lines.rename(columns=mapper)

        # A feeder is congested if any of its segments is congested.
        feeder_congestion = feeder_congestion.T.groupby(level=0).sum().T

        # Create signals based on feeder congestion.

        lam = {sys_id: feeder_congestion[self.sim_grid.feeder_map[
            sys_id.split("_")[-1]]].values for sys_id in schedules}

        weight = self.sim_config.optimizer_config.alpha

        return {sys_id: signals.FirstOrderSignal(weight * lam[sys_id] *
                                                 0.25 * np.random.choice(
            np.arange(
                                                     10.0)))
                for sys_id in schedules}
