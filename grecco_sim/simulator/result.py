from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from grecco_sim.models import sim_node
from grecco_sim.util import configs


class SimulationResult:
    """A class to store and interpret simulation results."""

    def __init__(
        self,
        sim_config: configs.SimulationConfiguration,
        sim_nodes: list[sim_node.SimulationNode],
    ):

        self.sim_len = sim_config.n_time_steps
        self.config = sim_config
        self.sim_nodes = sim_nodes

        self.grid_fees = {str(node): [] for node in self.sim_nodes}
        self.iteration_time = []

        # Log signals and schedules for each node for each market iteration.
        n_market = self.config.market_config.max_market_iterations
        n_optimizer = self.config.optimizer_config.horizon

        def log_tensor() -> dict[str, np.ndarray]:
            market_matrix_shape = (self.sim_len, n_market, n_optimizer)
            return {
                str(node): np.full(market_matrix_shape, np.nan)  # type: ignore
                for node in self.sim_nodes
            }

        self.signals = log_tensor()
        self.p_grid = log_tensor()

        # Log flexible loads for market steps.
        self.p_bat = log_tensor()
        self.p_hp = log_tensor()
        self.p_ev = log_tensor()
        self.market_iterations = np.zeros(self.sim_len)

    def __len__(self):
        self.config.n_time_steps = self.sim_len

    def log_market(self, signals: dict, schedules: dict, t: int, k: int):
        """Log the interactions for each market iteration.

        Args:
            signals:
            schedules:
            t: time step
            k: market_iteration

        """

        self.market_iterations[t] = k

        for node in self.sim_nodes:
            signal, schedule = signals[str(node)], schedules[str(node)]
            self.signals[str(node)][t, k, : len(signal)] = signal.mul_lambda
            self.p_grid[str(node)][t, k, : len(schedule)] = schedule.p_grid

            if node.has_bat:
                self.p_bat[str(node)][t, k, : len(schedule)] = schedule.p_bat

            if node.has_hp:
                self.p_hp[str(node)][t, k, : len(schedule)] = schedule.p_hp

            if node.has_ev:
                self.p_ev[str(node)][t, k, : len(schedule)] = schedule.p_ev

    def log_grid_fees(self, grid_fees: dict):
        """Add grid_fees as dictionaries for each node."""
        for node in grid_fees.keys():
            self.grid_fees[node].append(grid_fees[node])

    def log_iteration_time(self, iteration_time: float):
        """Log the time for a simulation step."""
        self.iteration_time.append(iteration_time)

    @property
    def p_trafo_ts(self) -> pd.DataFrame:
        """Transformer power is sum of nodal powers."""
        return self.state_ts(key2="p_node").sum(axis=1)

    # @functools.cached_property
    @property
    def _state_ts(self) -> pd.DataFrame:
        data = {}

        for node in self.sim_nodes:
            for param_name, param_val in node.state_history.items():
                # Trim parameters that exceed simulation length.
                if len(param_val) == self.config.n_time_steps + 1:
                    param_val = param_val[: self.config.n_time_steps]

                data[f"{str(node)}_{param_name}"] = param_val

        df = pd.DataFrame(data=data, index=self.config.time_index)
        df.index = df.index.tz_localize(None)

        return df

    def state_ts(
        self,
        key1: Optional[str] = None,
        key2: Optional[str] = None,
        key3: Optional[str] = None,
        drop_key: bool = True,
    ) -> pd.DataFrame:
        """Table with simple time-series results."""

        state_ts = self._state_ts.copy()
        keywords = [x for x in [key1, key2, key3] if x]

        for keyword in keywords:
            state_ts = state_ts.filter(like=keyword)

            # Remove superfluent identifiers if desired.
            if drop_key:
                # Remove keyword and clean up string from _x__y_ to x_y.
                cols = [
                    col.replace(keyword, "").replace("__", "_").strip("_")
                    for col in state_ts.columns
                ]
                state_ts.columns = cols

        return state_ts

    @property
    def grid_fee_ts(self) -> pd.DataFrame:
        df = pd.DataFrame(self.grid_fees, index=self.config.time_index)
        # Localized tz data annoys while plotting.
        df.index = df.index.tz_localize(None)
        return df

    def write(self, p: Path):

        self.state_ts().to_csv(p / "state_ts.csv")
        self.state_ts("p_node").to_csv(p / "p_node.csv")
        self.state_ts("p_node").sum(axis=1).to_csv(p / "p_trafo.csv")

        # Detailed unit timeseries.
        self.state_ts("baseload").to_csv(p / "baseload.csv")

        if self.config.use_pv:
            self.state_ts("pv").to_csv(p / "pv.csv")

        # Write soc data for BSS.
        if self.config.use_batteries:
            self.state_ts("bat").to_csv(p / "bss.csv")
            self.state_ts("soc").to_csv(p / "soc.csv")

        # Write temperature data for heat pumps.
        if self.config.use_heatpumps:
            self.state_ts("hp").to_csv(p / "hp.csv")
            self.state_ts("temp").to_csv(p / "temperature.csv")

        # Write request data for EV.
        if self.config.use_ev:
            self.state_ts("ev").to_csv(p / "ev.csv")

        # Write fee data for households.
        if self.config.coordinator_name in ["feeder_fee", "transformer_fee", "mixed"]:
            signal_df = pd.DataFrame(
                {key: val[:, -1, 0].tolist() for key, val in self.signals.items()}
            )
            signal_df.to_csv(p / "realized_signals.csv")
            signal_shape = next(iter(self.signals.values())).shape

            if not (p / "signals").exists():
                (p / "signals").mkdir()
                for i in range(signal_shape[1]):
                    for j in range(signal_shape[2]):
                        signal_df = pd.DataFrame(
                            {
                                key: val[:, i, j].tolist()
                                for key, val in self.signals.items()
                            }
                        )
                        file_name = f"market_{i}_horizon_{j}.csv"
                        signal_df.to_csv(p / "signals" / file_name)
