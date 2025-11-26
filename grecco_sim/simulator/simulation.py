import pickle
import time
from pathlib import Path

import pandas as pd

from grecco_sim.simulator import dataloader, result, forecaster
from grecco_sim.util import configs, build
from grecco_sim.models import sim_node


class Simulation:
    def __init__(self, sim_config: configs.SimulationConfiguration):
        """

        Args:
            sim_config: Data sources, simulation settings, optimization
                parameters, etc. See SimulationConfiguration class."""

        self.config = sim_config
        self.optimizer_config = sim_config.optimizer_config

        # Current simulation time step.
        self.t = 0

        self.dataloader = dataloader.Dataloader(self.config)
        self.grid = self.dataloader.grid
        # ToDo: Forecaster currently only used by central coordinator.
        self.forecaster = forecaster.Forecaster(
            horizon=self.opt_horizon,
            sim_dataloader=self.dataloader,
            time_index=self.index,  # type: ignore
        )

        # Download annual Energy Prices from ENTSO-E
        # https://newtransparency.entsoe.eu/
        if sim_config.is_energy_price_static is False:

            file_path = sim_config.energy_price_path
            if not file_path or not Path(file_path).exists():
                raise FileNotFoundError(
                    "Energy price file not found. Please provide a valid path "
                )
            prices = pd.read_csv(file_path)

            prices.columns = prices.columns.str.strip()
            prices["start_time"] = prices["MTU (CET/CEST)"].str.split(" - ").str[0]
            prices["start_time"] = pd.to_datetime(
                prices["MTU (CET/CEST)"]
                .str.split(" - ")
                .str[0]
                .str.split(r" \(")
                .str[0],
                format="%d/%m/%Y %H:%M:%S",
            )

            # get the sequence number (1 or 2)
            prices["Sequence"] = prices["Sequence"].str.extract(r"(\d+)").astype(int)

            prices = prices.drop_duplicates(subset=["start_time", "Sequence"])
            prices = prices.pivot(
                index="start_time",
                columns="Sequence",
                values="Day-ahead Price (EUR/MWh)",
            )
            prices.columns = [f"sequence_{c}" for c in prices.columns]

            # prices = prices.sort_index().asfreq("15min").tz_localize("UTC")
            self.config.market_config.c_supply = (
                prices["sequence_1"]
                .reindex(self.index, method="ffill")
                .to_numpy(dtype=float)
                / 100  # self.MWH_TO_KWH
            )
        # print("c_supply shape:", self.config.market_config.c_supply.shape)
        # print("c_supply sample:", self.config.market_config.c_supply[:5])

        # Nodes are controllable grid participants. As of now: private EMS.
        self.node_ids = self.dataloader.get_sys_ids()
        self.nodes = [self._build_simulation_node(node_id) for node_id in self.node_ids]  # type: ignore

        self.execution_time = 0.0
        self.results = result.SimulationResult(self.config, self.nodes)

        self.coordinator = build.coordinator(self)

    def __str__(self):
        return self.config.sim_tag

    def write(self, p: Path):

        if not (p.exists()):
            p.mkdir(parents=True)

        with open(p / "config.pkl", "wb") as handle:
            pickle.dump(self.config, handle)

        self.grid.write(p)
        self.results.write(p)

    @property
    def index(self) -> None | pd.DatetimeIndex:
        """Timesteps known to simulation."""
        return self.config.time_index

    @property
    def opt_horizon(self) -> int:
        """We need to adapt opt horizon if remaining steps are small."""
        remaining_steps = self.config.n_time_steps - self.t  # type: ignore
        return min(remaining_steps, self.config.optimizer_config.horizon)

    @property
    def ems_configs(self) -> dict[str, configs.EMSConfiguration]:
        return {
            sys_id: self.dataloader.get_ems_config(sys_id) for sys_id in self.node_ids  # type: ignore
        }

    @property
    def state(self) -> dict[str, dict[str, float]]:
        return {str(node): node.state for node in self.nodes}

    def _build_simulation_node(self, node_id: str) -> sim_node.SimulationNode:

        ems_config = self.dataloader.get_ems_config(node_id)
        timeseries = self.dataloader.get_input_data(node_id)

        simulation_node = sim_node.SimulationNode(
            simulation_config=self.config,
            ems_config=ems_config,
            timeseries=timeseries,
        )

        return simulation_node

    def run(self):
        """Run central and run local are not synchronized, yet."""

        print(f"Simulation: {self.config.sim_tag}.")

        while self.t < self.config.n_time_steps:  # type: ignore

            start_time = time.time()

            msg = (
                f"\rSimulation iteration: {self.t} / "
                f"{self.config.n_time_steps - 1}."  # pyright: ignore[reportOptionalOperand]
            )
            print(msg, end="", flush=True)

            signals = self.coordinator.get_signals(self.state)

            for node in self.nodes:
                node_forecast = self.forecaster.get_node_forecast(str(node))
                schedule = node.get_schedule(node_forecast, signals[str(node)])
                node.realize_schedule(schedule)

            self.results.log_iteration_time(time.time() - start_time)

            self.grid.write_loads(self.state, self.t)

            self.step()

        print("\n")
        # Powerflow is called once with all values as it is more efficient.
        # self.grid.n.lpf()

    def step(self) -> None:
        """Progress simulation time."""

        self.coordinator.step()

        for node in self.nodes:
            node.step()

        self.t += 1
        self.forecaster.set_time_window(t=self.t, horizon=self.opt_horizon)
