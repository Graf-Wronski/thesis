""" This module provides the main simulator functionality"""
import json
import os
import pickle
import time
from typing import Any, Union

import pandas as pd
import numpy as np

from grecco_sim.simulator import results
from grecco_sim.util.type_defs import RunParameters
from grecco_sim.coordinators import coord_interface
from grecco_sim.models import grid, grid_node
from grecco_sim.util import data_io, type_defs


class Simulator(object):
    """The class perfoming a simulation run."""

    def __init__(
        self,
        grid_nodes: list[grid_node.GridNode],
        coordinator: coord_interface.CoordinatorInterface,
        run_params: RunParameters,
        time_index: pd.DatetimeIndex,
        grid: Union[grid.Grid, None] = None
    ):

        self.horizon = run_params.sim_horizon
        self.run_params = run_params

        self.grid_nodes = grid_nodes
        self.coordinator = coordinator
        self.time_index = time_index
        self.ts = pd.DataFrame(index=self.time_index, columns=["iterations"])
        self.grid = grid

        self.max_market_iterations = run_params.max_market_iterations
        self.inspection_times = (
            run_params.inspection if run_params.inspection is not None else []
        )
        self.execution_time = 0.0
        self.multiprocess = False

    def _write_inspection(self, k):
        """Write a pickled version of the relevant objects for market clearing.
        Use unpickle_debugging.py to inspect and make custom market clearing.
        """

        os.makedirs(self.run_params.output_file_dir, exist_ok=True)

        obj = (self.grid_nodes, self.coordinator, k)

        with open(self.run_params.output_file_dir / f"coordination_state_at_k_{k}.pkl", "wb") as pickle_file:
            pickle.dump(obj, pickle_file)

    def _market_clearing(self, k: int):
        """
        Make a market clearing in a certain time step.

        args:
        :param: k: time step

        """
        # print(f"For {self.coordinator.coord_name} at k={k} | market initialization")

        # Get initial schedule of connected nodes without any central signal
        initial_futures = {
            node.sys_id: node.get_current_future(
                k, self.coordinator.horizon, self.coordinator.get_initial_signal())
            for node in self.grid_nodes
        }

        # for fut in initial_futures.values():
        # fut.validate()

        if k in self.inspection_times:
            self._write_inspection(k)

        signals = self.coordinator.get_signals(initial_futures)

        market_iterations = 1
        while (
            not self.coordinator.has_converged(k)
            and market_iterations <= self.max_market_iterations
        ):
            # Multiprocessing way to get futures is slower
            futures = {
                node.sys_id: node.get_current_future(
                    k, self.coordinator.horizon, signals[node.sys_id]
                )
                for node in self.grid_nodes
            }

            # for fut in futures.values():
            # fut.validate()

            # This function should update if the market has converged
            signals = self.coordinator.get_signals(futures)
            market_iterations += 1

        # if k % 100 == 0:
        print(f"For {self.coordinator.coord_name} at k={k} needed {market_iterations} market iterations")
        self.ts.loc[self.time_index[k], "iterations"] = market_iterations

        return initial_futures, signals

    def run_sim(self):
        """

        A simulation is executed by iterating through the simulation time from 0 to the horizon -1.
        In each step, the coordination mechanism is executed in order to reach the optimum schedule.

        Once this is reached the final signals are broadcast to the agents using the node.apply_control(...)
        function and the nodes are evolved to the next time step.

        :return: nothing
        """

        start_time = time.time()

        for k in np.arange(self.horizon):
            if k == 40:
                print(f"k: {k}")

            initial_futures, signals = self._market_clearing(k)

            # Broadcast binding signals back to the Households/GridNodes and iterate the local systems to the
            # next time step.
            realized_future: dict[str, float] = {}  # Grid power of households in the current time step
            for node in self.grid_nodes:
                node.apply_control(signals[node.sys_id])
                realized_future[node.sys_id] = node.get_grid_power_at(k)

            rew = self.coordinator.get_cost_realization(initial_futures, realized_future, signals)
            # TODO this line entails a high computation time for pandas indexing. -> rewrite using numpy
            self.ts.loc[self.time_index[k], list(rew.keys())] = list(rew.values())

            # TODO @Ramiz: here we would need the grid simulation of that time step
            if self.grid is not None:
                # self.grid.make_sim_one_timestep(realized_future)
                pass

        self.execution_time = time.time() - start_time

    def get_sim_result(self) -> results.SimulationResult:
        """Access simulation results."""
        if self.execution_time <= 0:
            raise TypeError("Perform simulation before accessing results!")

        return results.SimulationResult().from_simulation(
            self.run_params,
            {node.sys_id: node.get_output() for node in self.grid_nodes},
            self.ts,
            {node.sys_id: node.model_input["params"] for node in self.grid_nodes},
            self.time_index,
            self.execution_time
        )
