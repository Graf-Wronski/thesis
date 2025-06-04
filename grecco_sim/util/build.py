import os, sys
from typing import Union, Any, Optional

import pandas as pd

from grecco_sim.coordinator import first_order, central
from grecco_sim.util import configs, console

import casadi

def solver(
        opt_pars: configs.OptimizerConfiguration,
        p: dict[str, casadi.SX],
        discrete: Optional[list[bool]] = None) -> casadi.Function:

    solver_name = opt_pars.solver_name
    solver_options = {"error_on_fail": False, "verbose": False}

    # If discrete variables are marked: Hand them to solver.
    if discrete is not None:
        solver_options["discrete"] = discrete

    if solver_name == 'osqp':
        solver_options["osqp"] = {"verbose": False}
        s = casadi.qpsol("solver", "osqp", p, solver_options)
    elif solver_name == 'gurobi':
        solver_options["gurobi"] = {"TimeLimit": 5, 'OutputFlag': 0,
                                    "LogToConsole": 0}
        s = casadi.qpsol("solver", "gurobi", p, solver_options)
    elif solver_name == "bonmin":
        solver_options["bonmin"] = {'max_iter': 25}
        s = casadi.nlpsol("solver", "bonmin", p, solver_options)
    elif solver_name == 'ipopt':
        solver_options["ipopt"] = {'max_iter': 25}
        s = casadi.nlpsol("solver", "ipopt", p, solver_options)
    else:
        raise ValueError(f'Unknown solver name: {solver_name}')

    return s


def sys_id(bus_name: Union[str, pd.Series]) -> Union[str, pd.Series]:

    if type(bus_name) is str:
        return f"sys_at_bus_{bus_name}"

    if type(bus_name) is pd.Series:
        return bus_name.apply(lambda x: f"sys_at_bus_{x}")


def id_mapping(params: pd.DataFrame, unit: str) -> dict:
    """ Map pypsa.Name to corresponding unit id (= sys_id + _unit). """

    systems = sys_id(params["bus"])
    return {idx: f"{systems[idx]}_{unit}" for idx in params.index}

def coordinator(grecco_sim: Any) -> Any:
    """ Since coordinators require differnet simulation aspects, we simply
    pass the whole simulation. """

    name = grecco_sim.config.coordinator_name

    if name == "transformer_fee":
        return first_order.CoordinatorDailyGridFee(grecco_sim)
    elif name == "feeder_fee":
        return first_order.CoordinatorFeederDependentGridFee(grecco_sim)
    elif name == "central":
        # Central coordinator requires ems_configs for optimization setup.
        return central.CentralCoordinator(grecco_sim)
    else:
        msg = f"Unknown coordinator name {name}."
        raise ValueError(msg)
