from typing import Union, Any, Optional, Callable

import numpy as np
import pandas as pd

from grecco_sim.coordinator import first_order, central
from grecco_sim.util import configs

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
        solver_options["gurobi"] = {"TimeLimit": 5,
                                    "OutputFlag": 0,
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

    if name == "uncoordinated":
        g = None
        return first_order.Uncoordinated(grecco_sim, g)
    elif name == "transformer_fee":
        g = temporal_resolution(grecco_sim.config.temporal_resolution)
        return first_order.CoordinatorDailyGridFee(grecco_sim, g)
    elif name == "feeder_fee":
        g = temporal_resolution(grecco_sim.config.temporal_resolution)
        return first_order.CoordinatorFeederDependentGridFee(grecco_sim, g)
    elif name == "central":
        # Central coordinator requires ems_configs for optimization setup.
        return central.CentralCoordinator(grecco_sim)
    else:
        msg = f"Unknown coordinator name {name}."
        raise ValueError(msg)


def cummulative_ev_lims(
        now: int,
        horizon: int,
        config: configs.ChargerAndEVConfig,
        request_list: list[configs.ChargingRequest]) -> (
        tuple)[np.ndarray, np.ndarray]:
    """ Calculate the cummulative upper limits for WLS given requests. """

    lower_limits = np.zeros(horizon)
    upper_limits = np.zeros(horizon)

    # Filter requests relevant to the regarded timeframe.
    requests = [r for r in request_list
                if r.start_step <= now + horizon
                and r.end_step >= now]

    max_kwh_by_t = config.p_lim_effective * config.dt_h

    for i in range(horizon):
        # Get the request that is active at that point in time.
        active_requests = [r for r in requests
                           if r.active_at(i + now)]

        if len(active_requests) > 1:
            msg = "Charger can handle only one request per time step."
            raise NotImplementedError(msg)

        elif len(active_requests) == 0 or active_requests[0].capacity == 0:
            if i == 0:
                # Keep limits at 0.
                continue
            else:
                lower_limits[i] = lower_limits[i - 1]
                upper_limits[i] = upper_limits[i - 1]

        else:
            # We have an active request with positive capacity.
            r = active_requests[0]

            # If a request has capacity x and in the future y can be
            # delivered, then this time step has to deliver at least x - y.
            steps_left = r.end_step - (now + i)
            debt = r.capacity - steps_left * max_kwh_by_t
            lower_limits[i] = lower_limits[i - 1] + max(debt, 0)

            start = max(r.start_step, now)
            start_value = 0 if start == now else upper_limits[start - now - 1]
            before = upper_limits[i - 1]
            max_add = min(max_kwh_by_t, r.capacity + start_value - before)
            max_add = max(max_add, 0)
            upper_limits[i] = upper_limits[i - 1] + max_add

    return lower_limits, upper_limits


def temporal_resolution(name: str) -> Callable[[np.ndarray], np.ndarray]:
    # Corrected by ChatGPT

    if name == "cubic":
        return lambda x: np.clip(0.33 * ((10 / 9) * x) ** 3, a_min=-3, a_max=3)

    elif name == "cubic_restricted":
        return lambda x: np.clip(0.33 * ((10 / 9) * x) ** 3, a_min=-1, a_max=3)

    elif name == "step":
        def step_function(x: np.ndarray) -> np.ndarray:
            result = np.zeros_like(x)
            result[x >= 1.88] = 3.0
            result[(x >= 1.49) & (x < 1.88)] = 1.66
            result[(x >= 0.9) & (x < 1.49)] = 0.33
            result[(x <= -0.9) & (x > -1.49)] = -0.33
            result[x <= -1.49] = -1.0
            return result

        return step_function

    elif name == "gaussian":
        return lambda x : np.random.normal(loc=0., scale=0.33, size=x.shape)

    else:
        raise NotImplementedError(f"Unknown temporal resolution {name}.")
