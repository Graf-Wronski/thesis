import numpy as np
from grecco_sim.util import configs


def get_cummulative_ev_lims(
        horizon: int,
        now: int,
        request_list: list[configs.ChargingRequest]) -> (
        tuple)[np.ndarray, np.ndarray]:

    """ Calculate the cummulative upper limits for WLS given requests. """

    lower_limits = np.zeros(horizon)
    upper_limits = np.zeros(horizon)

    # Filter requests relevant to the regarded timeframe.
    requests = [r for r in request_list
                if r.start_step <= now + horizon
                and r.end_step >= now]

    max_kwh_by_t = 10.0

    for i in range(horizon):
        # Get the request that is active at that point in time.
        active_requests = [r for r in requests if r.active_at(i + now)]

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
            start_value = 0 if start == now else upper_limits[start - 1]
            before = upper_limits[i - 1]
            max_add = min(max_kwh_by_t, r.capacity + start_value - before)
            max_add = max(max_add, 0)
            upper_limits[i] = upper_limits[i - 1] + max_add

    return lower_limits, upper_limits

if __name__ == "__main__":
    horizon = 5
    req1 = configs.ChargingRequest(start_step=0, end_step=3, capacity=25)
    req2 = configs.ChargingRequest(start_step=5, end_step=100, capacity=500)
    req3 = configs.ChargingRequest(start_step=400, end_step=500, capacity=3)
    reqs = [req1, req2, req3]

    lims = get_cummulative_ev_lims(horizon=10, now=0, request_list=reqs)
    print(lims)