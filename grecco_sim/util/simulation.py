from grecco_sim.util import config


def build_system_config(
        sys_id: str,
        horizon: int,
        dt_h: float,
        unit_parameters: dict):

    pv, battery, heatpump, ev = None, None, None, None

    for key in unit_parameters.keys():
        if "pv" in key:
            pv = unit_parameters[key]
        if "bat" in key:
            battery = unit_parameters[key]
        if "hp" in key:
            heatpump = unit_parameters[key]
        if "ev" in key:
            ev = unit_parameters[key]

    return config.EMSConfiguration(
        sys_id = sys_id,
        horizon=horizon,
        dt_h=dt_h,
        pv=pv,
        battery=battery,
        heatpump=heatpump,
        ev=ev)