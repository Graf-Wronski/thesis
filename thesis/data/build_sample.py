import warnings

import pandas as pd
import pypsa

from thesis.data.data_configuration import DataConfiguration
from thesis.data.time_series_sampler import TimeSeriesSampler

from pathlib import Path

def build_sample(config: DataConfiguration) -> pypsa.Network:

    # Load topology. Loads are not used, but we need to know where loads are.
    n = pypsa.Network()
    n.name = config.sample_name

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        n.import_from_csv_folder(config.topology, skip_time=True)

    # noinspection PyTypeChecker
    n.set_snapshots([x for x in config.date_range])

    buses = [x for x in n.loads["bus"].unique()]

    # Remove previous generators (other than slack), storages and loads.
    n.loads = pd.DataFrame()
    n.generators = n.generators[n.generators["control"] == "Slack"]
    n.storage_units = pd.DataFrame()

    sampler = TimeSeriesSampler(config, buses)

    # Sample Units.
    baseload, baseload_p_ts = sampler.sample_baseload()
    pv, pv_p_ts = sampler.sample_pv()
    bss = sampler.sample_bss()
    hp = sampler.sample_hp()
    ev = sampler.sample_ev()

    for _, row in baseload.iterrows():
        bus = row["bus"]
        p_set = baseload_p_ts[bus].values # Data as series is missread.
        load_data = {"name": f"baseload at {bus}", "bus": bus, "p_set": p_set}
        # Carrier information is expected by GrECCo.
        n.add(class_name="Load", carrier="baseload", **load_data)

    for _, row in hp.iterrows():
        bus, p_set = row["bus"], row["p_set"]
        load_data = {"name": f"heat_pump at {bus}", "bus": bus, "p_set": p_set}
        # Carrier information is expected by GrECCo.
        n.add(class_name="Load", carrier="heat_pump", **load_data)

    for attr, val in n.loads_t.items():
        n.loads_t[attr] = val.copy()

    for _, row in pv.iterrows():
        bus = row["bus"]
        p_set = pv_p_ts[bus].values
        gen_data = {"name": f"pv at {bus}", "bus": bus, "p_set": p_set}
        # Carrier information is expected by GrECCo.
        n.add(class_name="Generator", carrier="solar", **gen_data)

    for attr, val in n.generators_t.items():
        n.generators_t[attr] = val.copy()

    for _, row in bss.iterrows():
        bus, p_nom = row["bus"], row["p_nom"]
        storage = {"name": f"Storage at {bus}", "bus": bus, "p_nom": p_nom}
        n.add(class_name="StorageUnit", type="h0_battery", **storage)

    for _, row in ev.iterrows():
        bus, p_nom, charger_profile = row["bus"], row["p_nom"], row["type"]
        charger = {"name": f"EV at {bus}", "bus": bus,
                   "p_nom": p_nom, "type": charger_profile}
        # ToDo:Non bi-directional chargers should be modelled as loads.
        n.add(class_name="StorageUnit", **charger)

    return n

if __name__ == "__main__":
    p_topology = (Path("/home/carl-wanninger/data/topologies")
                  / "opfingen")

    data_config = DataConfiguration(
        day=13,
        month=1,
        topology= p_topology,
        ts_data_base="Opfingen",
        pv_quota=0.9,
        bss_quota=0.9,
        hp_quota=0.9,
        ev_quota=0.9,
        seed=17)
    network = build_sample(data_config)
    network.name = "Sample Network Units 90 %"
    network.lpf()

    p = Path(f"/home/carl-wanninger/data/samples/{data_config.ts_data_base}")
    idx = 0

    # Check for existing versions.
    while (p / f"{data_config.sample_name}_{idx}").exists():
        idx += 1

    network.export_to_csv_folder((p / f"{data_config.sample_name}_{idx}"))


    print(network.loads)