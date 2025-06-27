import warnings

import pandas as pd
import pypsa

from thesis.data.data_configuration import DataConfiguration
from thesis.data.time_series_sampler import TimeSeriesSampler

from pathlib import Path

def build_sample(config: DataConfiguration) -> pypsa.Network:

    # Load topology. Loads are not used, but we need to know where loads are.
    n = pypsa.Network()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        n.import_from_csv_folder(config.topology, skip_time=True)

    n.set_snapshots(snapshots=[x for x in config.date_range])

    buses = [x for x in n.loads["bus"].unique()]

    n.loads = pd.DataFrame()
    n.generators = n.generators[n.generators["control"] == "Slack"]
    n.storage_units = pd.DataFrame()

    sampler = TimeSeriesSampler(config, buses)

    # Sample Units.
    baseload, baseload_p_ts = sampler.sample_baseload()
    pv, pv_p_ts = sampler.sample_pv()
    bss = sampler.sample_bss()
    hp = sampler.sample_hp()

    # Remove generators, storages and loads.

    for _, row in baseload.iterrows():
        bus = row["bus"]
        p_set = baseload_p_ts[bus]
        load_data = {"name": f"baseload at {bus}", "bus": bus, "p_set": p_set}
        # Carrier information is expected by GrECCo.
        n.add(class_name="Load", carrier="baseload", **load_data)

    for _, row in hp.iterrows():
        bus, p_set = row["bus"], row["p_set"]
        load_data = {"name": f"heat_pump at {bus}", "bus": bus, "p_set": p_set}
        # Carrier information is expected by GrECCo.
        n.add(class_name="Load", carrier="heat_pump", **load_data)

    for _, row in pv.iterrows():
        bus = row["bus"]
        p_set = pv_p_ts[bus]
        load_data = {"name": f"pv at {bus}", "bus": bus, "p_set": p_set}
        # Carrier information is expected by GrECCo.
        n.add(class_name="Generator", carrier="solar", **load_data)

    for _, row in bss.iterrows():
        bus, p_nom = row["bus"], row["p_nom"]
        storage = {"name": f"Storage at {bus}", "bus": bus, "p_nom": p_nom}
        n.add(class_name="StorageUnit", type="h0_battery", **storage)

    # irradiance = sampler.irrandiance
    # temperature = sampler.temperature

    return n

if __name__ == "__main__":
    p_topology = (Path("/home/carl-wanninger/data/topologies")
                  / "opfingen")

    data_config = DataConfiguration(
        topology= p_topology,
        ts_data_base="Opfingen",
        pv_quota=0.5,
        bss_quota=0.5,
        hp_quota=0.5,
        seed=17)
    n = build_sample(data_config)
    n.name = "Sample Network 0"
    n.lpf()

    print(n.loads)