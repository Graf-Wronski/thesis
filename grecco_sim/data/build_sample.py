import warnings

import numpy as np
import pandas as pd
import pypsa

from itertools import product

from grecco_sim.data.data_configuration import DataConfiguration
from grecco_sim.data.time_series_sampler import TimeSeriesSampler

from grecco_sim.graph.utils.format import Format
from pathlib import Path


def simbench_to_pypsa(p: Path):
    # Create network
    n = pypsa.Network()

    # --- Load CSVs ---
    node_df = pd.read_csv(p / "Node.csv", sep=";")
    line_df = pd.read_csv(p / "Line.csv", sep=";")
    trafo_df = pd.read_csv(p / "Transformer.csv", sep=";")
    ext_df = pd.read_csv(p / "ExternalNet.csv", sep=";")
    load_df = pd.read_csv(p / "Load.csv", sep=";")

    # --- 1. BUSES ---
    for _, row in node_df.iterrows():
        name = row["id"]
        v_nom = row["vmR"] if pd.notna(
            row["vmR"]) else 0.4  # default if missing
        n.add("Bus", name=name, v_nom=v_nom)

    # --- 1. BUSES ---
    for _, row in load_df.iterrows():
        name = row["id"]
        bus = row["node"]
        n.add("Load", name=name, bus=bus, carrier="inflex" ,p_set=0.)

    # --- 2. LINES ---
    # Assume basic line impedance per km (mocked). Real values should be read from a type library.
    # For NAYY 4x150SE: approx r=0.206, x=0.08 Ohm/km (LV cable)

    for _, row in line_df.iterrows():
        n.add("Line",
              name=row["id"],
              bus0=row["nodeA"],
              bus1=row["nodeB"],
              type="NAYY 4x150 SE",
              length=np.random.choice([0.1 * x for x in range(1, 8)]))
        n.lines["loadingMax"] = row["loadingMax"]

    # --- 3. TRANSFORMERS ---
    # Mock transformer type: 0.16 MVA, 20/0.4 kV, r = 0.01 pu, x = 0.04 pu
    for _, row in trafo_df.iterrows():
        n.add("Transformer",
              name=row["id"],
              bus0=row["nodeHV"],
              bus1=row["nodeLV"],
              s_nom=row["loadingMax"],
              type="0.4 MVA 20/0.4 kV")
        n.transformers["loadingMax"] = row["loadingMax"]

    # --- 4. EXTERNAL GRID / SLACK BUS ---
    for _, row in ext_df.iterrows():
        node = row["node"]
        n.add("Generator",
              name=row["id"],
              bus=node,
              control="Slack")

    return n

def build_sample(config: DataConfiguration) -> pypsa.Network:

    if "simbench-" in str(config.topology.resolve()):
        n = simbench_to_pypsa(config.topology)
    else:
        with warnings.catch_warnings():
            n = pypsa.Network()
            warnings.simplefilter("ignore", category=UserWarning)
            n.import_from_csv_folder(config.topology)

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

def main(topology_name: str):
    p_topology = Format().data_root / "topologies" / topology_name

    seeds = [3, 5, 17, 257, 65537]
    opfingen = ("Opfingen", [(8, 11), (10, 11), (8, 30), (2, 27)])
    simbench = ("SimBench", [(5, 27), (12, 25), (5, 11), (3, 8)])
    shed_50 = {"pv_quota": 0.7, "bss_quota": 0.68, "name": "shed_2050",
               "hp_quota": 0.68, "ev_quota": 0.78}
    simbench_34 = {"pv_quota": 0.7, "bss_quota": 0.33, "name": "simbench_2034",
                   "hp_quota": 0.23, "ev_quota": 0.19}

    iterator = product(seeds, [shed_50, simbench_34], [opfingen, simbench])

    for seed, load_distribution, load_data in iterator:
        db_name, dates = load_data

        for month, day in dates:
            sample_name = (f"{p_topology.name}_{db_name}_{month}_{day}_"
                           f"{load_distribution['name']}_{seed}")

            data_config = DataConfiguration(
                day=day,
                month=month,
                topology=p_topology,
                ts_data_base=db_name,
                sample_name=sample_name,
                pv_quota=load_distribution["pv_quota"],
                bss_quota=load_distribution["bss_quota"],
                hp_quota=load_distribution["hp_quota"],
                ev_quota=load_distribution["ev_quota"],
                seed=seed)

            sample = build_sample(data_config)
            sample.name = sample_name
            sample.lpf()

            p = Format().data_root / "samples" / db_name / load_distribution[
                'name']
            if not (p.exists()):
                p.mkdir(parents=True)
            sample.export_to_csv_folder(p / f"{sample_name}")


if __name__ == "__main__":
    # Für jede Topologie hätte ich am liebsten gleich Daten mit mehreren
    # Seeds, mit allen relevanten Tagen, mit SimBench und Opfingen Loads.
    # Mit SimBench 2034 und SHED 2050 Lastverteilungen.

    # Offene Fragen: Funktioniert SimBench Data wie gedacht?
    # Welche Daten sollte ich wählen? (15. jedes Monats)
    topologies = ["simbench-LV-semiurb4--2",
                  "simbench-LV-semiurb5--2",
                  "simbench-LV-urban6--2",
                  "simbench-LV-rural1--2",
                  "simbench-LV-rural2--2",
                  "simbench-LV-rural3--2"]
    for topology in topologies:
        main(topology)
