from pathlib import Path
from typing import Optional

import pandas as pd
import pypsa
from pypsa import Network


def get_four_node_lv_grid(
        load_data: Path = Path("../test_data"),
        export_dir: Optional[Path] = None) -> pypsa.Network:

    network: Network = pypsa.Network()

    # Load profiles.
    gen_profile = pd.read_csv(load_data / "RESProfile.csv", sep=";")
    load_profile = pd.read_csv(load_data / "LoadProfile.csv", sep=";", on_bad_lines='warn')
    storage_profile = pd.read_csv(load_data / "StorageProfile.csv", sep=";")
    load = pd.read_csv(load_data / "Load.csv", sep=";")
    gen = pd.read_csv(load_data / "RES.csv", sep=";")
    storage = pd.read_csv(load_data / "Storage.csv", sep=";")

    # Add snapshots.
    network.set_snapshots(gen_profile["time"])
    gen_profile.set_index('time', drop=True, inplace=True)
    load_profile.set_index('time', drop=True, inplace=True)
    storage_profile.set_index('time', drop=True, inplace=True)

    # Create a 4 bus radial network with two feeders.
    # Add buses.
    network.add(
        class_name="Bus",
        name="ExternalGrid",
        v_nom=10.0)

    # Add four buses with nominal voltage 0.4.
    for idx in range(5):
        network.add(
            class_name="Bus",
            name=f"Bus{idx}",
            v_nom=0.4)

    # 0.4 MVA transformer between lv radial network and external mv grid.
    network.add(
        class_name="Transformer",
        name="ExternalGrid",
        bus0="ExternalGrid",
        bus1="Bus0",
        type="0.4 MVA 10/0.4 kV")

    # Add lines of type NAYY 4x150 SE.
    edges = [(0, 1), (1, 2), (0, 3), (3, 4)]

    for idx, edge in enumerate(edges):

        network.add(
            class_name="Line",
            name=f"Line{idx}",
            bus0=f"Bus{edge[0]}",
            bus1=f"Bus{edge[1]}",
            type="NAYY 4x150 SE",
            length=0.1)

    # Add slack generator.
    network.add(
        class_name="Generator",
        name="ExternalGrid",
        bus="ExternalGrid",
        p_nom=10,
        p_set=0.1,
        control="Slack")

    # Add household loads.
    network.add(
        class_name="Load",
        name="HH1",
        bus="Bus1",
        p_set=load.iloc[10]['pLoad'] * load_profile["H0-G_pload"],
        q_set=load.iloc[10]['qLoad'] * load_profile["H0-G_qload"])

    network.add(
        class_name="Load",
        name="HH2",
        bus="Bus2",
        p_set=load.iloc[9]['pLoad'] * load_profile["H0-C_pload"],
        q_set=load.iloc[9]['qLoad'] * load_profile["H0-C_qload"])

    network.add(
        class_name="Load",
        name="HH3",
        bus="Bus3",
        p_set=load.iloc[13]['pLoad'] * load_profile["H0-A_qload"],
        q_set=load.iloc[13]['qLoad'] * load_profile["H0-A_pload"])

    network.add(
        class_name="Load",
        name="HH4",
        bus="Bus4",
        p_set=load.iloc[35]['pLoad'] * load_profile["H0-B_pload"],
        q_set=load.iloc[35]['qLoad'] * load_profile["H0-B_qload"])

    # Load profile includes Air_Semi-Parallel_1 Heat Pump Profile
    network.add(
        class_name="Load",
        name="hp3",
        bus="Bus3",
        p_set=load.iloc[108]['pLoad'] * load_profile["Air_Semi-Parallel_1_pload"],
        q_set=load.iloc[108]['qLoad'] * load_profile["Air_Semi-Parallel_1_qload"])

    # Load profile includes Air_Parallel_1 Heat Pump Profile
    network.add(
        class_name="Load",
        name="hp4",
        bus="Bus4",
        p_set=load.iloc[108]['pLoad'] * load_profile["Air_Parallel_1_pload"],
        q_set=load.iloc[108]['qLoad'] * load_profile["Air_Parallel_1_qload"])

    # Load profile includes EV Charging Profile
    network.add(
        class_name="Load",
        name="ev4",
        bus="Bus4",
        p_set=load.iloc[99]['pLoad'] * load_profile["HLS_C_3.7_pload"],
        q_set=load.iloc[99]['qLoad'] * load_profile["HLS_C_3.7_qload"])

    # Add generators - Generation limits (p_nom) and time series (p_set) in MW.
    network.add(
        class_name="Generator",
        name="PV2",
        bus="Bus2",
        control="PQ",
        p_nom=gen.iloc[0]["pRES"],
        p_set=gen.iloc[0]["pRES"] * gen_profile[gen.iloc[0]["profile"]],
        carrier="PV")

    network.add(
        class_name="Generator",
        name="PV3",
        bus="Bus3",
        control="PQ",
        p_nom=gen.iloc[1]["pRES"],
        p_set=gen.iloc[1]["pRES"] * gen_profile[gen.iloc[1]["profile"]],
        carrier="PV")

    network.add(
        class_name="Generator",
        name="PV4",
        bus="Bus4",
        control="PQ",
        p_nom=gen.iloc[7]["pRES"],
        p_set=gen.iloc[7]["pRES"] * gen_profile[gen.iloc[7]["profile"]],
        carrier="PV")

    # Add storages - Storage limits (p_nom) and time series (p_set) in MW.
    for idx in [0, 5, 6]:
        print(storage.iloc[idx])

    network.add(
        class_name="StorageUnit",
        name="Storage2",
        bus="Bus2",
        p_nom=storage.iloc[5]["sR"],
        p_set=storage.iloc[5]['sR'] * storage_profile["Storage_PV7_H0-C"])

    network.add(
        class_name="StorageUnit",
        name="Storage3",
        bus="Bus3",
        p_nom=storage.iloc[0]['sR'],
        p_set=storage.iloc[0]['sR'] * storage_profile["Storage_PV3_H0-A"])

    network.add(
        class_name="StorageUnit",
        name="Storage4",
        bus="Bus4",
        p_nom=storage.iloc[6]['sR'],
        p_set=storage.iloc[6]['sR'] * storage_profile["Storage_PV1_H0-B"])

    network.lpf(network.snapshots[0:100])
    network.pf(network.snapshots[0:100], use_seed=True)

    if export_dir:
        network.export_to_csv_folder(export_dir)

    return network