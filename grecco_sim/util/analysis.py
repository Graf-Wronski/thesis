from pathlib import Path

import numpy as np
import pypsa
import yaml
import pandas as pd

from grecco_sim.util.network_io import determine_feeders


def get_congestion(data_dir: Path) -> pd.DataFrame:
    with open(data_dir / 'meta.yaml', 'r') as f:
        meta = yaml.load(f, Loader=yaml.SafeLoader)

    # Load buses, lines, transformer and snapshots.
    buses = pd.read_csv(data_dir / "network" / "buses.csv")
    lines = pd.read_csv(data_dir / "network" / "lines.csv")
    lines.index = lines["name"]
    loads = pd.read_csv(data_dir / "network" / "loads.csv")
    loads = loads[loads["name"].apply(lambda x: x.startswith("baseload"))]
    loads["name"] = loads["name"].str.replace("baseload at ", "")
    loads["name"] = "sys_at_bus_" + loads["name"] + "_household_p_model"
    transformers = pd.read_csv(data_dir / "network" / "transformers.csv")
    transformers.index = transformers["name"]
    snapshots = pd.read_csv(data_dir / "network" / "snapshots.csv")["snapshot"]

    n = pypsa.Network()

    n.set_snapshots(snapshots)

    bus_data = buses.to_dict(orient='list')
    n.madd("Bus", names=bus_data["name"], **bus_data)
    line_data = lines.to_dict(orient='list')
    n.madd("Line", names=line_data["name"], **line_data)
    load_data = loads.to_dict(orient='list')
    n.madd("Load", names=load_data["name"], **load_data)
    trafo_data = transformers.to_dict(orient='list')
    n.madd("Transformer", names=trafo_data["name"], **trafo_data)

    # Apply loads as given by p_model.
    col_filter = lambda x: x.endswith("household_p_model")
    state_ts = pd.read_csv(data_dir / "state_ts.csv", usecols=col_filter)
    state_ts.set_index(snapshots, inplace=True)
    n.loads_t["p_set"] = state_ts / 1000  # mWh for PyPSA.
    n.lpf()

    # Determine transformer congestion.
    trafo_loading = np.maximum(n.transformers_t["p0"], n.transformers_t["p1"])
    transformer_congestion = trafo_loading / transformers["capacity"]
    trafo_loading.to_csv(data_dir / "transformers-loading.csv")
    transformer_congestion.to_csv(data_dir / "transformers-congestion.csv")

    # Determine line segment congestion.
    line_loading = np.maximum(n.lines_t["p0"], n.lines_t["p1"])
    line_congestion = line_loading / lines["capacity"]
    line_loading.to_csv(data_dir / "lines-loading.csv")
    line_congestion.to_csv(data_dir / "lines-congestion.csv")

    # Determine feeder congestion
    feeder_map = determine_feeders(n)
    with open(data_dir / 'feeder_map.yaml', 'w') as f:
        yaml.dump(feeder_map, f)

    feeder_loading = line_loading.copy()
    feeder_loading = feeder_loading.rename(columns=feeder_map)
    feeder_loading = feeder_loading.T.groupby(level=0).max().T

    feeder_congestion = line_congestion.copy()
    feeder_congestion = feeder_congestion.rename(columns=feeder_map)
    feeder_congestion = feeder_congestion.T.groupby(level=0).max().T

    feeder_loading.to_csv(data_dir / "feeder_loading.csv")
    feeder_congestion.to_csv(data_dir / "feeder_congestion.csv")


class OptimizationRun:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @property
    def meta(self) -> dict:
        with open(self.data_dir / 'meta.yaml', 'r') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    @property
    def line_loading(self) -> pd.DataFrame:
        if not(self.data_dir / 'lines-loading.csv').exists():
            get_congestion(self.data_dir)
        return pd.read_csv(self.data_dir / 'lines-loading.csv', index_col=0)

    @property
    def transformer_loading(self) -> pd.DataFrame:
        if not (self.data_dir / 'transformers-loading.csv').exists():
            get_congestion(self.data_dir)
        return pd.read_csv(self.data_dir / 'transformers-loading.csv', index_col=0)

    @property
    def line_congestion(self) -> pd.DataFrame:
        if not (self.data_dir / 'lines-congestion.csv').exists():
            get_congestion(self.data_dir)
        return pd.read_csv(self.data_dir / 'lines-congestion.csv', index_col=0)

    @property
    def trafo_congestion(self) -> pd.DataFrame:
        if not (self.data_dir / 'transformers-congestion.csv').exists():
            get_congestion(self.data_dir)
        return pd.read_csv(self.data_dir / 'transformers-congestion.csv', index_col=0)

    @property
    def feeder_loading(self) -> pd.DataFrame:
        if not (self.data_dir / 'feeder_loading.csv').exists():
            get_congestion(self.data_dir)
        return pd.read_csv(self.data_dir / 'feeder_loading.csv', index_col=0)

    @property
    def feeder_congestion(self) -> pd.DataFrame:
        if not (self.data_dir / 'feeder_congestion.csv').exists():
            get_congestion(self.data_dir)
        return pd.read_csv(self.data_dir / 'feeder_congestion.csv', index_col=0)

    @property
    def feeder_map(self) -> dict:
        with open(self.data_dir / 'feeder_map.yaml', 'r') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)


def evaluate_congestion(data: dict, x: OptimizationRun) -> dict:

    n_events = (x.feeder_congestion >= 1).values.sum()
    data["Feeder Congestion Events (Count)"] = n_events
    hours = data["Feeder Congestion Events (Count)"] / 4
    data["Feeder Congestion Time (hrs)"] = hours
    y = (x.feeder_congestion - 1.0).clip(0).values.sum()
    data["Total Feeder Congestion (kW)"] = y
    data["Total Feeder Load (kW)"] = x.feeder_loading.values.sum()

    n_events = (x.trafo_congestion >=1).values.sum()
    data["Trafo Congestion Events (Count)"] = n_events
    hours = data["Trafo Congestion Events (Count)"] / 4
    data["Trafo Congestion Time (hrs)"] = hours
    y = (x.trafo_congestion - 1.0).clip(0).values.sum()
    data["Total Trafo Congestion (kW)"] = y
    data["Total Trafo Load (kW)"] = x.transformer_loading.values.sum()

    return data