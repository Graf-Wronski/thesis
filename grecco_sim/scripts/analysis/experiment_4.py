import pickle
from pathlib import Path
import yaml


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging

from grecco_sim.util.analysis import OptimizationRun

logging.getLogger("pypsa").setLevel(logging.WARNING)

def get_meta(p: Path) -> dict:
    with open(p / 'meta.yaml', 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

class DataPoint:
    def __init__(self, data_dir, seed: int = 17):
        self.data_dir = data_dir
        self.seed = seed
        
    @property
    def is_valid(self) -> bool:
        p = self.data_dir / "central" / "meta.yaml"
        return p.exists()

    @property
    def is_semi_valid(self) -> bool:
        p_1 = self.data_dir / "central"
        p_2 = self.data_dir / "central" / f"bottleneck_{self.seed}.pkl"
        return (p_1.exists() and not p_2.exists())

    @property
    def central(self) -> OptimizationRun:
        return OptimizationRun(self.data_dir / "central")

    @property
    def uncoordinated(self) -> OptimizationRun:
        return OptimizationRun(self.data_dir / "uncoordinated")

    @property
    def transformer_opt(self) -> OptimizationRun:
        return OptimizationRun(self.data_dir / "transformer")

    @property
    def feeder_opt(self) -> OptimizationRun:
        return OptimizationRun(self.data_dir / "feeder")

    @property
    def bottleneck(self) -> list:
        p = self.data_dir / "uncoordinated" / f"bottleneck_{self.seed}.pkl"
        with open(p, "rb") as f:
            return pickle.load(f)

    @property
    def min_cut(self) -> pd.DataFrame:
        p = self.data_dir / "uncoordinated" / f"min_cut_{self.seed}.csv"
        df = pd.read_csv(p,   parse_dates=['snapshot'], index_col='snapshot')
        return df


def build_data(run_dir: Path, force_update: bool = False):
    gear_cuts = []
    congestion_maps = []

    if not force_update:
        if (run_dir / "congestion_table.csv").exists():
            if (run_dir / "gear_cut.csv").exists():
                congestion_map = pd.read_csv(run_dir / "congestion_table.csv")
                gear_cut = pd.read_csv(run_dir / "gear_cut.csv")
                return congestion_map, gear_cut

    for p in run_dir.glob("run_*"):
        x = DataPoint(p)

        if not "rural1" in str(p):
            # ToDo: Further topology?
            continue

        if not x.is_valid:
            print(f"{p} is not valid.")
            continue

        feeder_map = x.uncoordinated.feeder_map

        # Aggregate cuts for feeder.
        gear_cut = x.min_cut.rename(columns=feeder_map)
        # Feeder dict returns int.
        gear_cut.columns = gear_cut.columns.astype(str)
        gear_cut = gear_cut.T.groupby(level=0).max().T
        gear_cut_melted = pd.melt(gear_cut, ignore_index=False)
        gear_cut_melted = gear_cut_melted.groupby("variable").sum()

        meta = get_meta(p / "uncoordinated")
        meta["control"] = "push_relabel"
        for key, val in meta.items():
            gear_cut_melted[key] = val

        gear_cuts.append(gear_cut_melted)

        for data in [x.central, x.uncoordinated, x.feeder_opt,
                     x.transformer_opt]:

            congestion_map = data.feeder_congestion

            meta = data.meta
            congestion_map_melted = pd.melt(congestion_map, ignore_index=False)
            congestion_map_melted["value"] = (congestion_map_melted["value"]
                                              >= 1).astype(float)
            congestion_map_melted = congestion_map_melted.groupby("variable").sum()
            for key, val in meta.items():
                congestion_map_melted[key] = val

            congestion_maps.append(congestion_map_melted)

    df1, df2 =  pd.concat(congestion_maps), pd.concat(gear_cuts)


    df1.to_csv(run_dir / "congestion_table.csv")
    df2.to_csv(run_dir / "gear_cut.csv")

    return df1, df2

if __name__ == "__main__":


    run_dir_1 = Path("/home/carl-wanninger/runs/experiment_4")

    df_congestion, df_gear_cut = build_data(run_dir=run_dir_1,
                                            force_update=False)
    df = pd.concat([df_congestion, df_gear_cut])

    """_, ax = plt.subplots()
    data = df[df["feeder_trafo_ratio"] == 3.0]
    sns.catplot(data, ax=ax, x="kw_per_prosumer", y="value", kind="bar", estimator="sum")

    _, ax = plt.subplots()
    sns.barplot(df, x="variable", hue="control", estimator="sum", y="value", ax=ax)
    plt.show()"""

    """for df in [df_congestion, df_gear_cut]:
        df["feeder_trafo_ratio"] = df["feeder_trafo_ratio"].astype(float)
        df["kw_per_prosumer"] = df["kw_per_prosumer"].astype(float)"""

    data = df.copy()
    _, ax = plt.subplots()
    sns.barplot(data, x="feeder_trafo_ratio", hue="kw_per_prosumer",
                estimator="mean", y="value", ax=ax)

    data = df[df["kw_per_prosumer"] == 2.0]
    _, ax = plt.subplots()
    sns.lineplot(data, x="feeder_trafo_ratio", hue="control",
                estimator="mean", y="value", ax=ax)

    data = df.copy()
    _, ax = plt.subplots()
    sns.barplot(data, x="variable", hue="feeder_trafo_ratio", estimator="sum", y="value", ax=ax)

    _, ax = plt.subplots()
    sns.barplot(data, x="variable", hue="kw_per_prosumer",
                estimator="sum", y="value", ax=ax)




    plt.show()
