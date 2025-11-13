import pickle
from pathlib import Path
import yaml


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging

from grecco_sim.graph.utils.format import Format
from grecco_sim.util.analysis import OptimizationRun, evaluate_congestion

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
        p1 = self.data_dir / "uncoordinated" / f"min_cut_{self.seed}.csv"
        p2 = self.data_dir / "central" / "meta.yaml"
        return p1.exists() and p2.exists()

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


def add_costs(scalar_dict: dict):
    run_dir = scalar_dict["run_dir"]

    ts_df = pd.read_csv(
        run_dir / f"state_ts.csv",
        usecols=lambda x: x[-7:] == "p_model")

    ts_df.fillna(0)

    # Define the flex and inflex column identifiers
    flex_components = ['ev_p_model', 'bat_p_model', 'hp_p_model']
    inflex_components = ['pv_p_model', 'baseload_p_model']

    # Select columns containing the relevant component strings
    flex_cols = [col for col in ts_df.columns if
                 any(fc in col for fc in flex_components)]
    inflex_cols = [col for col in ts_df.columns if
                   any(ic in col for ic in inflex_components)]

    if scalar_dict["control"] not in ["central", "uncoordinated", "mixed"]:
        signals = pd.read_csv(run_dir / f"realized_signals.csv",
                              index_col=0)
        signals = signals.mean(axis=1)
        signal_costs = (signals * ts_df[flex_cols].sum(axis=1)).sum()
    else:
        signal_costs = 0.

    inflex_capacity_costs = ts_df[inflex_cols].values.sum(
        axis=None) * 0.66
    flex_capacity_costs = ts_df[flex_cols].clip(lower=0).values.sum(
        axis=None) * 0.66
    flex_capacity_costs += ts_df[flex_cols].clip(upper=0).values.sum(
        axis=None) * 0.33

    costs = inflex_capacity_costs + flex_capacity_costs + signal_costs

    # /4 because of 15-minute resolution.
    scalar_dict["Flexible Signal Costs"] = signal_costs / 4
    scalar_dict["Total Costs"] = costs / 4
    scalar_dict["Inflexible Capacity Costs"] = inflex_capacity_costs / 4
    scalar_dict["Flexible Capacity Costs"] = flex_capacity_costs / 4
    scalar_dict["Capacity Costs"] = (inflex_capacity_costs / 4 +
                                     flex_capacity_costs / 4)

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

        print(p)

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


def scalar_data(run_dir: Path, force_update: bool = False):

    buffer_dir = Format().data_root / "tmp"

    data_list = []
    congestion_by_feeder = []

    if not force_update:
        if (buffer_dir / "ex4_df_scalars.csv").exists():
            df_scalars = pd.read_csv(buffer_dir / "ex4_df_scalars.csv")
            return df_scalars

    paths = []
    for tag in ["central", "transformer", "feeder", "uncoordinated", "mixed"]:
        for p in run_dir.glob("run_*"):
            paths.append(p / tag)

    for p in paths:

        print(p)
        x = OptimizationRun(p)

        try:
            data = x.meta
        except FileNotFoundError:
            print(f"{p} is not valid. Skipping.")
            continue

        data["run_dir"] = p
        data["file_name"] = p.name
        data = evaluate_congestion(data=data, x=x)
        add_costs(data)
        data_list.append(data)

        cg_by_feeder = x.feeder_congestion.melt()
        cg_by_feeder["file_name"] = p.name
        congestion_by_feeder.append(cg_by_feeder)

    df_scalars = pd.DataFrame(data_list)
    df_feeder = pd.concat(congestion_by_feeder)

    df_scalars.to_csv(buffer_dir / "ex4_df_scalars.csv")

    return df_scalars


if __name__ == "__main__":

    run_dir = Path("/home/carl-wanninger/runs/experiment_5")
    plot_dir = Format().plot_root / "experiment_4"

    df_congestion, df_gear_cut = build_data(run_dir=run_dir,
                                            force_update=False)
    df = pd.concat([df_congestion, df_gear_cut])
    print(df.columns)

    data = df[df["feeder_trafo_ratio"] == 3.0]
    sns.catplot(data, x="kw_per_prosumer", y="value", kind="bar", estimator="sum")

    _, ax = plt.subplots()
    sns.barplot(df, x="variable", hue="control", estimator="sum", y="value", ax=ax)
    plt.show()

    """for df in [df_congestion, df_gear_cut]:
        df["feeder_trafo_ratio"] = df["feeder_trafo_ratio"].astype(float)
        df["kw_per_prosumer"] = df["kw_per_prosumer"].astype(float)"""

    df1 = scalar_data(run_dir=run_dir, force_update=False)

    for col in df1.columns:
        nice_col_name = col.title()
        nice_col_name = nice_col_name.replace("_", " ")
        df1[nice_col_name] = df1[col]

    data = []
    for _, x in df1.groupby(
            ["Seed", "Date", "Solver", "Topology", "feeder_trafo_ratio",
             "kw_per_prosumer"]):
        uncoordinated = x[x["control"] == "uncoordinated"]
        alpha = uncoordinated["Trafo Congestion Time (hrs)"].item()
        beta = uncoordinated["Total Trafo Congestion (kW)"].item()

        if alpha > 0:
            if beta == 0:
                raise ValueError
            x["Transformer Relief (Time)"] = (
                        1 - x["Trafo Congestion Time (hrs)"] / alpha)
            x["Transformer Relief (Load)"] = (
                        1 - x["Total Trafo Congestion (kW)"] / beta)

        else:
            x["Transformer Relief (Time)"] = (
                        1 - x["Trafo Congestion Time (hrs)"])
            x["Transformer Relief (Load)"] = (
                        1 - x["Total Trafo Congestion (kW)"])

        alpha = uncoordinated["Feeder Congestion Time (hrs)"].item()
        beta = uncoordinated["Total Feeder Congestion (kW)"].item()

        if alpha > 0:
            if beta == 0:
                raise ValueError
            x["Feeder Relief (Time)"] = (
                    1 - x["Feeder Congestion Time (hrs)"] / alpha)
            x["Feeder Relief (Load)"] = (
                    1 - x["Total Feeder Congestion (kW)"] / beta)

        else:
            x["Feeder Relief (Time)"] = (1 - x["Feeder Congestion Time (hrs)"])
            x["Feeder Relief (Load)"] = (1 - x["Total Feeder Congestion (kW)"])

        data.append(x)

    df_grouped = pd.concat(data)  # .query("control != 'uncoordinated'")

    control_order = ['central', 'gaussian', 'step', 'cubic',
                     'cubic_restricted']


    data = df_grouped.fillna(0)  # .query("feeder_trafo_ratio == 3.0")
    data["Topology"] = data["Topology"].str.replace("simbench-LV-", "")
    data["Topology"] = data["Topology"].str.replace("--2", "")

    num_feeders = {"rural1": 4, "rural2": 4, "semiurb4": 3}
    data["num_feeder"] = data["Topology"].apply(lambda x: num_feeders[x])

    # data = data.query("Topology == 'simbench-LV-rural2--2'")
    data["Trafo Capacity (kW) per Prosumer"] = data["kw_per_prosumer"]
    data["Feeder Capacity (kW) per Prosumer"] = data["kw_per_prosumer"] * data["feeder_trafo_ratio"]
    data["Feeder Capacity (kW) per Prosumer"] /= data["num_feeder"]
    data["fcpc"] = data["Feeder Capacity (kW) per Prosumer"]

    data = data.query("fcpc < 5")
    data = data.query("1 <= fcpc")

    # , "Feeder Relief (Load)",
    #               "Feeder Relief (Time)"

    for y in ["Feeder Congestion Time (hrs)"]:

        # data["topology"] = data["topology"] + "_" + data["solver"].apply(
        # lambda x: x[0])

        def rename_control(old_name: str):
            if old_name == "central_transformer":
                return "central"
            elif "_transformer" in old_name:
                return f"transformer: {old_name[:-len('_transformer')]}"
            elif "_feeder" in old_name:
                return f"feeder: {old_name[:-len('_feeder')]}"
            else:
                raise ValueError

        # data["control"] = data["control"].apply(lambda c: rename_control(c))
        _ = sns.relplot(data, x="Feeder Capacity (kW) per Prosumer", hue="control", y=y,
                        col="date", kind="line", col_wrap=2)

        for y in ["Trafo COngestime Time (hrs)"]:

            # data["topology"] = data["topology"] + "_" + data["solver"].apply(
            # lambda x: x[0])

            def rename_control(old_name: str):
                if old_name == "central_transformer":
                    return "central"
                elif "_transformer" in old_name:
                    return f"transformer: {old_name[:-len('_transformer')]}"
                elif "_feeder" in old_name:
                    return f"feeder: {old_name[:-len('_feeder')]}"
                else:
                    raise ValueError


            # data["control"] = data["control"].apply(lambda c: rename_control(c))
            _ = sns.relplot(data, x="kw_per_prosumer", hue="control", y=y,
                            col="Date", kind="line")


    plt.show()