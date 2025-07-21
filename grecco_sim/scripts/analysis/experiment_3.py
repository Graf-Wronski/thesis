from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging

from grecco_sim.graph.utils.format import Format
from grecco_sim.util.analysis import OptimizationRun, evaluate_congestion

logging.getLogger("pypsa").setLevel(logging.WARNING)

DATA_PATH = Format().data_root


def main():
    pass


def build_data(
        run_dir_trafo: Path,
        run_dir_feeder: Path,
        force_update: bool = False):

    data_list = []
    congestion_by_feeder = []

    if not force_update:
        if (DATA_PATH / "tmp" / "ex3_df1.csv").exists():
            if (DATA_PATH / "tmp" / "ex3_df2.csv").exists():
                df_scalars = pd.read_csv(DATA_PATH / "tmp" / "ex3_df1.csv")
                df_feeder = pd.read_csv(DATA_PATH / "tmp" / "ex3_df2.csv")

                return df_scalars, df_feeder

    for p in run_dir_trafo.glob("run_*"):
        print(p)
        x = OptimizationRun(p)
        data = x.meta
        data["control"] += "_transformer"
        data["file_name"] = p.name
        data = evaluate_congestion(data=data, x=x)
        data_list.append(data)

        cg_by_feeder = x.feeder_congestion.melt()
        cg_by_feeder["file_name"] = p.name
        congestion_by_feeder.append(cg_by_feeder)

    for p in run_dir_feeder.glob("run_*"):
        print(p)
        x = OptimizationRun(p)
        data = x.meta
        data["control"] += "_feeder"
        data["file_name"] = p.name
        data = evaluate_congestion(data=data, x=x)
        data_list.append(data)

        cg_by_feeder = x.feeder_congestion.melt()
        cg_by_feeder["file_name"] = p.name
        congestion_by_feeder.append(cg_by_feeder)

    df_scalars = pd.DataFrame(data_list)
    df_feeder = pd.concat(congestion_by_feeder)

    df_scalars.to_csv(DATA_PATH / "tmp" / "ex3_df1.csv")
    df_feeder.to_csv(DATA_PATH / "tmp" / "ex3_df2.csv")

    return df_scalars, df_feeder


def plot_transformer_congestion():
    pass


def plot_feeder_congestion():
    pass


if __name__ == "__main__":
    feeder_congestion_count = pd.DataFrame()

    run_dir_1 = Format().output_root / "experiment_3"
    run_dir_2 = Format().output_root / "experiment_2"
    plot_dir = Format().plot_root / "experiment_2b"

    df1, df2 = build_data(run_dir_feeder=run_dir_1, run_dir_trafo=run_dir_2,
                          force_update=False)

    df1["topology"] = df1["file_name"].apply(lambda x: x.split("_")[3])
    df1["solver"] = df1["file_name"].apply(lambda x: x.split("_")[2])
    df1["topology"] = df1["topology"].str.replace("simbench-LV-", "")
    df1["topology"] = df1["topology"].str.replace("--2", "")

    for df in [df1, df2]:
        for col in df.columns:
            nice_col_name = col.title()
            nice_col_name = nice_col_name.replace("_", " ")
            df[nice_col_name] = df[col]

    data = []
    for _, x in df1.groupby(["Seed", "Date", "Solver", "Topology"]):
        uncoordinated = x[x["control"] == "uncoordinated_transformer"]
        alpha = uncoordinated["Trafo Congestion Time (hrs)"].item()
        beta = uncoordinated["Total Trafo Congestion (kW)"].item()

        if alpha > 0:
            if beta == 0:
                raise ValueError
            x["Transformer Relief (Time)"] = (1 - x["Trafo Congestion Time (hrs)"] / alpha)
            x["Transformer Relief (Load)"] = (1 - x["Total Trafo Congestion (kW)"] / beta)

        else:
            x["Transformer Relief (Time)"] = (1 - x["Trafo Congestion Time (hrs)"])
            x["Transformer Relief (Load)"] = (1 - x["Total Trafo Congestion (kW)"])


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


    df_grouped = pd.concat(data).query("control != "
                                       "'uncoordinated_transformer'")

    control_order = ['central','gaussian','step','cubic','cubic_restricted']

    for y in ["Transformer Relief (Time)", "Feeder Relief (Time)"]:

        data = df_grouped.fillna(0)
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

        data["control"] = data["control"].apply(lambda c: rename_control(c))

        data = data.pivot_table(index="control", columns="topology",
                                values=y,
                                aggfunc='mean')

        fig = plt.figure(constrained_layout=True)
        # plt.title(y)

        cmap = sns.color_palette("magma", as_cmap=True)
        ax = sns.heatmap(data, cmap=cmap , annot=True, linewidth=.5,
                         fmt=".2f", center=0, vmax=1.0, cbar=False, vmin=-2.0)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

        plt.savefig(plot_dir / f"{y.lower().replace(' ', '_')}.png")

    plt.show()