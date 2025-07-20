from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging

from grecco_sim.util.analysis import OptimizationRun, evaluate_congestion

logging.getLogger("pypsa").setLevel(logging.WARNING)


if __name__ == "__main__":
    feeder_congestion_count = pd.DataFrame()

    run_dir_1 = Path("/home/carl-wanninger/runs/experiment_3")
    run_dir_2 = Path("/home/carl-wanninger/runs/experiment_2")

    data_list = []
    congestion_by_feeder = []

    for p in run_dir_1.glob("run_*"):
        x = OptimizationRun(p)
        data = x.meta
        data["control"] += "_feeder"
        data["file_name"] = p.name
        data = evaluate_congestion(data=data)
        data_list.append(data)

        cg_by_feeder = x.feeder_congestion.melt()
        cg_by_feeder["file_name"] = p.name
        congestion_by_feeder.append(cg_by_feeder)

    for p in run_dir_2.glob("run_*"):
        x = OptimizationRun(p)
        data = x.meta
        data["control"] += "_transformer"
        data["file_name"] = p.name
        data = evaluate_congestion(data=data)
        data_list.append(data)

        cg_by_feeder = x.feeder_congestion.melt()
        cg_by_feeder["file_name"] = p.name
        congestion_by_feeder.append(cg_by_feeder)

    df = pd.DataFrame(data_list)
    df["topology"] = df["file_name"].apply(lambda x: x.split("_")[3])
    df["solver"] = df["file_name"].apply(lambda x: x.split("_")[2])
    df["topology"] = df["topology"].str.replace("simbench-LV-", "")
    df["topology"] = df["topology"].str.replace("--2", "")

    for y in ["Trafo Congestion Events (Count)",
              "Feeder Congestion Events (Count)"]:

        for date in df["date"].unique()[0:1]:
            data = df.query("date == @date").copy()
            data = data.fillna(0)
            data["topology"] = data["topology"] + "_" + data[
                "solver"].apply(lambda x: x[0])

            data = data.pivot_table(index="control", columns="topology",
                                    values=y,
                                    aggfunc='mean')
            plt.figure(constrained_layout=True)
            plt.title(y)
            ax = sns.heatmap(data, cmap="cool", annot=True, linewidth=.5,
                             fmt=".0f")
            ax.set(xlabel="", ylabel="")
            ax.xaxis.tick_top()

    df = pd.concat(congestion_by_feeder)
    df["topology"] = df["file_name"].apply(lambda x: x.split("_")[3])
    df["solver"] = df["file_name"].apply(lambda x: x.split("_")[2])
    df["topology"] = df["topology"].str.replace("simbench-LV-", "")
    df = df.query("value > 1").copy()
    for topology in df["topology"].unique():
        data = df.query("topology == @topology")
        plt.figure()
        sns.histplot(data, x="variable", hue="solver",
                     multiple="stack")
        plt.show()
