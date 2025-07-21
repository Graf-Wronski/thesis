from pathlib import Path

import numpy as np
import yaml

from grecco_sim.graph.utils.format import Format
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from grecco_sim.util.analysis import OptimizationRun, evaluate_congestion

sns.color_palette("Set2")
sns.set_theme(style="darkgrid")


def rename_grid(old_name: str) -> str:
    new_name = old_name.replace("simbench-LV-", "")
    new_name = new_name.replace("--2", "")

    name_map = dict()
    name_map["rural1"] = "rural-1"
    name_map["rural2"] = "rural-2"
    name_map["rural3"] = "rural-3"
    name_map["semiurb4"] = "semiurb-1"
    name_map["semiurb5"] = "semiurb-2"
    name_map["urban6"] = "urban-1"

    return name_map[new_name]


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

    if scalar_dict["control"] not in ["central", "uncoordinated"]:
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


def build_data(result_dir: Path, force_update: bool = False):
    results_scalar = []
    tmp_dir = Format().data_root / "tmp"

    if not force_update:
        if (tmp_dir / "exp_2_results_scalar.csv").exists():
            # if (tmp_dir / "exp_2_long_df.csv").exists():
            return pd.read_csv(tmp_dir / "exp_2_results_scalar.csv")
            #            pd.read_csv(tmp_dir / "exp_2_long_df.csv"))

    n_files = len([x for x in result_dir.glob("run_*")])

    for i, run_dir in enumerate(result_dir.glob("run_*"), 1):
        print(f"Processing {i}/{n_files}: {run_dir}")

        x = OptimizationRun(run_dir)
        result_scalar = x.meta
        result_scalar["file_name"] = run_dir.name
        result_scalar = evaluate_congestion(data=result_scalar, x=x)
        result_scalar["run_dir"] = run_dir
        result_scalar["run_name"] = run_dir.name
        result_scalar["Date"] = f"2023-{result_scalar['month']}-{result_scalar['day']}"

        add_costs(result_scalar)

        # Create new DataFrame with p_flex and p_inflex

        results_scalar.append(result_scalar)

    df = pd.DataFrame(results_scalar)
    df.to_csv(tmp_dir / "exp_2_results_scalar.csv")

    return df


def main():
    result_dir = Format().output_root / "experiment_2"

    df = build_data(result_dir)

    plot_dir = Path("/home/carl-wanninger/plots/experiment_2")

    """Index(['date', 'day', 'end', 'feeder_lim', 'heat_pump_model', 'horizon',
       'month', 'runtime', 'seed', 'solver', 'start', 'topology',
       'transformer_lim', 'total_load', 'total_absolute_load',
       'Congestion Events (Count)', 'average_congestion_size',
       'congestion_peak'],
      dtype='object')"""

    # soc_data = long_df_dict["soc"]

    # hp_data = long_df_dict["hp"]
    # ev_data = long_df_dict["ev"]

    # p_ev = ev_data[ev_data["parameter"].str.contains("p_ac_set")]
    # sns.boxplot(p_ev[p_ev["value"] > 0], x="time", y="solver", hue="solver")

    for col in df.columns:
        nice_col_name = col.title()
        nice_col_name = nice_col_name.replace("_", " ")
        df[nice_col_name] = df[col]

    print(df.columns)

    df["Solver"] = df["run_name"].apply(lambda x: x.split("_")[2])
    df["Topology"] = df["run_name"].apply(lambda x: x.split("_")[3])
    df["Topology"] = df["Topology"].apply(lambda x: rename_grid(x))
    df.fillna(0)

    print(df["date"].unique())
    # data = df[df["date"] == df["date"].unique()[0]]

    data = []
    for _, x in df.groupby(["Seed", "Date", "Solver", "Topology"]):
        normalization_value = (
        x[x["control"] == "uncoordinated"]["Trafo Congestion Time (hrs)"]).item()
        x["Trafo Congestion Time (hrs) Compared To Uncoordinated"] = \
            (x["Trafo Congestion Time (hrs)"] - normalization_value)
        normalization_value = (
            x[x["control"] == "uncoordinated"]["Total Costs"]).item()
        x["Total Costs Compared To Uncoordinated"] = \
            (x["Total Costs"] - normalization_value)
        data.append(x)

    df_grouped = pd.concat(data).query("control != 'uncoordinated'")

    control_order = ['central','gaussian','step','cubic','cubic_restricted']

    g = sns.catplot(
        data=df_grouped, kind="swarm",
        y="Control", x="Trafo Congestion Time (hrs) Compared To "
                       "Uncoordinated", hue="Solver",
        errorbar="sd", palette="Set2", alpha=.6, height=6,
        col="Topology", aspect=.5, dodge=True, order=control_order,
    )
    g.despine(left=True)

    for ax in g.axes.flat:
        ax.axvline(x=0, color='black', linewidth=1.5, zorder=0)
    g.axes.flat[0].set(xlabel=None)
    g.axes.flat[2].set(xlabel=None)

    plt.savefig(plot_dir / "congestion_compared_to_uncoordinated.png")

    g = sns.catplot(
        data=df_grouped, kind="box",
        y="Control", x="Trafo Congestion Time (hrs) Compared To Uncoordinated",
        hue="Solver", palette="Set2",height=6,
        col="Topology", order=control_order,
    )
    g.despine(left=True)

    for ax in g.axes.flat:
        ax.axvline(x=0, color='black', linewidth=1.5, zorder=0)
    g.axes.flat[0].set(xlabel=None)
    g.axes.flat[2].set(xlabel=None)

    plt.savefig(plot_dir / "congestion_compared_to_test.png")

    df_grouped["Total Costs (€) Compared To Uncoordinated"] = (
        (df_grouped["Total Costs Compared To Uncoordinated"] / 0.66) * 0.24)

    g = sns.catplot(
        data=df_grouped, kind="swarm",
        y="Control", x="Total Costs (€) Compared To Uncoordinated",
        hue="Solver",
        errorbar="sd", palette="Set2", alpha=.6, height=6,
        col="Topology", aspect=.5, dodge=True, order=control_order,
    )
    g.despine(left=True)

    for ax in g.axes.flat:
        ax.axvline(x=0, color='black', linewidth=1.5, zorder=0)
    g.axes.flat[0].set(xlabel=None)
    g.axes.flat[2].set(xlabel=None)

    plt.savefig(plot_dir / "total_costs_compared_to-uncoordinated.png")

    plt.show()


if __name__ == "__main__":
    main()