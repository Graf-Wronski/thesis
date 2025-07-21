from pathlib import Path

import numpy as np
import yaml

from grecco_sim.graph.utils.format import Format
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="darkgrid")


def build_data(result_dir: Path, force_update: bool = False):
    results_scalar = []
    tmp_dir = Format().data_root / "tmp"

    if not force_update:
        if (tmp_dir / "exp_1_results_scalar.csv").exists():
            #if (tmp_dir / "exp_1_long_df.csv").exists():
            return pd.read_csv(tmp_dir / "exp_1_results_scalar.csv")
            #            pd.read_csv(tmp_dir / "exp_1_long_df.csv"))


    params = ["ev", "bss", "baseload", "hp", "temperature", "soc",
              "p_trafo", "p_bat"]
    long_df_dict = {param: [] for param in params}

    n_files = len([x for x in result_dir.glob("run_*")])

    for i, run_dir in enumerate(result_dir.glob("run_*"), 1):
        print(f"Processing {i}/{n_files}: {run_dir}")
        result_scalar = dict()

        with open(run_dir / 'meta.yaml', 'r') as f:
            meta = yaml.load(f, Loader=yaml.SafeLoader)

        meta["run_name"] = run_dir.name
        meta["Date"] = f"2023-{meta['month']}-{meta['day']}"

        # Skip bad values.
        if meta["end"] == "2023-08-30 07:30:00":
            print(f"Bad run: {meta['run_name']}")
            continue

        result_scalar.update(meta)

        trafo_ts = pd.read_csv(run_dir / "p_trafo.csv", index_col=0)

        result_scalar["total_load"] = trafo_ts.sum().iloc[0]
        result_scalar["total_absolute_load"] = trafo_ts.abs().sum().iloc[0]

        transformer_limit = meta["transformer_lim"] * 1000
        congestions = trafo_ts[trafo_ts > transformer_limit].dropna()

        n_congestion_events = len(congestions)
        result_scalar["Congestion Events (Count)"] = n_congestion_events

        avg_congestion_size = np.mean(congestions / transformer_limit)
        result_scalar["average_congestion_size"] = avg_congestion_size

        result_scalar["congestion_peak"] = np.max(
            congestions / transformer_limit)

        for param in []:
            ts_df = pd.read_csv(run_dir / f"{param}.csv", index_col=0)

            snapshots = pd.read_csv(run_dir / "network" / "snapshots.csv")
            ts_df["Time"] = pd.DatetimeIndex(snapshots["snapshot"])
            long_df = ts_df.melt(id_vars=['Time'], var_name='parameter',
                                 value_name='value')

            for key, val in meta.items():
                long_df[key] = val

            long_df_dict[param].append(long_df)

        for param in ["p_bat"]:
            ts_df = pd.read_csv(
                run_dir / f"state_ts.csv",
                usecols=lambda x: x[-8:] == "bat_p_dc",
                index_col=0)

            snapshots = pd.read_csv(run_dir / "network" / "snapshots.csv")
            ts_df["Time"] = pd.DatetimeIndex(snapshots["snapshot"])
            long_df = ts_df.melt(id_vars=['Time'], var_name='parameter',
                                 value_name='value')

            for key, val in meta.items():
                long_df[key] = val

            long_df["Event"] = "None"

            long_df_dict[param].append(long_df)

        for param in ["costs"]:
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

            signals = pd.read_csv(run_dir / f"realized_signals.csv",
                                  index_col=0)
            signals = signals.mean(axis=1)

            inflex_capacity_costs = ts_df[inflex_cols].values.sum(axis=None) * 0.66
            signal_costs = (signals * ts_df[flex_cols].sum(axis=1)).sum()
            flex_capacity_costs = ts_df[flex_cols].clip(lower=0).values.sum(
                axis=None) * 0.66
            flex_capacity_costs += ts_df[flex_cols].clip(upper=0).values.sum(
                axis=None) * 0.33

            costs = inflex_capacity_costs + flex_capacity_costs + signal_costs

            # /4 because of 15-minute resolution.
            result_scalar["Total Costs"] = costs / 4
            result_scalar["Inflexible Capacity Costs"] = inflex_capacity_costs / 4
            result_scalar["Flexible Capacity Costs"] = flex_capacity_costs / 4
            result_scalar["Flexible Signal Costs"] = signal_costs / 4
            result_scalar["Capacity Costs"] = (inflex_capacity_costs + flex_capacity_costs) / 4
            
            # Create new DataFrame with p_flex and p_inflex

        results_scalar.append(result_scalar)

    df = pd.DataFrame(results_scalar)
    df.to_csv(tmp_dir / "exp_1_results_scalar.csv")

    """
    for param, df_list in long_df_dict.items():
        if len(df_list) == 0:
            continue
        long_df_dict[param] = pd.concat(df_list, ignore_index=True)

    long_df_dict["p_bat"].loc[
        long_df_dict["p_bat"]["value"] < 1.0, "Event"] = "Charge"
    long_df_dict["p_bat"].loc[
        long_df_dict["p_bat"]["value"] > 1.0, "Event"] = "Discharge"
        
    long_df = pd.DataFrame
    long_df.to_csv(tmp_dir / "exp_1_long_df.csv")"""

    return df


def plot_wall_clock_time(df: pd.DataFrame, plot_dir):
    df["Wall Clock Time (s)"] = df["runtime"]

    _ = sns.relplot(df, x="Horizon", y="Wall Clock Time (s)", hue="Solver",
                    marker="o",
                    kind="line", palette="Set2")

    plt.savefig(plot_dir / f"Wall Clock Time (s).png")


def plot_congestion_time(df: pd.DataFrame, plot_dir):
    g = sns.catplot(
        data=df,
        x="Heat Pump Model",
        y="Congestion Time (hrs)",
        hue="Solver",  # used for splitting the violin
        col="Date",  # faceting by second binary variable
        kind="box",
        col_wrap=2,
        height=4,
        palette="Set2")

    hue_levels = df["Heat Pump Model"].unique()

    for ax, (date_val, subdata) in zip(g.axes.flat, df.groupby("Date")):
        # For each x-category (Solver) and hue level
        for i, solver_val in enumerate(sorted(subdata["Solver"].unique())):
            for j, hue_val in enumerate(hue_levels):
                # Count points in this group
                count = len(subdata[(subdata["Solver"] == solver_val) & (
                        subdata["Heat Pump Model"] == hue_val)])

                # Calculate x position for annotation:
                # Violin split places two halves close, so shift the text slightly left or right
                # 'i' is the categorical x-position (0-based)
                # split violin: left half = hue_levels[0], right half = hue_levels[1]
                x_pos = i + (-0.15 if hue_val == hue_levels[0] else 0.15)

                # y position - put text near top of axis
                y_pos = ax.get_ylim()[1] * 0.95

                ax.text(
                    x_pos, y_pos,
                    f"n={count}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black"
                )

    plt.savefig(plot_dir / "Congestion Time (hrs).png")


def plot_cost_estimation(df: pd.DataFrame, plot_dir: Path):

    df["Total Costs (€)"] = (df["Total Costs"] / 0.66) * 0.24

    data = df.melt(
        value_vars=["Flexible Signal Costs", "Capacity Costs"],
        id_vars=["Horizon", "Solver"],
        var_name="Cost Source",
        value_name="Cost Estimation (€)")

    data["Cost Source"] = data["Cost Source"].str.lower()
    data["Cost Source"] = data["Cost Source"].str.replace(" costs", "")
    data["Cost Source"] = data["Cost Source"].str.replace("flexible ", "")

    fig, ax = plt.subplots()
    sns.lineplot(df, x="Horizon", y="Total Costs (€)",
                 hue="Solver", style="Heat Pump Model", ax=ax)

    fig, ax = plt.subplots()
    sns.lineplot(data, x="Horizon", y="Cost Estimation (€)", hue="Solver",
                 style="Cost Source", ax=ax, palette="Set2")

    plt.tight_layout()

    plt.savefig(plot_dir / f"cost_estimation.png")


def plot_congestion_time_by_horizon(df: pd.DataFrame, plot_dir: Path):
    fig, ax = plt.subplots()
    sns.relplot(df, x="Horizon", y="Congestion Time (hrs)", hue="Solver",
                marker="o", kind="line", palette="Set2", ax=ax)

    plt.savefig(plot_dir / f"congestion_time_by_horizon.png")


def main():
    result_dir = Format().output_root / "experiment_1"
    df = build_data(result_dir)

    plot_dir = Path("/home/carl-wanninger/plots/experiment_1")

    df["Total Congestion (p. u.)"] = df["Congestion Events (Count)"] * df[
        "average_congestion_size"]
    df["Average Congestion (p. u.)"] = df["average_congestion_size"]
    df["Total Absolute Load (kW)"] = df["total_absolute_load"]
    df["Total Load"] = df["total_load"]
    df["Heat Pump Model"] = df["heat_pump_model"]
    df["Solver"] = df["solver"]
    df["Congestion Time (hrs)"] = df["Congestion Events (Count)"] / 4
    df["Horizon"] = df["horizon"]
    df.fillna(0)

    plot_congestion_time(df, plot_dir)
    plot_cost_estimation(df, plot_dir)

    plt.show()


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



    """data = long_df_dict["p_bat"][long_df_dict["p_bat"]["day"] == 30]

    sns.histplot(
        data.query("Event == 'Discharge'"),
        x="Time", hue="solver",
        multiple="stack",
        palette="Set2",
        edgecolor=".3",
        linewidth=.5)"""

    """_ = sns.boxplot(df, x="date", y="Total Costs", hue="Solver")"""

    """for date in df["date"].unique():
        data = df.query("date == @date")
        plt.figure()
        _ = sns.lineplot(data, x="Horizon", y="Total Costs", hue="Solver")"""

    """for date in df["date"].unique()[0:2]:
        data = df.query("date == @date")
        for y in ["Total Congestion (p. u.)", "Total Costs", "Runtime (s)"]:
            plt.figure()
            sns.boxplot(data, x="Heat Pump Model", y=y, hue="Solver")
            # x = seed
            
"""




    # Which horizon should we choose?



if __name__ == "__main__":
    main()