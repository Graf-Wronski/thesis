from pathlib import Path

import numpy as np
import yaml

from grecco_sim.graph.utils.format import Format
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.color_palette("Set2")

def main():
    result_dir = Format().output_root / "experiment_2"
    results_scalar = []

    params = ["ev", "bss", "baseload", "hp", "temperature", "soc",
              "p_trafo", "p_bat"]
    long_df_dict = {param: [] for param in params}



    for run_dir in result_dir.glob("run_*"):
        result_scalar = dict()

        with open(run_dir / 'meta.yaml', 'r') as f:
            meta = yaml.load(f, Loader=yaml.SafeLoader)

        meta["run_name"] = run_dir.name
        meta["Date"] = f"2023-{meta['month']}-{meta['day']}"
        meta["solver"] = meta["run_name"].split("_")[2]
        meta["Grid"] = meta["run_name"].split("_")[3]

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

        result_scalar["congestion_peak"] = np.max(congestions / transformer_limit)

        for param in []:
            ts_df = pd.read_csv(run_dir / f"{param}.csv", index_col=0)

            snapshots = pd.read_csv(run_dir / "network" / "snapshots.csv")
            ts_df["Time"] = pd.DatetimeIndex(snapshots["snapshot"])
            long_df = ts_df.melt(id_vars=['Time'], var_name='parameter',
                                 value_name='value')

            for key, val in meta.items():
                long_df[key] = val

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

            if (run_dir / f"realized_signals.csv").exists():

                signals = pd.read_csv(run_dir / f"realized_signals.csv",index_col=0)
                signals = signals.mean(axis=1)

                inflex_costs = ts_df[inflex_cols].values.sum(axis=None) * 0.66
                flex_costs = (signals * ts_df[flex_cols].sum(axis=1)).sum()
                flex_costs += ts_df[flex_cols].clip(lower=0).values.sum(axis=None) * 0.66
                flex_costs += ts_df[flex_cols].clip(upper=0).values.sum(axis=None) * 0.33

                costs = inflex_costs + flex_costs

                result_scalar["Costs"] = costs

            else:
                inflex_costs = ts_df[inflex_cols].values.sum(axis=None) * 0.66

                flex_costs = ts_df[flex_cols].clip(lower=0).values.sum(
                    axis=None) * 0.66
                flex_costs += ts_df[flex_cols].clip(upper=0).values.sum(
                    axis=None) * 0.33

                costs = inflex_costs + flex_costs
                result_scalar["Costs"] = costs
            # Create new DataFrame with p_flex and p_inflex

        results_scalar.append(result_scalar)

    plot_dir = Path("/home/carl-wanninger/plots/experiment_2")
    df = pd.DataFrame(results_scalar)

    for param, df_list in long_df_dict.items():
        if len(df_list) == 0:
            continue
        long_df_dict[param] = pd.concat(df_list, ignore_index=True)

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

    df["Total Congestion (p. u.)"] = df["Congestion Events (Count)"] * df[
        "average_congestion_size"]
    df["Average Congestion (p. u.)"] = df["average_congestion_size"]
    df["Total Absolute Load (kW)"] = df["total_absolute_load"]
    df["Total Load"] = df["total_load"]

    df["Solver"] = df["solver"]
    df["Control"] = df["control"]

    df.fillna(0)

    for y in ["Total Congestion (p. u.)", "Average Congestion (p. u.)",
              "Total Absolute Load (kW)"]:

        g = sns.catplot(
            data=df,
            x="Control",
            y=y,
            hue="Solver",
            col="Date",  # faceting by second binary variable
            kind="violin",
            split=True,  # only works if hue has 2 levels
            inner="quartile",  # can be "box", "point", or "stick"
            col_wrap=2,
            height=4,
            cut=0,
            palette="Set2",
            density_norm="width",
            aspect=1)

        hue_levels = df["Control"].unique()

        for ax, (date_val, subdata) in zip(g.axes.flat, df.groupby("Date")):
            # For each x-category (Solver) and hue level
            for i, solver_val in enumerate(sorted(subdata["Solver"].unique())):
                for j, hue_val in enumerate(hue_levels):
                    # Count points in this group
                    count = len(subdata[(subdata["Solver"] == solver_val) & (
                                subdata["Control"] == hue_val)])

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

        plt.savefig(plot_dir / f"{y}.png")

    df["Runtime (s)"] = df["runtime"]

    _ = sns.relplot(df, x="control", y="Runtime (s)", hue="Solver", marker="x",
                    kind="line", palette="Set2")

    plt.savefig(plot_dir / f"Runtime.png")

    _ = sns.relplot(df, x="control", y="Average Congestion (p. u.)", hue="Solver", marker="x",
                    kind="line", palette="Set2")

    plt.savefig(plot_dir / f"Average Congestion.png")

    _ = sns.relplot(df, x="control", y="Total Congestion (p. u.)",
                    hue="Solver", marker="x",
                    kind="line", palette="Set2")

    plt.savefig(plot_dir / f"Total Congestion.png")

    _ = sns.relplot(df, x="control", y="Congestion Events (Count)", hue="Solver", marker="x",
                    kind="line", palette="Set2")

    plt.savefig(plot_dir / f"Congestion Events (Count).png")

    # plt.show()

    # The first experiment should answer 3 questions:
    # What are differences between osqp and gurobi? Which should we prefer?

    # Does it make a difference if we model heat pumps continously or discrete?

    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    """data = long_df_dict["p_bat"][long_df_dict["p_bat"]["day"] == 30]

    sns.histplot(
        data.query("Event == 'Discharge'"),
        x="Time", hue="solver",
        multiple="stack",
        palette="Set2",
        edgecolor=".3",
        linewidth=.5)"""

    """_ = sns.boxplot(df, x="date", y="Costs", hue="Solver")"""

    """for date in df["date"].unique():
        data = df.query("date == @date")
        plt.figure()
        _ = sns.lineplot(data, x="Horizon", y="Costs", hue="Solver")"""

    for date in df["date"].unique()[0:2]:
        data = df.query("date == @date")
        for y in ["Total Congestion (p. u.)", "Costs"]:
            plt.figure()
            sns.boxplot(data, x="seed", y=y, hue="Solver", palette="Set2")

    for date in df["date"].unique():
        data = df.query("date == @date and Solver == 'gurobi'").copy()
        data = data.pivot_table(index="Control", columns="Grid",
                          values="Total Congestion (p. u.)",
                          aggfunc='mean')
        plt.figure()
        sns.heatmap(data, cmap="Reds")

    plt.show()


    # Which horizon should we choose?



if __name__ == "__main__":
    main()