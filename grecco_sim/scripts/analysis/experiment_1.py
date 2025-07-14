from pathlib import Path

import numpy as np
import yaml

from grecco_sim.graph.utils.format import Format
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def main():
    result_dir = Format().output_root / "experiment_1"
    results_scalar = []

    params = ["ev", "bss", "baseload", "hp", "temperature", "soc", "p_trafo"]
    # long_df_dict = {param: [] for param in params}

    for run_dir in result_dir.glob("run_*"):
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

        result_scalar["congestion_peak"] = np.max(congestions / transformer_limit)
        results_scalar.append(result_scalar)

        for param in params:
            ts_df = pd.read_csv(run_dir / f"{param}.csv", index_col=0)
            ts_df["time"] = pd.DatetimeIndex(ts_df.index)
            long_df = ts_df.melt(id_vars=['time'], var_name='parameter',
                                 value_name='value')

            for key, val in meta.items():
                long_df[key] = val

            # long_df_dict[param].append(long_df)

    plot_dir = Path("/home/carl-wanninger/plots")
    df = pd.DataFrame(results_scalar)

    # for param, df_list in long_df_dict.items():
    #    long_df_dict[param] = pd.concat(df_list, ignore_index=True)

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

    df["Total Congestion"] = df["Congestion Events (Count)"] * df[
        "average_congestion_size"]
    df["Average Congestion Size"] = df["average_congestion_size"]
    df["Total Absolute Load"] = df["total_absolute_load"]
    df["Total Load"] = df["total_load"]
    df["Heat Pump Model"] = df["heat_pump_model"]
    df["Solver"] = df["solver"]

    df.fillna(0)

    for y in ["Total Congestion", "Average Congestion Size",
              "Total Absolute Load"]:

        g = sns.catplot(
            data=df,
            x="Solver",
            y=y,
            hue="Heat Pump Model",  # used for splitting the violin
            col="Date",  # faceting by second binary variable
            kind="violin",
            split=True,  # only works if hue has 2 levels
            inner="quartile",  # can be "box", "point", or "stick"
            col_wrap=2,
            height=4,
            cut=0,
            density_norm="width",
            aspect=1)

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

        #        g = sns.FacetGrid(df, col="date", hue="solver")
        # g.map_dataframe(sns.violinplot, y=y, x="horizon" , split=True)
        #g.add_legend()

    df["Runtime"] = df["runtime"]
    df["Horizon"] = df["horizon"]
    plt.figure()
    _ = sns.relplot(df, x="Horizon", y="Runtime", hue="Solver", marker="x",
                    kind="line")
    plt.show()

    plt.figure()
    _ = sns.relplot(df, x="Horizon", y="Average Congestion Size", hue="Solver", marker="x",
                    kind="line")
    plt.show()

    plt.figure()
    _ = sns.relplot(df, x="Horizon", y="Congestion Events (Count)", hue="Solver", marker="x",
                    kind="line")
    plt.show()

    # The first experiment should answer 3 questions:
    # What are differences between osqp and gurobi? Which should we prefer?

    # Does it make a difference if we model heat pumps continously or discrete?

    # Which horizon should we choose?



if __name__ == "__main__":
    main()