from typing import Any

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from grecco_sim.util import plot


class Plotter:
    def __init__(self, sim: Any):

        self.simulation = sim
        self.sim_config = sim.config
        self.dataloader = sim.dataloader
        self.results = sim.results
        self.grid = sim.grid
        self.plot_dir = self.sim_config.plot_dir / str(self.simulation)

        if not(self.plot_dir.exists()):
            self.plot_dir.mkdir(parents=True)

        # Default color_list from GrECCo
        self.color_list = [c[1] for c in mcolors.XKCD_COLORS.items()]

        sns.set_theme(style="whitegrid")

    @property
    def max_market_iterations(self) -> int:
        return self.sim_config.market_config.max_market_iterations

    @property
    def dina4(self) -> dict[str, tuple[float, float]]:
        return {"portrait": (8.27, 11.69), "landscape": (11.69, 8.27)}

    @property
    def sys_ids(self) -> list[str]:
        return [str(x) for x in self.simulation.nodes]

    @property
    def n_agents(self) -> int:
        return len(self.simulation.nodes)

    @property
    def trafo_p_lim_kw(self) -> float:
        return self.simulation.grid.trafo_p_lim * 1000 # mW -> kW

    @property
    def time_index(self) -> pd.DatetimeIndex:
        return self.simulation.index

    def plot(self):
        self.model_power()

        # self.plot_signal_development()
        # self.plot_market_iterations()

        # self.plot_promises()

        if self.sim_config.use_batteries:
            self.plot_bat_power()
            self.plot_bat_soc()

        if self.sim_config.use_heatpumps:
            self.plot_hp_power()
            self.plot_hp_temp_in()

        if self.sim_config.use_ev:
            self.plot_ev_power()

        # ToDo: This seems very specific.
        if self.sim_config.coordinator_name != "local_self_suff" and False:
            self.plot_assigned_grid_fees()

        plt.show()

    def plot_congesiton(self):
        pass

    def model_power(self):

        fig, ax = plt.subplots(1)
        ax.set_title("Cumulative Loads")
        ax.set_ylabel(f"Load kW")


        p_trafo_ts = self.results.p_trafo_ts
        ax = sns.lineplot(data=p_trafo_ts, label="Transformer",
                           color="blue", drawstyle='steps-pre')

        load_p_ts = self.results.state_ts(key1="baseload", key2="p_model")
        load_p = pd.DataFrame({"P": load_p_ts.sum(axis=1), "type": "Baseload"})
        pv_p_ts = self.results.state_ts(key1="pv", key2="p_model")
        pv_p = pd.DataFrame({"P": pv_p_ts.sum(axis=1), "type": "PV"})
        bat_p_ts = self.results.state_ts(key1="bat", key2="p_model")
        bat_p = pd.DataFrame({"P": bat_p_ts.sum(axis=1), "type": "Bat"})
        hp_p_ts = self.results.state_ts(key1="hp", key2="p_model")
        hp_p = pd.DataFrame({"P": hp_p_ts.sum(axis=1), "type": "HP"})
        ev_p_ts = self.results.state_ts(key1="ev", key2="p_model")
        ev_p = pd.DataFrame({"P": ev_p_ts.sum(axis=1), "type": "EV"})
        data = pd.concat([load_p, pv_p, bat_p, hp_p, ev_p])
        data["Time"] = pd.DatetimeIndex(data.index)
        sns.lineplot(data, ax=ax, x="Time", y="P", hue="type",
                     drawstyle='steps-pre')

        plt.hlines([-self.trafo_p_lim_kw, self.trafo_p_lim_kw],
                   xmin=p_trafo_ts.index[0],
                   xmax=p_trafo_ts.index[-1],
                   color="red",
                   linestyle="dashed", label="Transformer limit")

        # Design y-axis limits symmetrically.
        low, high = plt.ylim()
        ax.set_ylim(-max(abs(low), abs(high)), max(abs(low), abs(high)))
        ax.legend(loc="upper right", title="Legend")

        plot.set_two_hours_x_axis(ax)

        fig.savefig(self.plot_dir / f"p_model.pdf")

    def plot_ev_power(self):
        ev_data = self.results.state_ts("ev_p_model")
        self.cummulated_unit_plot(
            data=ev_data,
            title="EV power",
            value_name="Power (kW)")

    def plot_hp_power(self):
        hp_data = self.results.state_ts("hp_p_model")
        ax = self.cummulated_unit_plot(
            data=hp_data,
            title="Heat pump power",
            value_name="Power (kW)")

    def plot_hp_temp_in(self):
        fig, ax = plt.subplots()
        temp_data = self.results.state_ts("hp_temp_in")
        sns.lineplot(data=temp_data, ax=ax, drawstyle='steps-pre')
        ax.set_title("Thermal system temperatures")
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature / C")
        fig.tight_layout()
        # Show only hours with a step size of 2.
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.get_legend().remove()

    def plot_bat_power(self):
        bat_data = self.results.state_ts("bat_p_model")
        self.cummulated_unit_plot(
            data=bat_data,
            title="Battery Power",
            value_name="Power (kW)")

    @staticmethod
    def cummulated_unit_plot(
            data: pd.DataFrame,
            title: str,
            value_name: str)-> plt.Axes:

        data = data.reset_index().melt(id_vars="index", var_name="Unit",
                                       value_name=value_name)
        data = data.rename(columns={"index": "Time"})

        fig, ax = plt.subplots()
        fig.suptitle(title)

        ax = sns.lineplot(data, x="Time", y=value_name, hue="Unit", ax=ax,
                             alpha=0.4, drawstyle='steps-pre')

        # Show only hours with a step size of 2.
        ax = plot.set_two_hours_x_axis(ax)
        ax.get_legend().remove()
        fig.tight_layout()

        return ax

    def plot_assigned_grid_fees(self):
        data = ...
        self.cummulated_unit_plot(
            data=data,
            title="Signal incurred costs",
            value_name="Costs (Euro)")

    def plot_bat_soc(self):
        bat_data = self.results.state_ts("bat_soc")
        self.cummulated_unit_plot(
            data=bat_data,
            title="Battery SoC",
            value_name="SoC (0 - 1)")

    def plot_cummulated_loads(self):
        data = self.results.state_ts("p_model")

    def plot_signal_development(self):
        n_interactions = self.max_market_iterations

        n_plots = int(np.ceil((n_interactions - 1) / 4 ))

        # Signals: spans 1 row
        # Loads: spans 2 rows.

        for page in range(n_plots):

            fig = plt.figure(figsize=self.dina4["landscape"])

            gs = gridspec.GridSpec(nrows=12, ncols=1, figure=fig, hspace=0.4)
            row_idx = 0
            # Plot signals and cummulaitve loads alternating.

            def get_promise(df, lookback, k) -> np.ndarray:
                """Promise for each time step from the past."""
                result = np.zeros(len(self.time_index))
                valid_range = np.arange(lookback, len(self.time_index))
                idx = np.minimum(self.results.market_iterations, k).astype(int)
                result[valid_range] = df[valid_range - lookback, idx, lookback]

                return result

            load_p_ts = self.results.state_ts(key1="baseload", key2="p_model")
            pv_p_ts = self.results.state_ts(key1="pv", key2="p_model")
            inflex_p_ts = (load_p_ts + pv_p_ts).sum(axis=1)
            p_hp_cum_signal = np.sum(list(self.results.p_hp.values()), axis=0)
            p_bat_cum_signal = np.sum(list(self.results.p_bat.values()), axis=0)

            for i in range(4 * page, 4 * page + 4):

                if i == n_interactions:
                    continue

                # Plot signals.
                ax_signals = fig.add_subplot(gs[row_idx: row_idx + 1, 0])
                row_idx += 1

                sample_bus = self.grid.sys_ids[0]

                for horizon in [0]:
                    data = get_promise(self.results.signals[sample_bus],
                                       horizon, i)
                    signal_df = pd.DataFrame(data, index=self.time_index,
                                             columns=[f"Horizon {horizon}"])
                    alpha = 1 - 0.05 * horizon
                    sns.lineplot(data=signal_df, ax=ax_signals, alpha=alpha, drawstyle='steps-pre')

                ax_signals.set_ylabel("Signal (€)")
                ax_signals.set_xlabel("Time")
                ax_signals.set_title(f"Signals {i}.")
                # ax_signals.set_ylim((-6., 6.))
                ax_signals.get_xaxis().set_visible(False)

                # Plot cummulative loads.
                ax_cum_loads = fig.add_subplot(gs[row_idx: row_idx + 2, 0])
                row_idx += 2
                sns.lineplot(data=inflex_p_ts, ax=ax_cum_loads, label="Inflex",
                             color="black", drawstyle='steps-pre')

                for horizon in [0]:
                    alpha = 1 - 0.08 * horizon
                    legend = True if horizon == 0 else False

                    # Plot heat pumps.
                    data = get_promise(p_hp_cum_signal, horizon, i)
                    hp_df = pd.DataFrame(data, index=self.time_index, columns=["HP"])
                    hp_df.plot(ax=ax_cum_loads, alpha=alpha, color="purple",
                               legend=legend, drawstyle='steps-pre')

                    # Plot batteries.
                    data = get_promise(p_bat_cum_signal, horizon, i)
                    bat_df = pd.DataFrame(data, index=self.time_index, columns=["Bat"])
                    bat_df.plot(ax=ax_cum_loads, alpha=alpha, color="green",
                                legend=legend, drawstyle='steps-pre')

                ax_cum_loads.set_ylabel("Loads (kWh)")
                # ax_cum_loads.set_ylim((-3 * self.trafo_p_lim_kw,
                # 3 * self.trafo_p_lim_kw))
                ax_cum_loads.set_title(f"Cumulative loads {i}.")

                if i < n_interactions - 1:
                    ax_cum_loads.get_xaxis().set_visible(False)
                    ax_cum_loads.get_legend().remove()
                    ax_signals.get_legend().remove()
                else:
                    plot.set_two_hours_x_axis(ax_cum_loads)

    def plot_promises(self):
        """ Compare promised schedules with promised signals at different
            time steps. """

        sns.set_theme(style="ticks")

        # DataFrame with cols t (time step), k (market iteration),
        # promised signal and promised schedule. Index are timestamps.
        any_bus = self.sys_ids[0]
        schedules = np.sum(list(self.results.p_grid.values()), axis=0)
        signals = self.results.signals[any_bus]

        def unfold_3d_array(arr: np.ndarray) -> list:
            """ Unfold a 3 dimensional array into a value list. """
            rows = []
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    for k in range(arr.shape[2]):
                        rows.append([i, j, k, arr[i, j, k]])
            return rows

        signal_df = pd.DataFrame(
            data = unfold_3d_array(signals),
            columns=["Step", "Market iteration", "Optimizer step", "Signal"])

        schedule_df = pd.DataFrame(
            data=unfold_3d_array(schedules),
            columns=["Step", "Market iteration", "Optimizer step", "Schedule"])

        df = pd.merge(schedule_df, signal_df, how="outer",
                      on=["Step", "Market iteration", "Optimizer step"])

        def facetgrid_two_axes(data: pd.DataFrame, x_name: str, y1_name: str,
                               y2_name: str, ylabel1: str,
                               **kwargs):


            # Add a fictive value to data, for step plots.
            data = data[data["Signal"].notna()]
            row = data.loc[data['Optimizer step'].idxmax()].copy()
            row['Optimizer step'] += 1
            data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)

            ax1 = plt.gca()
            sns.lineplot(x=data[x_name], y=data[y1_name], alpha=1., ax=ax1,
                         drawstyle='steps-post', label=y1_name, color='blue')
            ax1.set_ylabel(ylabel1)
            opt_horizon = self.sim_config.optimizer_config.horizon
            ax1.set(xlim=(-1, opt_horizon))
            # ax1.set(ylim=(-5., 5.))
            ax1.get_legend().remove()

            ax2 = ax1.twinx()
            sns.lineplot(x=data[x_name], y=data[y2_name], alpha=1., ax=ax2,
                         color="orange", drawstyle='steps-post',
                         label=y2_name)
            # ax2.set_ylim((-schedule_df["Schedule"].max() - 0.5,
            #              schedule_df["Schedule"].max() + 0.5))
            ax2.tick_params(right=False)
            ax2.set(yticklabels=[])
            ax2.set(ylabel=None)
            ax2.get_legend().remove()

        # Interesting steps are those with actions on the market.
        interesting_steps = np.where(self.results.market_iterations > 0)[0]
        n_samples = min(len(interesting_steps), 5)
        steps = np.random.choice(interesting_steps, n_samples, replace=False)

        for t in steps:
            k = self.results.market_iterations[t]

            data = df.query("Step == @t and `Market iteration` <= @k")
            col_wrap = max(4, int(k / 3))
            grid = sns.FacetGrid(data, col="Market iteration", col_wrap=col_wrap)

            grid.map_dataframe(facetgrid_two_axes, x_name="Optimizer step", y1_name="Signal",
                                y2_name="Schedule",
                                ylabel1='Signal (€)')

            for idx, ax in enumerate(grid.axes):
                if (idx + 1) % col_wrap == 0 or idx == len(grid.axes) - 1:
                    right_ax = ax.twinx()
                    right_ax.set_ylabel("Schedule (kW)")
                    y_max = schedule_df["Schedule"].max()
                    right_ax.set_ylim(-y_max-0.5, y_max+0.5)

            grid.fig.tight_layout(w_pad=1)

            # Add a custom legend
            legend_elements = [
                plt.Line2D([0], [0], color='blue', lw=2, label='Signal'),
                plt.Line2D([0], [0], color='orange', lw=2, label='Schedule')]
            grid.axes[-1].legend(handles=legend_elements, loc='lower right')

            grid.fig.subplots_adjust(top=0.8 + k * 0.01)
            timestamp = self.time_index[t].strftime("%Y-%m-%d %H:%M:%S")
            grid.fig.suptitle(f'Fee interaction at {timestamp}.')

    def plot_market_iterations(self):
        fig, ax = plt.subplots()
        fig.suptitle("Market Iterations")
        # Plus one as iterations start at 0.
        data = self.results.market_iterations
        ax = sns.scatterplot(x=self.time_index, y=data)
        ax = plot.set_two_hours_x_axis(ax)

        # Add a line to mark maximum iterations.
        plt.hlines(
            y=self.max_market_iterations - 1,
            xmin=self.time_index[0],
            xmax=self.time_index[-1],
            color="red",
            alpha=0.5,
            linestyle="dashed",
            label="Maximum iterations")

        ax.set_ylim((0, self.max_market_iterations + 2))
        ax.set_xlabel("Time (hour)")
        ax.set_ylabel("Iterations")

        ax.legend()
