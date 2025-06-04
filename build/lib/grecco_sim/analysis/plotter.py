from curses import meta
from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from grecco_sim.simulator import results
from grecco_sim.util import style, type_defs


@dataclass
class PlottingMetaInf():
    """Dataclass for meta information needed for plotting."""
    n_agents: int
    sys_ids: List[str]
    sim_tag: str
    color_list = [xkcd_color[1] for xkcd_color in list(mcolors.XKCD_COLORS.items())]


def make_plots(
        sim_result: results.SimulationResult
):
    """Make plots for visual analysis."""
    if not sim_result.run_params.plot:
        # Nothing to do
        return

    meta_inf = PlottingMetaInf(
        sys_ids = list(sim_result.ts_grid.columns.values),
        n_agents = len(sim_result.ts_grid.columns),
        sim_tag=sim_result.run_params.sim_tag,
    )
    # Plot grid everytime 
    plot_grid(sim_result.ts_grid, meta_inf, sim_result.run_params)

    if sim_result.run_params.show:
        # Plot these 'analysis' plots only when showing
        plot_batteries(sim_result.flex_ts, meta_inf)
        plot_hps(sim_result.flex_ts, sim_result.agents_ts, meta_inf)
        plot_signals(sim_result.assigned_grid_fees, meta_inf)
        plt.show()


def plot_grid(ts_grid: pd.DataFrame, meta_inf: PlottingMetaInf, run_params: type_defs.RunParameters):
    """Plot the grid time series with combined and individual agents."""
    fig, ax = style.styled_plot(
        xlabel="Time", ylabel="Load at transformer / kW", figsize="landscape",
        # ylim=(-1.2 * meta_inf.n_agents, 1.2 * meta_inf.n_agents)
        ylim=(0., 100.)
    )

    ts_grid["_prev_sys"] = 0

    for i, sys_id in enumerate(meta_inf.sys_ids):
        ax.fill_between(ts_grid.index, ts_grid["_prev_sys"],
                        ts_grid[sys_id] + ts_grid["_prev_sys"],
                        color=meta_inf.color_list[i], alpha=0.2, step="post")
        ax.plot(ts_grid.index, ts_grid[sys_id] + ts_grid["_prev_sys"],
                label=None, drawstyle="steps-post", color=meta_inf.color_list[i])
        ts_grid["_prev_sys"] += ts_grid[sys_id]

    ax.plot(ts_grid["_prev_sys"], label="Sum", linestyle="dashed", color="black", drawstyle="steps-post")

    # ax.legend(title=meta_inf.sim_tag)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_params.output_file_dir / f"plot_grid_{run_params.sim_tag}.pdf")


def plot_hps(ts_hp: pd.DataFrame, agents_ts: dict[str, pd.DataFrame], meta_inf:PlottingMetaInf):

    fig, ax = style.styled_plot(xlabel="Time", ylabel="Power Heat Pumps", figsize="landscape", title="Heat Pumps")

    sum_hp = np.zeros(len(ts_hp.index))
    _label = "Agents"
    for i, col_name in enumerate(ts_hp.columns.values):
        if "hp_p_in" in col_name:
            ax.plot(
                ts_hp[col_name],
                label=_label,
                drawstyle="steps-post",
                color=meta_inf.color_list[i],
                linestyle="dashed",
            )
            sum_hp += ts_hp[col_name].values
            if _label == "Agents":
                _label = None
    
    ax.plot(ts_hp.index, sum_hp, label="All batteries", drawstyle="steps-post", color="black")
    ax.legend()
    fig.tight_layout()

    fig_temp, ax_temp = style.styled_plot(xlabel="Time", ylabel="Temperature / C", figsize="landscape")
    for ag_name, ts_ag in agents_ts.items():
        if "hp_temp" in ts_ag:
            ax_temp.plot(
                ts_ag["hp_temp"],
                label=ag_name,
                drawstyle="steps-post"
            )
    fig_temp.legend()
    fig_temp.tight_layout()




def plot_batteries(ts_bat: pd.DataFrame, meta_inf:PlottingMetaInf):

    fig, ax = style.styled_plot(xlabel="Time", ylabel="Power", figsize="landscape", title="Battery Power")

    sum_bat = np.zeros(len(ts_bat.index))
    _label = "Agents"
    for i, col_name in enumerate(ts_bat.columns.values):
        if "bat_p_ac" in col_name:
            ax.plot(
                ts_bat.index,
                ts_bat[col_name],
                label=_label,
                drawstyle="steps-post",
                color=meta_inf.color_list[i],
                linestyle="dashed"
            )
            sum_bat += ts_bat[col_name].values
            if _label == "Agents":
                _label = None

    ax.plot(ts_bat.index, sum_bat, label="All batteries", drawstyle="steps-post", color="black")

    ax.legend()
    fig.tight_layout()


def plot_signals(ts_signals: pd.DataFrame, meta_inf: PlottingMetaInf):
    fig, ax = style.styled_plot(xlabel="Time", ylabel="Costs / Euro", figsize="landscape", title="Signal incurred costs")

    for i, sys_id in enumerate(meta_inf.sys_ids):
        ax.plot(ts_signals.index, ts_signals[f"fee_{sys_id}"], label=sys_id, drawstyle="steps-post", color=meta_inf.color_list[i])

    if len(ts_signals) < 13:
        ax.legend()

    fig.tight_layout()
