from dataclasses import dataclass
import dataclasses
import os
from typing import Any
import numpy as np
import pandas as pd

from grecco_sim.models import battery
from grecco_sim.util import sig_types
from grecco_sim.simulator import results
from grecco_sim.util import type_defs
from grecco_sim.util import style


def _get_aggregated_ts_result(sim_results: results.SimulationResult):
    """
    Calculate the KPIs in DOCUMENTED_KPIS from the aggregated grid power profile:
    """
    # Define a set to keep track of the returned elements of the dict.
    DOCUMENTED_KPIS = {
        "dt_h",  # time step of profile in hours
        "max_load",  # maximum load (positive) of aggregated profile
        "max_feed",  # minimum of aggregated profile (feed-back)
        "agg_consumption",
        "agg_feed_back",
        "agg_grid_fees"
    }
    
    res_load = sim_results.ts_grid.sum(axis=1)
    dt_h = (res_load.index[1] - res_load.index[0]).total_seconds() / 3600.

    res = {}
    
    res["dt_h"] = dt_h

    res["max_load"] = max(0, res_load.max())
    res["max_feed"] = max(0, -res_load.min())

    res["agg_consumption"] = res_load.loc[res_load > 0].sum() * dt_h
    res["agg_feed_back"] = -res_load.loc[res_load < 0].sum() * dt_h

    # Signals
    sys_ids = sim_results.sys_ids
    res["agg_grid_fees"] = sim_results.assigned_grid_fees.loc[:, [f"fee_{sys_id}" for sys_id in sys_ids]].sum().sum()
    

    windows = pd.DataFrame(columns=["time", "max_load", "max_feed"])
    for i, window in enumerate(res_load.rolling(10)):
        windows.loc[i, :] = [window.index[0], window.max(), -window.min()]


    assert set(res.keys()) == DOCUMENTED_KPIS, "Make sure that the return is correctly documented!"

    return res

# TODO bring the agent analysis result to the set approach from above. It seems more straightforward.
@dataclass
class AgentAnalysisResult:
    # Combined individual costs (owed to utility, no coordination penalties)
    costs_all: float
    # time series of summed energy in storages
    en_in_bat_ts: np.ndarray
    # Net charged energy over sim horizon
    charged_energy: float
    # Combined capacity of all individual storages
    combined_capacity: float
    # Summed up losses through battery operation
    losses_bat: float
    # feed in aggregated over all agents
    agg_feed: float
    # feed in from battery aggregated over all agents
    agg_bat_to_grid: float


def agent_analysis(agent_ts: dict[str, pd.DataFrame], sizing: dict[str, type_defs.SysParsPVBat], dt_h: float):
    """Do some analysis of time series of individual agents."""
    costs_all = 0
    en_in_bat_ts = None

    combined_cap = 0.
    en_loss = 0.

    agg_feed = 0.
    agg_bat_to_grid = 0.

    for sys_id, df in agent_ts.items():

        pv = df["p_el_pv"] if "p_el_pv" in df.columns else pd.Series(index=df.index, data=np.zeros(len(df.index)))

        grid = df["grid"]
        supp = grid[grid>0]
        feed = grid[grid<0]

        costs_supp = (supp * df["c_supp"][grid>0]).sum() * dt_h
        costs_feed = (feed * df["c_feed"][grid<0]).sum() * dt_h

        costs_all += costs_supp + costs_feed

        bat_to_grid = grid + pv
        bat_to_grid = bat_to_grid[bat_to_grid < 0]
        agg_bat_to_grid += bat_to_grid.sum() * dt_h
        agg_feed += feed.sum() * dt_h

        if isinstance(sizing[sys_id], type_defs.SysParsPVBat):
            if en_in_bat_ts is None:
                en_in_bat_ts = df["bat_soc"].values * sizing[sys_id].capacity
            else:
                en_in_bat_ts += df["bat_soc"].values * sizing[sys_id].capacity

            combined_cap += sizing[sys_id].capacity

            en_loss += (df["bat_p_ac"] - df["bat_p_net"]).sum() * sizing[sys_id].dt_h
        
        charged_energy = 0. if en_in_bat_ts is None else en_in_bat_ts[-1] - en_in_bat_ts[0]

    return AgentAnalysisResult(
        costs_all,
        en_in_bat_ts,
        charged_energy,
        combined_cap,
        en_loss,
        agg_feed,
        agg_bat_to_grid
    )


def signal_analysis(
        run_params: type_defs.RunParameters,
        raw_output: dict[str, dict]
    ):

    if not run_params.plot:
        return

    if run_params.coordination_mechanism not in ["admm", "second_order"]:
        print(f"No analysis to be made for '{run_params.coordination_mechanism}' coordination mechanism.")
        return

    fig_sig, ax_sig = style.styled_plot(title="Signal over time", ylabel="Signal", figsize=(16, 8))
    fig_ref, ax_ref = style.styled_plot(title="Reference power over time", ylabel="Power / kW", figsize=(16, 8))

    for k, signal in list(raw_output.values())[0]["signals"].items():
        # Show change of signal over simulation horizon for one agent.
        if not isinstance(signal, sig_types.SecondOrderSignal):
            print(f"Signal not Second order signal but {type(signal)}")
            return

        sig = ax_sig.plot(np.arange(start=k, stop=k+signal.signal_len), signal.mul_lambda, label=f"k = {k}", drawstyle="steps-post")
        ax_sig.plot(k, signal.mul_lambda[0], marker="s", color=sig[0].get_color(), label=None)
        ref = ax_ref.plot(np.arange(start=k, stop=k+signal.signal_len), signal.res_power_set, label=f"k = {k}",drawstyle="steps-post")
        ax_ref.plot(k, signal.res_power_set[0], marker="s", color=ref[0].get_color(), label=None)

    ax_sig.legend()
    ax_ref.legend()

    fig_sig.tight_layout()
    fig_ref.tight_layout()


# ================== Write results to file =============================



def _write_to_files(
        run_params: type_defs.RunParameters,
        eval_res: dict[str, Any],
    ):

    # Abbreviate
    out_dir = run_params.output_file_dir

    # check that output directory extists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    eval_df = pd.DataFrame(index={"tag":[run_params.sim_tag]}, data={tag: [eval_res[tag]] for tag in eval_res}).set_index("tag")
    
    eval_file_path = out_dir / "kpis.csv"
    if os.path.exists(eval_file_path):
        eval_df = pd.concat([pd.read_csv(eval_file_path, index_col="tag"), eval_df])
                            
    eval_df.to_csv(eval_file_path)


def evaluate_sim(sim_result: results.SimulationResult):
    # Evaluate Simulation Results ==========================================
    
    eval_res = _get_aggregated_ts_result(sim_result)
    eval_res["tag"] = sim_result.run_params.sim_tag

    eval_res.update(dataclasses.asdict(agent_analysis(sim_result.agents_ts, sim_result.sizing, eval_res["dt_h"])))
    eval_res.pop("en_in_bat_ts")

    eval_res["calc_time"] = sim_result.exec_time

    print(f"Analysis result for sim {sim_result.run_params.sim_tag}: {eval_res}")

    _write_to_files(sim_result.run_params, eval_res)

    # Reactivate if needed. However, sim_result must get a signals field
    # signal_analysis(sim_result.run_params, sim_result.agents_ts)

    return eval_res


if __name__ == "__main__":
    pass
