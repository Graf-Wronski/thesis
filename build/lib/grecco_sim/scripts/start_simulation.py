import pathlib
import datetime
import pytz
import matplotlib.pyplot as plt

from grecco_sim.simulator import simulation_setup
from grecco_sim.util import type_defs

DUMMY_SCENARIO = {"name": "dummy_data", "n_agents": 4}
SAMPLE_SCENARIO = {"name": "sample_scenario", "focus": "pv_bat", "n_agents": 1}
HP_SCENARIO = {"name": "sample_scenario", "focus": "hp", "n_agents":1}
OPFINGEN = {
    "name": "opfingen",
    "n_agents": 4,
    #     # Create symlink into data directory!
    "grid_data_path": pathlib.Path(__file__).parent.parent.parent
    / "data"
    / "Opfingen_Profiles_2023"
    /
    # "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all_ts_reduced",
    "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all",
    "weather_data_path": pathlib.Path(__file__).parent.parent.parent
    / "data"
    / "Opfingen_Profiles_2023"
    / "weather_data.csv",
    "hp": False
}


def main():

    # coord_type = "central_optimization"
    # coord_type = "second_order"
    # coord_type = "admm"
    coord_type = "plain_grid_fee"
    # coord_type = "local_self_suff"

    run_parameters = simulation_setup.RunParameters(
        sim_horizon=72,
        # Summer: peak feed in
        # start_time = datetime.datetime(2023, 7, 11, 0, 0, 0, tzinfo=pytz.utc),
        start_time = datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=pytz.utc),
        max_market_iterations=4,
        coordination_mechanism=coord_type,
        # scenario=SAMPLE_SCENARIO,
        scenario=OPFINGEN,
        sim_tag=f"{coord_type}",
        # inspection=[36],
        use_prev_signals=False,
        plot=True,
        show=True,
        profile_run=True,
        output_file_dir=pathlib.Path("default") / datetime.datetime.now().strftime("%y%m%d_%H%M")
        # output_file_dir=pathlib.Path("whole_year")
    )

    opt_pars = type_defs.OptParameters(
        rho=100.0,
        mu=5000.0,
        horizon=40,
        alpha=0.05,
        solver_name="gurobi",
        fc_type="perfect",
    )

    grid_pars = type_defs.GridDescription(
        p_lim=75.0,  # Algorithm parameters
    )


    simulator = simulation_setup.SimulationSetup(run_parameters)
    simulator.run_sim(opt_pars, grid_pars)

    plt.show()


if __name__ == "__main__":
    main()
