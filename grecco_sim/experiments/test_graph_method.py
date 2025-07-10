from grecco_sim.experiments.series import series_0
from grecco_sim.graph import complex_network_analysis

from grecco_sim.graph.utils import network
from grecco_sim.graph.utils import config


def main():
    # Run experiment.
    experiment = series_0.experiment_1()
    experiment.run()

    for ts_data in experiment.ts_data:
        n = ts_data.n
        print(n.loads_t["p"].sum().sum())
        # Log congested grid parts.
        # ToDo: Move code part.
        loading = network.get_p_transmission_mw(n)
        capacity = network.get_p_capacity_mw(n)
        sim_congestion_table = loading > capacity

        # Compare congested grid parts and critical graph parts.
        print(f"Simulation")
        print(sim_congestion_table.sum())

    # Analyze graph structure.
    pr_config = config.PushRelabelConfiguration()
    cna = complex_network_analysis.ComplexNetworkAnalysis(n, pr_config)
    graph_congestion_table = cna.run()
    print("Graph")
    print(graph_congestion_table.sum())

if __name__ == "__main__":
    main()
