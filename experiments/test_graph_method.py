from experiments.series import series_0
from thesis.graph import complex_network_analysis

from thesis.graph.utils import config, network


def main():
    # Run experiment.
    experiment = series_0.experiment_1()
    experiment.run()
    for ts_data in experiment.ts_data:
        n = ts_data.n
        # Log congested grid parts.
        # ToDo: Move code part.
        loading = network.get_p_transmission_mw(n)
        capacity = network.get_p_capacity_mw(n)
        sim_congestion_table = loading > capacity

        # Compare congested grid parts and critical graph parts.
        print("Simulation")
        print(sim_congestion_table.sum())

    # Analyze graph structure.
    pr_config = config.PushRelabelConfiguration()
    cna = complex_network_analysis.ComplexNetworkAnalysis(n, pr_config)
    graph_congestion_table = cna.run()
    print("Graph")
    print(graph_congestion_table.sum())

if __name__ == "__main__":
    main()
