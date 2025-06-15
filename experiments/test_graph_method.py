from experiments.series import series_0
from thesis.graph import complex_network_analysis

from thesis.graph.utils import config, network


def main():
    # Run experiment.
    experiment = series_0.experiment_0()
    experiment.run()
    n = experiment.ts_data[0].n

    # Log congested grid parts.
    # ToDo: Move code part.
    loading = network.get_loading_mw(n)
    capacity = network.get_p_capacity_mw(n)
    sim_congestion_table = loading > capacity

    # Analyze graph structure.
    pr_config = config.PushRelabelConfiguration()
    cna = complex_network_analysis.ComplexNetworkAnalysis(n, pr_config)
    graph_congestion_table = cna.run()

    print(sim_congestion_table.sum())
    print(graph_congestion_table.sum())

    # Compare congested grid parts and critical graph parts.
    ...

if __name__ == "__main__":
    main()
