from experiments.series import series_0
from thesis.graph import complex_network_analysis
from thesis.graph.utils import config


def main():
    # Run experiment.
    experiment = series_0.experiment_0()
    experiment.run()

    # Log congested grid parts.
    congestion_table = ...

    # Analyze graph structure.
    network = experiment.ts_data[-1].n
    pr_config = config.PushRelabelConfiguration()
    cna = complex_network_analysis.ComplexNetworkAnalysis(network, pr_config)
    cna.run()

    # Compare congested grid parts and critical graph parts.
    ...

if __name__ == "__main__":
    main()
