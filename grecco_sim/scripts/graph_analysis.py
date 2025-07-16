import warnings
from pathlib import Path

import pypsa

from grecco_sim.graph import complex_network_analysis
from grecco_sim.graph.utils import config

import datetime

from grecco_sim.graph.utils.format import Format

import pandas as pd
from warnings import simplefilter


simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")


if __name__ == "__main__":
    runs_dir = Format().output_root
    n_name = "run_2_osqp_simbench-LV-rural2--2_cubic_02_27_17"
    n_path = runs_dir / "experiment_3" / n_name / "network"

    with warnings.catch_warnings():
        n = pypsa.Network()
        warnings.simplefilter("ignore", category=UserWarning)
        n.import_from_csv_folder(n_path)

    n.set_snapshots(n.snapshots[0:16])

    # Analyze graph structure.
    pr_config = config.PushRelabelConfiguration(n.snapshots, max_runtime=60*40)
    cna = complex_network_analysis.ComplexNetworkAnalysis(n, pr_config)
    graph_congestion_table = cna.run()
    grecco_sim = Path("/home/carl-wanninger/thesis/grecco_sim")
    result_path = grecco_sim / "results" / "tables" / "congestion"
    if not result_path.exists():
        result_path.mkdir(parents=True)
    graph_congestion_table.to_csv(result_path / "test.csv")
    print("Congestion table")
    print(graph_congestion_table.sum()[graph_congestion_table.sum() > 0])
    print(graph_congestion_table.values.sum(axis=None))