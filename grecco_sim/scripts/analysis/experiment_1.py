import numpy as np
import yaml
from grecco_sim.graph.utils.format import Format
import pandas as pd

def main():
    result_dir = Format().output_root / "experiment_1"

    for run_dir in result_dir.glob("run_*"):
        result = dict()

        with open(run_dir / 'meta.yaml', 'r') as f:
            meta = yaml.load(f, Loader=yaml.SafeLoader)

        result.update(meta)

        trafo_ts = pd.read_csv(run_dir / "p_trafo.csv", index_col=0)

        result["total_load"] = trafo_ts.sum()
        result["total_absolute_load"] = trafo_ts.abs().sum()

        transformer_limit = meta["transformer_lim"] * 1000
        congestions = trafo_ts[trafo_ts > transformer_limit].dropna()

        n_congestion_events = len(congestions)
        result["number_of_congestion_events"] = n_congestion_events

        avg_congestion_size = np.mean(congestions / transformer_limit)
        result["average_congestion_size"] = avg_congestion_size

        result["congestion_peak"] = np.max(congestions / transformer_limit)

        print(result)

if __name__ == "__main__":
    main()