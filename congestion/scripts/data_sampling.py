from datetime import datetime
from pathlib import Path

from termcolor import colored

import logging

from congestion.data.data_sampler import DataSampler
from congestion.utils.config import SamplerConfiguration
from congestion.utils.network import utilization_ratio

# Mute PyPSA Info:
logging.getLogger("pypsa").setLevel(logging.WARNING)

def main(n_nodes: int = 10):
    data_root = Path("/home/carl-wanninger/data")
    simbench_path = data_root / "simbench_csv" / "all_profiles_2034"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_path = data_root / "pypsa" / "test" / timestamp
    if not result_path.exists():
        result_path.mkdir()
    sampler_config = SamplerConfiguration(
        s_nom=0.05,
        p_heatpump=0.6,
        p_pv=0.0,
        seed=17
    )
    sampler = DataSampler(simbench_path, sampler_config)
    samples = [sampler.sample(len_timeseries=60, n_nodes=n_nodes)
               for _ in range(25)]

    for grid_idx, sampled_grid in enumerate(samples):
        print(f"\nGrid: {grid_idx + 1} of {len(samples)}.")

        print("Calculating powerflow.")
        # ToDo: Transformer type could be added more elegantly.
        sampled_grid.lpf()
        sampled_grid.pf(use_seed=True)
        print("Done.")

        utilization = utilization_ratio(sampled_grid)

        if (utilization >= 100).any(axis=None):
            if (utilization["MV/LV Transformer"] >= 100).any(axis=None):
                print(colored("Transformer limit violation.", "red"))
                print(f"Max. {utilization['MV/LV Transformer'].max()} % "
                      f"Min. {utilization['MV/LV Transformer'].min()} %")
            else:
                print(colored("Line limit violation.", "red"))
        else:
            print(colored("No capacity violation.", "green"))

        grid_name = f"{n_nodes}_grid_{grid_idx + 1}"
        print(f"Exporting {result_path / grid_name}.")
        sampled_grid.export_to_csv_folder(result_path / grid_name)


if __name__ == "__main__":
    main(25)
