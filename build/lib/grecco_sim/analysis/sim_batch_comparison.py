"""
Module with some analysis over different simulation runs.
"""
import matplotlib.pyplot as plt

from grecco_sim.util import type_defs
from grecco_sim.util import data_io
from grecco_sim.util import style


def compare(run_par_sets: list[type_defs.RunParameters], sim_tags: list[str]):
    """Plot the grid time series of a set of sim results for an arbitrary agent to comapre."""
    ts = {
        sim_tag: data_io.read_ts(
            run_params.output_file_dir / f"results_{sim_tag}.csv"
        )
        for run_params, sim_tag in zip(run_par_sets, sim_tags)
    }

    fig, ax = style.styled_plot()

    for sim_tag in sim_tags:
        ag_name = ts[sim_tag].columns.values[0]
        ax.plot(ts[sim_tag][ag_name], label=sim_tag)

    ax.legend()
    plt.show()
