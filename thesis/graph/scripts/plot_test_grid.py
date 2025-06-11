from typing import Tuple

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from thesis.test_data.four_node_lv_grid import get_four_node_lv_grid
import seaborn as sns


def main():
    """ For each bus, the household load, the pv generation, the storage and
    the heatpump load and the ev load are plotted. """

    # Load network.
    four_node_grid = get_four_node_lv_grid()
    # Create a dataframe containing all data.

    load_data = four_node_grid.loads_t.log_p_grid
    generation_data = four_node_grid.generators_t.log_p_grid
    storage_data = four_node_grid.storage_units_t.log_p_grid

    assert (load_data.index == generation_data.index).all()
    assert (load_data.index == storage_data.index).all()

    data = pd.concat([load_data, generation_data, storage_data], axis=1)
    data["Time"] = pd.to_datetime(data.index, format="%d.%m.%Y %H:%M")
    print(data["Time"])
    data["Timestep"] = (data["Time"] - data["Time"].min()).dt.total_seconds()
    data["Timestep"] = data["Timestep"].transform(lambda x: int(x / 900))
    data = data.melt(id_vars=["Time", "Timestep"], var_name="Type", value_name="Value (MW)")

    def get_unit_type_and_bus_name(load_name: str) -> Tuple[str, str]:
        if (load_name[-1]).isnumeric():
            return load_name[:-1], load_name[-1]
        else:
            return load_name, "Slack"

    data = data[data["Timestep"] < 96]
    data[['Unit Type', 'Bus']] = data['Type'].apply(
        lambda x: pd.Series(get_unit_type_and_bus_name(x)))

    sns.relplot(data=data, row="Bus", x="Timestep", y="Value (MW)", hue="Unit Type", alpha=0.4, kind="line")
    plt.show()

    pass

    """sns.relplot(
        data=dots,
        x="time", y="firing_rate",
        hue="coherence", size="choice", col="align",
        kind="line", size_order=["T1", "T2"], palette=palette,
        height=5, aspect=.75, facet_kws=dict(sharex=False),
    )"""

    # Plot generation.

    # Plot storage units.
    loaded_storage_data = [
        x for x in four_node_grid.storage_units_t.keys() if
        len(four_node_grid.storage_units_t[x].columns) > 0]



if __name__ == "__main__":
    main()