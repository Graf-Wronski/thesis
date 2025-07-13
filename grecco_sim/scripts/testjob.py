
import pandas as pd
from grecco_sim.graph.utils.format import Format


if __name__ == "__main__":
    data_root = Format().data_root
    pd.DataFrame().to_csv(data_root / "test_export.csv")
