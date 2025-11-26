import pandas as pd
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
from grecco_sim.graph.utils.format import Format


if __name__ == "__main__":
    data_root = Format().data_root
    pd.DataFrame().to_csv(data_root / "test_export.csv")
