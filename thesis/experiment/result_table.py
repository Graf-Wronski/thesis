import pickle
from typing import Any

import pandas as pd
from pathlib import Path


def load_df_from_pickle(path: Path) -> pd.DataFrame:
    """ Build saved object from pickle file."""
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def write_df_to_pickle(data: pd.DataFrame, path: Path, parents: bool = True):
    """ Write Object to pickle file."""
    if (not path.parent.exists()) and parents:
        path.parent.mkdir(parents=True)

    with open(path, "wb") as file:
        pickle.dump(data, file)


class ResultTable:

    def __init__(self, path: Path, overwrite: bool = False):
        """ This class manages tabular (DataFrame) result data.

        Args:
        - path: The path where the result will be stored.
        - overwrite: If true, the previous results will be deleted. """

        self.path = path
        if not self.path.suffix == ".pkl":
            raise ValueError("Result table is saved as a pickle file.")

        # If table does not exist, create an empty one.
        if not (self.path.exists()) or overwrite:
            print(f"\nCreating table at:\n {self.path} .")
            table = pd.DataFrame()
            write_df_to_pickle(table, self.path)

        # Load data.
        self.data = load_df_from_pickle(self.path)

    def __len__(self) -> int:
        return len(self.data)

    def add_result(self, result: dict, index: Any = None):
        """ Add a new result line to table.

        Args:
        - result: The result data.
        - The index where the result should be stored. """

        # If index is not given: Add result at the end.
        if index is None:
            index = self.data.index.max() + 1

        # Load data and concat new data.
        self.data = load_df_from_pickle(self.path)
        new_line = pd.DataFrame(result, [index])
        self.data = pd.concat([self.data, new_line])
        write_df_to_pickle(self.data, self.path)

    def store_as_csv(self):
        """ Store data as csv file next to pkl file. """
        csv_path = Path(str(self.path).replace(".pkl", ".csv"))
        print(f"\nStoring table at:\n {csv_path} .")
        self.data.to_csv(csv_path)

