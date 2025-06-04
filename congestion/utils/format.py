from pathlib import Path

import numpy as np
import pandas as pd


class Format:
    def __init__(self):
        """ Defines globally used constants, formats and transformations. """

        # The format for timestamps, e.g. snapshots.
        self.date = '%Y-%m-%d %H:%M'

        # Transformations between different watt units.
        self.mw_to_kw = lambda x: np.round(1_000 * x)
        self.mw_to_w = lambda x: np.round(1_000_000 * x)
        self.kw_to_mw = lambda x: x / 1000
        self.w_to_mw = lambda x: x / 1_000_000

        # The time difference between two snapshots.
        self.timestep = pd.Timedelta(minutes=15)

        # The main directory for all sorts of data.
        self.data_root = Path("/home/carl-wanninger/data")
