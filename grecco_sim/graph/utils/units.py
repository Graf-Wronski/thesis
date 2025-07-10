import pandas as pd


class CustomPyPSAObjects:
    def transformers(self) -> pd.DataFrame:
        mini_transformer = {
            'f_nom': 50.0,
            's_nom': 0.05,
            'v_nom_0': 20.0,
            'v_nom_1': 0.4,
            'vsc': 6.0,
            'vscr': 1.44,
            'pfe': 0.8,
            'i0': 0.32,
            'phase_shift': 150,
            'tap_side': 0,
            'tap_neutral': 0,
            'tap_min': -2,
            'tap_max': 2,
            'tap_step:': 2.5,
            'references': "PyPSA: 0.25 MVA 20/0.4 kV"
        }

        return pd.DataFrame(index=["0.05 MVA 20/0.4 kV"],
                            data=mini_transformer)