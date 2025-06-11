from pathlib import Path

import pypsa
import pytest


@pytest.fixture
def import_path(data_root) -> Path:
    return (data_root / "pypsa" / "trafo_violation" / "26-02-2025" /
            "25_grid_23")


def test_transformer_import(import_path: Path, data_root: Path):
    """ Redefine 's_nom' of an existing transformer type. Assure that the
    transformer is exported and imported correctly. """

    s_nom = 0.65537  # Arbitrary value.
    export_path = data_root / "test" / "transformer_s_nom"

    # Import a network
    network = pypsa.Network()
    network.import_from_csv_folder(import_path)

    # Redefine Transformer
    network.transformers.at["MV/LV Transformer", "s_nom"] = s_nom
    network.transformers.at["MV/LV Transformer", "type"] = ""

    # Check S_nom.
    assert network.transformers.loc["MV/LV Transformer"]["s_nom"] == s_nom

    # Conduct Powerflow.
    network.lpf()
    network.pf(use_seed=True)

    # Check S_nom.
    assert network.transformers.loc["MV/LV Transformer"]["s_nom"] == s_nom

    # Export network.
    network.export_to_csv_folder(export_path)

    # Re-Import network.
    network = pypsa.Network()
    network.import_from_csv_folder(export_path)

    # Check S_nom.
    assert network.transformers.loc["MV/LV Transformer"]["s_nom"] == s_nom

    # Conduct Powerflow.
    network.lpf()
    network.pf(use_seed=True)

    # Check S_nom.
    assert network.transformers.loc["MV/LV Transformer"]["s_nom"] == s_nom
