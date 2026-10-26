import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "meridian_pathway_eda.ipynb"


def _run_pathway_calculation():
    notebook = json.loads(NOTEBOOK.read_text())
    marker = "# Part H: Patient pathway timing"
    start = next(i for i, cell in enumerate(notebook["cells"]) if marker in "".join(cell["source"]))
    code = next(cell for cell in notebook["cells"][start + 1 :] if cell["cell_type"] == "code")
    namespace = {"pd": pd, "np": np, "display": lambda *args, **kwargs: None}
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(NOTEBOOK.parent)
        exec("".join(code["source"]), namespace)
    finally:
        os.chdir(old_cwd)
    return namespace


def test_pathway_maps_every_curated_code_and_builds_pairwise_matrices():
    ns = _run_pathway_calculation()

    expected_stages = [
        "Earliest record",
        "Imaging",
        "Referred",
        "Injections",
        "See surgeon",
        "Surgery",
    ]
    encounters = pd.read_csv(ROOT / "data" / "meridian_health_encounters.csv", dtype=str)
    expected_codes = set(encounters["event_code"].dropna())

    assert ns["STAGES"] == expected_stages
    assert set(ns["code_taxonomy"]["code"].dropna()) == expected_codes
    assert ns["pathway_events"]["event_status"].eq("Performed").all()

    for name in ("mean_days", "median_days", "pair_counts"):
        matrix = ns[name]
        assert matrix.index.tolist() == expected_stages
        assert matrix.columns.tolist() == expected_stages
        assert matrix.shape == (6, 6)

    assert np.allclose(ns["mean_days"], -ns["mean_days"].T, equal_nan=True)
    assert np.allclose(ns["median_days"], -ns["median_days"].T, equal_nan=True)
    assert (np.diag(ns["pair_counts"]) > 0).all()
