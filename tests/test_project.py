"""Verification tests for food-safety-analytics deliverables and codebase."""

from __future__ import annotations

import json
import pathlib
import tomllib
import unittest
from typing import Any

import joblib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "food-safety-analytics.ipynb"
DATASET_PATH = ROOT / "dataset" / "sf_health_inspections.csv"
MODEL_PATH = ROOT / "model" / "food_safety_random_forest_model.pkl"
VIZ_DIR = ROOT / "visualizations"

EXPECTED_VISUALIZATIONS = (
    "outcome_and_violations.png",
    "monthly_compliance_trend.png",
    "neighborhood_risk_and_map.png",
    "top_calcode_citations.png",
    "model_evaluation.png",
    "sf_food_safety_map.html",
)


class ProjectConfigTests(unittest.TestCase):
    """Validate project, IDE, and Pyrefly configuration files."""

    def test_python_version_file(self) -> None:
        version_text = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
        self.assertEqual(version_text, "3.14.7")

    def test_toml_and_json_configs_parse(self) -> None:
        with (ROOT / "pyrefly.toml").open("rb") as fh:
            pyrefly_cfg = tomllib.load(fh)
        self.assertIn("**/*.ipynb", pyrefly_cfg.get("project-includes", []))

        with (ROOT / "pyproject.toml").open("rb") as fh:
            project_cfg = tomllib.load(fh)
        self.assertIn("3.14.7", project_cfg["project"]["requires-python"])

        for rel in (".vscode/settings.json", ".vscode/extensions.json"):
            data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
            self.assertIsInstance(data, dict)

    def test_notebook_cells_compile_and_have_no_errors(self) -> None:
        nb: dict[str, Any] = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(nb["metadata"]["language_info"]["version"], "3.14.7")
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreater(len(code_cells), 0)

        for idx, cell in enumerate(code_cells):
            source = "".join(cell["source"])
            compile(source, f"notebook_cell_{idx}", "exec")
            errors = [
                out
                for out in cell.get("outputs", [])
                if out.get("output_type") == "error"
            ]
            self.assertEqual(errors, [], f"Code cell {idx} contains error output")


class DeliverableTests(unittest.TestCase):
    """Validate dataset, serialized model artifact, and visualization outputs."""

    def test_dataset_schema_and_cleaning_invariants(self) -> None:
        self.assertTrue(DATASET_PATH.is_file())
        df = pd.read_csv(DATASET_PATH, low_memory=False)
        self.assertEqual(df.shape, (23243, 26))
        required_cols = {
            "permit_number",
            "dba",
            "permit_type",
            "inspection_date",
            "inspection_type",
            "inspection_frequency_type",
            "facility_rating_status",
            "violation_count",
            "violation_codes",
            "total_time",
            "analysis_neighborhood",
            "supervisor_district",
            "latitude",
            "longitude",
            "data_as_of",
        }
        self.assertTrue(required_cols.issubset(set(df.columns)))

    def test_serialized_model_loads_and_predicts(self) -> None:
        self.assertTrue(MODEL_PATH.is_file())
        artifact: dict[str, Any] = joblib.load(MODEL_PATH)
        self.assertEqual(
            set(artifact.keys()),
            {"model", "numeric_features", "categorical_features", "threshold"},
        )
        sample = pd.DataFrame(
            [
                {
                    "prior_inspections": 2,
                    "prior_non_pass_count": 1,
                    "prior_avg_violations": 4.5,
                    "days_since_last_inspection": 120.0,
                    "month": 11,
                    "day_of_week": 2,
                    "lat_clean": 37.7749,
                    "lon_clean": -122.4194,
                    "permit_type": "H25",
                    "inspection_type": "Routine",
                    "inspection_frequency_type": "Annual",
                    "analysis_neighborhood": "Tenderloin",
                    "supervisor_district": "6.0",
                }
            ]
        )
        probs = artifact["model"].predict_proba(sample)
        self.assertEqual(probs.shape, (1, 2))
        self.assertGreaterEqual(float(probs[0, 1]), 0.0)
        self.assertLessEqual(float(probs[0, 1]), 1.0)

    def test_visualizations_exist_and_non_empty(self) -> None:
        for name in EXPECTED_VISUALIZATIONS:
            path = VIZ_DIR / name
            self.assertTrue(path.is_file(), f"Missing visualization: {name}")
            self.assertGreater(path.stat().st_size, 1024, f"Empty visualization: {name}")
        html = (VIZ_DIR / "sf_food_safety_map.html").read_text(encoding="utf-8")
        self.assertIn("leaflet", html.lower())
        self.assertIn("server.arcgisonline.com", html)
        self.assertNotIn("tile.openstreetmap.org", html)
        self.assertNotIn("basemaps.cartocdn.com", html)


if __name__ == "__main__":
    unittest.main()
