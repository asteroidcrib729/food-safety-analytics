# Food Safety Analysis - San Francisco Restaurants

Exploratory data analysis, data quality auditing, predictive non-pass risk
modeling (`RandomForestClassifier`), and interactive geospatial visualization
of San Francisco food facility health inspections using the official DataSF
Health Inspections (2023-Present) dataset (`tvy3-wexg`).

## Dataset

- **File:** `dataset/sf_health_inspections.csv` (`23,243` rows, `26` columns,
  ~22.8 MB)
- **Source:** [DataSF Health Inspections (2023-Present)](https://data.sfgov.org/d/tvy3-wexg)
  (San Francisco Department of Public Health, `data_as_of` through
  `2026-11-21`)
- **Direct CSV Endpoint:**
  `https://data.sfgov.org/api/views/tvy3-wexg/rows.csv?accessType=DOWNLOAD`

### Key Fields

- **Facility Metadata:** `permit_number`, `dba`, `permit_type`,
  `street_address_clean`, `analysis_neighborhood`, `supervisor_district`,
  `latitude`, `longitude`
- **Inspection Details:** `inspection_date`, `inspection_type`,
  `inspection_frequency_type`, `total_time`, `inspector`
- **Outcomes & Violations:** `facility_rating_status` (`Pass`,
  `Conditional Pass`, `Closure`), `violation_count`, `violation_codes`,
  `suspension_notes`, `inspection_notes`

## Project Structure

- `food-safety-analytics.ipynb` - Unified end-to-end notebook covering:
  1. **Data Loading & Schema Profiling** - Missingness audit and feature
     overview
  2. **Data Cleaning & Anomaly Resolution** - Filtering future-date outliers
     (`2031-05-16`), nullifying negative/extreme `total_time` durations,
     imputing missing `violation_count`, flagging restaurant permits, and
     validating SF coordinates
  3. **Exploratory Data Analysis** - Rating distributions across restaurant
     size tiers (`H24`, `H25`, `H26`), violation counts by outcome/inspection
     type, and monthly compliance trends
  4. **Neighborhood & Geospatial Risk Analysis** - Non-pass rates
     (`Conditional Pass` + `Closure`) by neighborhood and SF spatial
     distribution
  5. **Violation Code Mining & Reinspection Turnaround** - Regex extraction of
     CalCode sections (`11xxxx`) and repeat-offender / reinspection intervals
  6. **Predictive Non-Pass Risk Modeling (`RandomForestClassifier`)** -
     Leak-free prior-inspection features, chronological `80/20` split,
     ROC/confusion-matrix/feature-importance diagnostics, and model
     serialization to `model/food_safety_random_forest_model.pkl`
  7. **Interactive Geospatial Risk Map (`folium`)** - Maps recent SF facility
     compliance outcomes and predicted risk probabilities to
     `visualizations/sf_food_safety_map.html`
  8. **Key Findings** - Summary of food safety compliance and operational
     risk patterns
- `dataset/`
  - `sf_health_inspections.csv` - Unified inspection dataset
- `model/`
  - `food_safety_random_forest_model.pkl` - Serialized
    `RandomForestClassifier` pipeline (`model`, `numeric_features`,
    `categorical_features`, `threshold`)
- `visualizations/`
  - Exported analysis charts (`outcome_and_violations.png`,
    `monthly_compliance_trend.png`, `neighborhood_risk_and_map.png`,
    `top_calcode_citations.png`, `model_evaluation.png`) and interactive map
    (`sf_food_safety_map.html`)
- `tests/`
  - `test_project.py` - Automated validation tests for configuration, notebook
    compilation/outputs, dataset invariants, serialized model inference, and
    visualization artifacts
- `.vscode/`
  - `settings.json` - Dedicated VS Code / Antigravity workspace configuration
    for global Python 3.14.7, Jupyter kernel filtering, Ruff, and Pyrefly
  - `extensions.json` - Recommended editor extensions
- `.github/workflows/ci.yml` - GitHub Actions workflow running Pyrefly checks
  and unit tests on Python 3.14
- `.python-version`, `pyproject.toml`, `pyrefly.toml`, `requirements.txt` -
  Python version pinning, project metadata, Pyrefly configuration, and frozen
  dependencies with flexible version ranges

## Complete Setup & Configuration Guide

### 1. Python 3.14.7 Configuration

This project targets **Python 3.14.7** and is pre-configured to recognize and
connect directly to the global Python 3.14.7 distribution:

- **`.python-version`**: Pins `3.14.7` so Python environment managers and IDEs
  automatically identify the target interpreter version.
- **`pyproject.toml`**: Declares `requires-python = ">=3.14.7,<3.15"`, core
  runtime dependencies, and typing stub dependencies (`pandas-stubs`,
  `types-seaborn`).
- **`requirements.txt`**: Contains the frozen project dependency set (core
  libraries, Jupyter execution packages, typing stubs, and transitive
  dependencies) using flexible version bounds (e.g., `>=2.5.5,<3.0.0` for
  SemVer packages and `>=` lower bounds for CalVer packages) rather than rigid
  `==` pins.

Verify your active Python installation before installing dependencies:

```bash
python --version
# Expected: Python 3.14.7
```

> **Windows Note:** If multiple Python versions are installed, verify that
> `py -0p` lists `3.14` as active or invoke `py -3.14 -m pip ...` explicitly.

Install all required packages into the global Python 3.14.7 environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer an isolated virtual environment instead of installing globally:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux:      source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Pyrefly Configuration (`pyrefly.toml`)

Pyrefly is configured in `pyrefly.toml` to perform syntax, type, and dependency
(`missing-import`) checks across both Python files and Jupyter notebooks:

- `project-includes = ["**/*.py*", "**/*.ipynb"]`
- `project-excludes = ["**/.venv", "**/__pycache__", "**/.ipynb_checkpoints"]`
- `python-version = "3.14"`
- `python-interpreter-path` & `site-package-path`: Pre-pointed to the global
  Python 3.14.7 executable and both system and user `site-packages` directories
  (`C:/Users/DESKTOP-Q2TMP8U/AppData/Local/Python/pythoncore-3.14-64/python.exe`).

If you are running on a different machine or OS, update
`python-interpreter-path` and `site-package-path` in `pyrefly.toml` to match
your local Python 3.14.7 paths (or pass them via CLI flags):

```bash
# Optional: install Pyrefly CLI if running outside VS Code
python -m pip install pyrefly

# Run Pyrefly syntax, dependency, and type checks
pyrefly check

# Inspect resolved interpreter and site-packages paths
pyrefly dump-config
```

### 3. Dedicated IDE Configuration (`.vscode/`)

Opening this folder in **VS Code** or **Antigravity** automatically applies
the workspace settings in `.vscode/settings.json` and prompts for the
recommended extensions in `.vscode/extensions.json`:

- **Recommended Extensions:**
  - `ms-python.python` — Python support
  - `ms-python.vscode-python-envs` — Python Environments manager
  - `meta.pyrefly` — Pyrefly language server (syntax, import/dependency, and type checking)
  - `ms-toolsai.jupyter` — Jupyter notebook runner and renderer
  - `charliermarsh.ruff` — Ruff linting and formatting
- **Key Workspace Settings (`.vscode/settings.json`):**
  - `python.defaultInterpreterPath` & `ruff.interpreter`: Point to the global
    Python 3.14.7 executable (`C:\Users\DESKTOP-Q2TMP8U\AppData\Local\Python\pythoncore-3.14-64\python.exe`).
    Update these paths if your Python 3.14.7 installation is located elsewhere.
  - `python-envs.defaultEnvManager`: Set to `ms-python.python:system` with
    `ms-python.python:pip` and `python.globalModuleInstallation: true` so the
    IDE connects directly to the global Python 3.14.7 environment instead of
    creating or prompting for a `.venv`.
  - `python.languageServer`: Set to `"None"` so Pylance does not duplicate or
    conflict with Pyrefly diagnostics.
  - `pyrefly.pyreflyExecutable`: Set to `"bundled"` (uses the extension's
    built-in binary), with `python.pyrefly.diagnosticMode: "workspace"`,
    `python.pyrefly.typeCheckingMode: "default"`, and
    `python.pyrefly.syncNotebooks: true` enabled for live `.ipynb` and `.py`
    syntax and dependency checks.
  - `jupyter.excludeUserSitePackages: false` &
    `jupyter.kernels.excludePythonEnvironments`: Ensures Jupyter loads user-site
    packages and filters out unrelated local Python 3.12 / Anaconda / `.uv-cache`
    interpreters.

## First-Time Run & Verification Checklist

1. **Confirm Python 3.14.7 & Dependencies:**

   ```bash
   python --version
   python -m pip install -r requirements.txt
   ```

2. **Verify Project Integrity & Deliverables:**

   ```bash
   python -m unittest discover -s tests -v
   ```

3. **Run Pyrefly Syntax & Dependency Checks:**

   ```bash
   pyrefly check
   ```

4. **Open & Run the Notebook:**
   - Open `food-safety-analytics.ipynb` in VS Code / Antigravity (select the
     **Python 3.14.7** / `Python 3` kernel in the top-right kernel picker,
     powered by `ipykernel` and `ms-toolsai.jupyter`) and click **Run All**.
   - Alternatively, if you prefer launching the standalone browser UI outside
     VS Code, install `notebook` (not bundled in `requirements.txt`) and start
     it:

     ```bash
     python -m pip install notebook
     python -m notebook food-safety-analytics.ipynb
     ```

   - Executing all cells regenerates `model/food_safety_random_forest_model.pkl`,
     the 5 PNG charts in `visualizations/`, and
     `visualizations/sf_food_safety_map.html`.

### Troubleshooting First-Time Setup

- **Missing imports in Pyrefly or notebook cells:** Confirm `python -m pip list`
  shows `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `folium`,
  `joblib`, `pandas-stubs`, and `types-seaborn` under the **same** Python 3.14.7
  executable configured in `pyrefly.toml` (`pyrefly dump-config`) and VS Code.
- **Wrong Jupyter kernel selected:** Open the VS Code Command Palette
  (`Ctrl+Shift+P`), run **Notebook: Select Notebook Kernel**, and choose the
  global **Python 3.14.7** interpreter (`Python 3`).
- **Interactive map rendering:** If `IFrame` does not render inline in a
  restricted viewer, open `visualizations/sf_food_safety_map.html` directly in
  any web browser.

### Verified Deliverable Snapshot

- **Cleaned Records:** `23,103` inspections (`2024-01-02` to `2026-11-20`) after
  removing `139` exact duplicates and `1` future-date outlier (`2031-05-16`),
  with `20,350` `Pass`, `694` `Conditional Pass`, `476` `Closure`, and `1,583`
  `Unspecified` outcomes.
- **Model Evaluation (Chronological 80/20 Split — `17,216` train / `4,304` test):**
  `ROC-AUC = 0.6300`, `PR-AUC = 0.0811`, `Accuracy = 0.95`.

## License

This project is licensed under the [MIT License](LICENSE).
