# SwissCAT+ GPC/HPLC Workflow

Python tooling for the SwissCAT+ pipeline that ingests Shimadzu QuickReport exports, normalises metadata, and produces publication-ready calibration artefacts for both GPC and HPLC batches. The codebase lives under `src/parser` and exposes either a one-shot `run_workflow` façade or a stepwise `GPCWorkflow` helper for notebooks and debugging.

## Highlights
- Batch parser understands the SwissCAT+ directory convention (`QuickReport/`, `AqMethodReport/`, `Chromatograms/`, optional `QualResults/`).
- Calibration logic merges QuickReport peaks with chromatogram-derived integrations, automatically handles V₀ runs, and stores only the combined calibration table (per-sample CSVs were intentionally removed to avoid redundant outputs).
- Optional Plotly visualisations (alignment summaries, calibration overlays, curve fits) saved to `results/calibration/plots` and `results/calibration/fit_plots` when enabled.
- Utility helpers for standardised filenames and Mark–Houwink fitting alongside exploratory tooling in `GPC_analysis.ipynb`.

## Repository Layout
| Path | Purpose |
| --- | --- |
| `src/parser/` | Core package (`workflow_gpc.py`, `workflow_hplc.py`, parsers, utilities, visualisation helpers). |
| `data/` | Example batch drop with the expected Shimadzu folders. Safe to replace with your own runs. |
| `results/` | Default output root. Only the combined calibration CSV and plotting artefacts are produced now. |
| `GPC_analysis.ipynb` | Notebook scratchpad for ad‑hoc analysis/visualisation. |
| `environment.yaml` | Conda environment descriptor that matches the dependencies declared in `pyproject.toml`. |

## Installation
The project follows a standard `src/` layout with metadata in `pyproject.toml`.

```bash
# clone, then create a virtual environment (uv/venv/conda all work)
python -m venv .venv && source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

If you prefer Conda/Mamba:
```bash
mamba env create -f environment.yaml
mamba activate swisscat-gpc
pip install -e .  # ensures the local package is on PYTHONPATH
```

## Running Workflows
```python
from parser.run import run_workflow

project_info = {
    "Workflow": "GPC",          # or "HPLC"
    "CalibPlot": True,          # save Plotly overlays
    "Project": "A069",
    "Task": "T02",
    "Experiment": "E01",
    "Equipment": "GPC1",
    "Batch": "B01",
    "Repeat": "R01",
    "Method": "M01",
}

run_workflow(project_info, base_folder="data", out_dir="results")
```

- `Workflow="GPC"` executes the full calibration path (`run_gpc_workflow`). Plotting is controlled via `CalibPlot`.
- `Workflow="HPLC"` calls `run_hplc_workflow` which simply writes tidy peak tables for each QuickReport.
- Advanced users can instantiate `parser.workflow_gpc.GPCWorkflow` directly to pause between stages (baseline correction, integration, calibration fitting, etc.).

## Expected Inputs
- `data/QuickReport/*.csv` — Shimadzu QuickReport exports per sample (\*_rX.csv naming supported).
- `data/AqMethodReport/*.csv` — structured method dumps for method normalisation.
- `data/Chromatograms/*` — RID/DAD chromatograms that feed integration + alignment.
- `data/QualResults/*.csv` (optional) — processed via `_extract_table_from_shimadzu_csv` when present.

## Outputs
All artefacts land in `results/` (or your custom `out_dir`):
- `results/calibration/combined_CLBRTN.csv` — merged calibration table with injected V₀ refs and computed Kav values.
- `results/sets.json` — quick lookup of which samples were treated as calibration vs production.
- `results/calibration/plots/*.html` — optional alignment + calibration overlays (when `plot=True`).
- `results/calibration/fit_plots/*.html` — optional curve-fit diagnostics from `fit_calibration_curves`.
- **Note:** Per-sample `*_CLBRTN.csv` files were removed; QuickReport copies were redundant and confused downstream automation.

## Development Notes
- Use `ruff` and `black` (declared under the `dev` extra) if you want linting/formatting parity.
- Add large vendor data under `data/` and generated artefacts under `results/`; both paths are ignored by default so you do not accidentally commit sensitive files.
- The `construct_filename` helper in `parser.utils` keeps filenames aligned with SwissCAT+ conventions; prefer it when writing new exporters.

Questions or ideas? Open an issue/PR or keep iterating locally—this README should give future you the necessary context.
