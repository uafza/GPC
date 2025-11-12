# workflow_gpc.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from parser.batch_parser import HPLCBatchParser
from parser.calibration_parser import GPCCalibrationParser

def _to_number(x):
    s = str(x).strip().replace(",", "")
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return np.nan

def _guess_vial_key_from_quickreport(qdf: pd.DataFrame, parser: GPCCalibrationParser) -> str | None:
    """
    Decide if a sample is a GPC calibration vial based on its QuickReport.
    Returns a key from parser.DEFAULT_EXPECTED (e.g., 'PS-M_Blue') or None.
    Strategy: if ≥1 expected Mp for a vial appears in Name (as number/string) → treat as that vial.
    If multiple match, prefer the one with the most matches (ties: arbitrary first).
    """
    if qdf is None or qdf.empty:
        return None
    # numeric Name helper
    df = qdf.copy()
    df["Name_num"] = df["Name"].apply(_to_number)
    best_key, best_hits = None, 0
    for k, spec in parser.DEFAULT_EXPECTED.items():
        mps = set(spec.get("mp", []))
        hits = 0
        for mp in mps:
            # match numeric or exact string
            hit_num = np.isclose(df["Name_num"], mp, atol=1e-12)
            if hit_num.any():
                hits += 1
                continue
            if (df["Name"].astype(str).str.strip() == str(mp)).any():
                hits += 1
        if hits > best_hits:
            best_hits, best_key = hits, k
    return best_key if best_hits > 0 else None

def _combine_calibration_rows(rows: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate multiple per-vial calibration tables and de-duplicate by (MW,Vial).
    Keep the row with the largest Peak Area for duplicates. Sort by V0 first, then RT.
    """
    if not rows:
        return pd.DataFrame(columns=["Exp. RT (min)", "MW", "Mass", "Peak Area", "Signal", "Vial"])
    df = pd.concat(rows, ignore_index=True)
    if df.empty:
        return df
    # keep max-area per (MW,Vial) (V0 rows have empty MW; treat them as unique by using Vial+Exp.RT)
    key_cols = ["MW", "Vial"]
    df["_rank"] = df.groupby(key_cols)["Peak Area"].rank(method="first", ascending=False)
    df = df[df["_rank"] == 1.0].drop(columns="_rank")
    # order: V0 rows first (sort key big), then by Exp RT
    df["sort_key"] = df.apply(lambda x: 1e6 if str(x["Vial"]) == "V0" else float(x["Exp. RT (min)"]), axis=1)
    df = df.sort_values("sort_key", ignore_index=True).drop(columns="sort_key")
    return df

def run_gpc_workflow(base_folder=".", out_dir="results", signal_label="RID", project_info=None):

    """
    - Uses HPLCBatchParser to load all samples.
    - Splits into:
        * calibration_set  (EasiVial standards identified by QuickReport peak names)
        * sample_set       (everything else)
    - Builds a combined calibration CSV from calibration_set using GPCCalibrationParser.
    - Writes:
        results/calibration/combined_CLBRTN.csv
        results/calibration/per_cal_sample/<Sample>_CLBRTN.csv
        results/sets.json   (lists sample names in each set)
    """
    out = Path(out_dir)
    out_cal = out / "calibration" / "per_cal_sample"
    out_cal.mkdir(parents=True, exist_ok=True)
    (out / "calibration").mkdir(parents=True, exist_ok=True)

    # 1) Load via your base parser
    parser = HPLCBatchParser(base_folder=base_folder)
    samples = parser.parse_batch()  # list[dict], each with keys incl. 'quickreport'
    print(f"Loaded {len(samples)} parsed entries.")

    # 2) Split into sets based on QuickReport peaks
    gpc = GPCCalibrationParser(signal_label=signal_label)
    calibration_set = []
    sample_set = []

    for s in samples:
        name = f"{s.get('barcode','')}_r{s.get('repeat',1)}"
        qdf = s.get("quickreport")
        vial_key = _guess_vial_key_from_quickreport(qdf, gpc)
        if vial_key:
            calibration_set.append({"name": name, "vial_key": vial_key, "quickreport": qdf})
        else:
            sample_set.append({"name": name, "quickreport": qdf})

    print(f"Calibration set: {len(calibration_set)}  |  Sample set: {len(sample_set)}")

    # 3) Build per-calibration-sample tables and a combined table
    per_tables: list[pd.DataFrame] = []
    for entry in calibration_set:
        name = entry["name"]; vial_key = entry["vial_key"]; qdf = entry["quickreport"]
        cal_df = gpc.from_quick_report_df(qdf, vial_key=vial_key)
        per_tables.append(cal_df.assign(_source=name))
        # write per-sample calibration table (use standard CLBRTN header)
        gpc.write_output(cal_df, str(out_cal / f"{name}_CLBRTN.csv"))

    combined = _combine_calibration_rows(per_tables)
    if not combined.empty:
        gpc.write_output(combined, str(out / "calibration" / "combined_CLBRTN.csv"))
        print("Wrote combined calibration →", out / "calibration" / "combined_CLBRTN.csv")
    else:
        print("WARNING: No calibration rows found. Combined calibration not written.")

    # 4) Save set membership for downstream steps
    sets_payload = {
        "calibration_set": [e["name"] for e in calibration_set],
        "sample_set": [e["name"] for e in sample_set],
    }
    (out / "sets.json").write_text(json.dumps(sets_payload, indent=2))
    print("Wrote set membership →", out / "sets.json")
