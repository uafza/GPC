#!/usr/bin/env python3
"""
Combine Agilent detector CSVs per detector into one table with a common time axis.

Usage: open this file in VS Code from the ROOT that contains your barcode folders
(e.g., 4000000001, 4000000002, ...), then press Run ▶. Outputs go to ./_combined/.
"""

from pathlib import Path
import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# ---- Configuration ----------------------------------------------------------
DETECTORS = [f"DAD1{c}" for c in "ABCDEFGH"] + ["RID1A"]

# If your raw time is in seconds and you want minutes, set to True.
CONVERT_SECONDS_TO_MINUTES = False

# -----------------------------------------------------------------------------


def _first_numeric_line(path: Path) -> int:
    """
    Find the first 0-based line index where the row begins with two numeric cells.
    Falls back to 0 if undecidable.
    """
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 1:
                continue
            # Heuristic: split single big fields that still contain delimiters
            cells = row
            if len(cells) == 1 and (";" in cells[0] or "," in cells[0]):
                sep = ";" if ";" in cells[0] else ","
                cells = cells[0].split(sep)

            if len(cells) >= 2:
                try:
                    float(str(cells[0]).strip().replace(",", "."))
                    float(str(cells[1]).strip().replace(",", "."))
                    return i
                except Exception:
                    pass
    return 0


def read_signal_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a two-column time/signal CSV robustly.
    - Skips header junk until first numeric row.
    - Accepts comma or semicolon delimiter.
    - Uses the first two numeric-like columns.
    Returns (time_min, signal).
    """
    start = _first_numeric_line(path)

    for sep in [",", ";", None]:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                header=None,
                skiprows=start,
                engine="python",
                comment="#",
                dtype=str,
            )

            # Normalize text and drop empty columns
            df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else x)
            df = df.dropna(how="all", axis=1)

            # Convert to float where possible
            num = df.applymap(
                lambda x: float(x.replace(",", ".")) if isinstance(x, str) and x not in ("", "nan", "None") else np.nan
            )

            usable = [c for c in num.columns if num[c].notna().sum() > 0]
            if len(usable) < 2:
                continue

            t = num[usable[0]].to_numpy(dtype=float)
            y = num[usable[1]].to_numpy(dtype=float)

            mask = np.isfinite(t) & np.isfinite(y)
            t, y = t[mask], y[mask]

            if t.size == 0:
                continue

            # Sort and deduplicate times
            order = np.argsort(t)
            t, y = t[order], y[order]
            if len(np.unique(t)) != len(t):
                d = pd.DataFrame({"t": t, "y": y}).groupby("t", as_index=False).mean()
                t, y = d["t"].to_numpy(), d["y"].to_numpy()

            if CONVERT_SECONDS_TO_MINUTES:
                t = t / 60.0

            return t, y
        except Exception:
            continue

    raise ValueError(f"Could not parse numeric time/signal from {path.name}")


def interpolate_to_axis(t: np.ndarray, y: np.ndarray, master_t: np.ndarray) -> np.ndarray:
    """
    Linear interpolation onto master_t. Outside original range -> NaN.
    """
    if t.size < 2:
        return np.full_like(master_t, np.nan, dtype=float)

    out = np.full(master_t.shape, np.nan, dtype=float)
    in_range = (master_t >= t[0]) & (master_t <= t[-1])
    if in_range.any():
        out[in_range] = np.interp(master_t[in_range], t, y)
    return out


def collect_files_by_detector(root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Map: {detector: {barcode: file_path}}
    - Barcode = immediate subfolder name
    - Picks a file whose name contains the detector label (case-insensitive)
    - If multiple, picks the longest filename (typical Agilent export)
    """
    mapping: Dict[str, Dict[str, Path]] = {d: {} for d in DETECTORS}

    for folder in sorted([p for p in root.iterdir() if p.is_dir()]):
        barcode = folder.name
        files = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
        if not files:
            continue
        names_upper = {f: f.name.upper() for f in files}

        for det in DETECTORS:
            cand = [f for f, up in names_upper.items() if det in up]
            if not cand:
                continue
            cand.sort(key=lambda p: len(p.name), reverse=True)
            mapping[det][barcode] = cand[0]
    return mapping


def build_and_save_tables(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files_by_det = collect_files_by_detector(root)

    for det in DETECTORS:
        entries = files_by_det.get(det, {})
        if not entries:
            print(f"[INFO] No files found for {det}.")
            continue

        # Read and collect times
        all_times: List[np.ndarray] = []
        ts: Dict[str, np.ndarray] = {}
        ys: Dict[str, np.ndarray] = {}

        for barcode, path in sorted(entries.items()):
            try:
                t, y = read_signal_csv(path)
            except Exception as e:
                print(f"[WARN] {det} | {barcode}: {e}")
                continue
            ts[barcode], ys[barcode] = t, y
            all_times.append(t)

        if not all_times:
            print(f"[INFO] No parsable data for {det}.")
            continue

        # Master time axis = union of all times (keeps native sampling)
        master_t = np.unique(np.concatenate(all_times))
        master_t = np.round(master_t, 6)

        # Interpolate all barcodes
        data = {"RT (min)": master_t}
        for barcode in sorted(ts.keys(), key=lambda b: (0, int(b)) if b.isdigit() else (1, b)):
            data[barcode] = interpolate_to_axis(ts[barcode], ys[barcode], master_t)

        df_out = pd.DataFrame(data)
        out_path = out_dir / f"{det}_combined.csv"
        df_out.to_csv(out_path, index=False)
        print(f"[OK] Wrote {out_path} — rows: {df_out.shape[0]}, columns: {df_out.shape[1]}")


def main():
    root = Path(__file__).resolve().parent  # your root with barcode folders
    out_dir = root / "_combined"
    print(f"[RUN] Root: {root}")
    build_and_save_tables(root, out_dir)
    print("[DONE]")


if __name__ == "__main__":
    main()
