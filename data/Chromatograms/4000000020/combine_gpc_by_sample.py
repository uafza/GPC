#!/usr/bin/env python3
"""
combine_gpc_by_sample.py

Combine multiple detector CSVs (RID + DAD A–H) for a single sample folder
into one table with a common time axis.

Simple behavior:
- Read each CSV as (time, signal)
- Build one common time axis (union of all)
- Interpolate each signal to that axis
- Remove all rows where time < 0
- Trim the end so all columns have the same length (shortest column)
- No unit conversion, no rescaling, no rounding — everything stays as is.

Output: <folder_name>.csv
Columns: R.T. (min.), RID (nRU), DAD A (mAU), DAD B (mAU), ...
"""

from pathlib import Path
import pandas as pd
import numpy as np
import csv
import re
from typing import Dict, List, Tuple

# -------------------------------------------------------------------------
OUTPUT_ORDER = [
    "RID (nRU)",
    "DAD A (mAU)",
    "DAD B (mAU)",
    "DAD C (mAU)",
    "DAD D (mAU)",
    "DAD E (mAU)",
    "DAD F (mAU)",
    "DAD G (mAU)",
    "DAD H (mAU)",
]

CHANNEL_PATTERNS: Dict[str, List[str]] = {
    "RID (nRU)":   ["RID1A"],
    "DAD A (mAU)": ["DAD1A"],
    "DAD B (mAU)": ["DAD1B"],
    "DAD C (mAU)": ["DAD1C"],
    "DAD D (mAU)": ["DAD1D"],
    "DAD E (mAU)": ["DAD1E"],
    "DAD F (mAU)": ["DAD1F"],
    "DAD G (mAU)": ["DAD1G"],
    "DAD H (mAU)": ["DAD1H"],
}
# -------------------------------------------------------------------------


def _first_numeric_line(path: Path) -> int:
    """Find the first line with at least two numeric entries."""
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            cells = row
            if len(cells) == 1 and (";" in cells[0] or "," in cells[0]):
                sep = ";" if ";" in cells[0] else ","
                cells = cells[0].split(sep)
            if len(cells) >= 2:
                try:
                    float(cells[0].replace(",", "."))
                    float(cells[1].replace(",", "."))
                    return i
                except Exception:
                    continue
    return 0


def read_signal_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (time, signal) as float arrays — no conversions."""
    start = _first_numeric_line(path)
    for sep in [",", ";", None]:
        try:
            df = pd.read_csv(path, sep=sep, header=None, skiprows=start, engine="python", dtype=str)
            df = df.dropna(how="all", axis=1)
            if df.shape[1] < 2:
                continue
            t = df.iloc[:, 0].astype(str).str.replace(",", ".").astype(float).to_numpy()
            y = df.iloc[:, 1].astype(str).str.replace(",", ".").astype(float).to_numpy()
            return t, y
        except Exception:
            continue
    raise ValueError(f"Could not parse: {path.name}")


def find_channel_files(folder: Path) -> Dict[str, Path]:
    """Map channel names to CSV files by pattern."""
    files = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
    found: Dict[str, Path] = {}
    upper = {f: f.name.upper() for f in files}
    for col, pats in CHANNEL_PATTERNS.items():
        matches = [f for f, name in upper.items() if all(p.upper() in name for p in pats)]
        if matches:
            matches.sort(key=lambda p: len(p.name), reverse=True)
            found[col] = matches[0]
    return found


def interpolate_to_axis(t: np.ndarray, y: np.ndarray, master_t: np.ndarray) -> np.ndarray:
    """Linear interpolation; fill outside range with NaN."""
    if len(t) < 2:
        return np.full_like(master_t, np.nan)
    out = np.full_like(master_t, np.nan, dtype=float)
    mask = (master_t >= t[0]) & (master_t <= t[-1])
    out[mask] = np.interp(master_t[mask], t, y)
    return out


def main():
    root = Path.cwd()
    out_file = root / f"{root.name}.csv"
    print(f"[RUN] Combining signals in folder: {root.name}")

    channel_files = find_channel_files(root)
    if not channel_files:
        print("No detector CSVs found.")
        return

    times = {}
    signals = {}
    for col in [c for c in OUTPUT_ORDER if c in channel_files]:
        f = channel_files[col]
        try:
            t, y = read_signal_csv(f)
            times[col] = t
            signals[col] = y
            print(f"[OK] {col}: {len(t)} points from {f.name}")
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")

    if not times:
        print("No usable signals.")
        return

    # Common time axis = union of all
    all_t = np.unique(np.concatenate([t for t in times.values()]))
    master_t = np.sort(all_t)

    # Interpolate all signals to master axis
    data = {"R.T. (min.)": master_t}
    for col in OUTPUT_ORDER:
        if col in times:
            data[col] = interpolate_to_axis(times[col], signals[col], master_t)

    df = pd.DataFrame(data)

    # Remove all rows where time < 0
    df = df[df["R.T. (min.)"] >= 0].copy()

    # Trim end so all columns have equal length (drop trailing NaNs)
    valid_mask = np.all(np.isfinite(df.drop(columns=["R.T. (min.)"]).to_numpy()), axis=1)
    if not np.all(valid_mask):
        first_false = np.where(~valid_mask)[0]
        if len(first_false) > 0:
            df = df.iloc[:first_false[0]]

    df.to_csv(out_file, index=False)
    print(f"[DONE] Saved {out_file.name}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
