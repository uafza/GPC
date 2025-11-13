# workflow_gpc.py
from __future__ import annotations
from pathlib import Path
import re
import json
from typing import Any, Callable, Sequence
import numpy as np
import pandas as pd

from parser.batch_parser import HPLCBatchParser
from parser.calibration_parser import GPCCalibrationParser
from parser.analyze import HPLCGroupAnalyzer

# Optional: enable inline display when running in notebooks
try:  # pragma: no cover
    from IPython.display import display as _ipython_display
except Exception:  # pragma: no cover
    _ipython_display = None

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
    # Require at least 2 expected Mp matches to classify as a calibration vial
    # (reduces false positives from arbitrary numeric names in samples)
    return best_key if best_hits >= 2 else None


def _guess_vial_key_from_identity(sample: dict, parser: GPCCalibrationParser) -> str | None:
    """
    Classify calibration vial using the sample identity (barcode or quick filename),
    based on DEFAULT_EXPECTED keys (e.g., 'PS-H_White', 'PS-M_Blue').
    Graceful w.r.t. spaces, dashes, underscores, and capitalization.
    Returns one of the DEFAULT_EXPECTED keys or None.
    """
    if not isinstance(sample, dict):
        return None
    def canon(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())
    text = canon(sample.get("barcode", "")) + " " + canon(sample.get("quick_filename", ""))
    text = text.strip()
    if not text:
        return None
    # Build canonical lookup for expected vials
    for key in parser.DEFAULT_EXPECTED.keys():
        key_canon = canon(key.replace("_", " "))
        if key_canon and key_canon in text:
            return key
    return None

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

def _is_blank_sample(sample: dict) -> bool:
    """Detect blanks by sample identity (not peak names):
    Accepts names like 'Blank', 'THF Blank', 'THF Blank_1', case-insensitive.
    """
    if not isinstance(sample, dict):
        return False
    def txt(x):
        return str(x or "").strip().lower()
    barcode = txt(sample.get("barcode"))
    quick_fn = txt(sample.get("quick_filename"))
    return ("blank" in barcode) or ("blank" in quick_fn)


def _is_v0_sample(sample: dict) -> bool:
    """Detect V0 (Toluene) runs by sample identity.
    Matches tokens like 'V0 Toluene' in barcode or quick filename.
    """
    if not isinstance(sample, dict):
        return False
    def txt(x):
        return str(x or "").strip().lower()
    barcode = txt(sample.get("barcode"))
    quick_fn = txt(sample.get("quick_filename"))
    return ("v0" in barcode and "toluene" in barcode) or ("v0" in quick_fn and "toluene" in quick_fn)


def _sample_display_name(sample: dict) -> str:
    return f"{sample.get('barcode','NA')}_r{sample.get('repeat',1)}"


def _baseline_correct_dataframe(
    df: pd.DataFrame,
    *,
    start_window: tuple[float, float] = (7.0, 8.0),
    end_window: tuple[float, float | None] = (21.0, None),
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Apply a linear baseline fit using two retention-time windows and subtract it from every signal column.
    Returns (corrected_df, baseline_df, original_df_aligned) or (None, None, None) when fitting fails.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None, None, None

    working = df.copy()
    time_col = next((c for c in working.columns if "time" in str(c).lower()), None)

    if time_col is not None:
        rt_series = pd.to_numeric(working[time_col], errors="coerce")
        working = working.drop(columns=[time_col])
        time_name = str(time_col)
    else:
        rt_series = pd.Series(pd.to_numeric(pd.Index(working.index), errors="coerce"), index=working.index)
        time_name = working.index.name or "Time (min)"

    if working.empty:
        return None, None, None

    rt = pd.Series(rt_series, index=working.index)
    rt_valid = rt.notna()
    if rt_valid.sum() < 2:
        return None, None, None

    start_lo, start_hi = map(float, start_window)
    end_lo = float(end_window[0])
    end_hi = float(end_window[1]) if end_window[1] is not None else np.inf
    start_mask = (rt >= start_lo) & (rt <= start_hi)
    end_mask = (rt >= end_lo) & (rt <= end_hi)
    fit_mask = (start_mask | end_mask) & rt_valid
    if fit_mask.sum() < 2:
        return None, None, None

    original = working.copy()
    corrected = working.copy()
    baseline_df = pd.DataFrame(index=working.index)
    times_all = rt.to_numpy(dtype=float)
    finite_time_mask = np.isfinite(times_all)
    applied = False

    for col in working.columns:
        col_numeric = pd.to_numeric(working[col], errors="coerce")
        y_fit = col_numeric[fit_mask].to_numpy(dtype=float)
        t_fit = rt[fit_mask].to_numpy(dtype=float)
        fit_valid = np.isfinite(y_fit) & np.isfinite(t_fit)
        if fit_valid.sum() < 2:
            continue
        y_fit = y_fit[fit_valid]
        t_fit = t_fit[fit_valid]
        X = np.vstack([np.ones_like(t_fit), t_fit]).T
        try:
            beta, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
        except Exception:
            continue
        baseline_vals = beta[0] + beta[1] * times_all
        values_all = col_numeric.to_numpy(dtype=float)
        corrected_vals = values_all.copy()
        subtract_mask = np.isfinite(values_all) & finite_time_mask
        corrected_vals[subtract_mask] = values_all[subtract_mask] - baseline_vals[subtract_mask]
        corrected[col] = corrected_vals
        baseline_df[col] = baseline_vals
        applied = True

    if not applied:
        return None, None, None
    corrected.index = pd.Index(times_all, name=time_name or "Time (min)")
    original.index = corrected.index
    baseline_df.index = corrected.index
    return corrected, baseline_df, original


def _plot_baseline_subtractions_combined(
    sample_name: str,
    chrom_payloads: dict[str, dict],
    plot_dir: Path | None,
    *,
    display_inline: bool = False,
) -> None:
    """
    Render a single baseline-subtraction figure per sample with RID on the left axis
    and DAD (or other) signals on the right axis.
    """
    if plot_dir is None or not chrom_payloads:
        return
    try:
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
    except Exception:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    color_idx = 0
    left_label = None
    right_label = None
    for chrom_label, payload in chrom_payloads.items():
        corrected = payload.get("corrected")
        original = payload.get("original")
        baseline = payload.get("baseline")
        axis_pref = str(payload.get("axis", "left")).lower()
        secondary = axis_pref == "right"
        if secondary:
            right_label = right_label or f"{chrom_label} signal"
        else:
            left_label = left_label or f"{chrom_label} signal"
        if not isinstance(corrected, pd.DataFrame) or corrected.empty:
            continue
        x = corrected.index.to_numpy(dtype=float)
        for col in corrected.columns:
            series_corrected = pd.to_numeric(corrected[col], errors="coerce")
            if not np.any(np.isfinite(series_corrected)):
                continue
            color = colors[color_idx % len(colors)]
            color_idx += 1
            label_base = f"{chrom_label} {col}"
            if isinstance(original, pd.DataFrame) and col in original.columns:
                series_raw = pd.to_numeric(original[col], errors="coerce")
                if np.any(np.isfinite(series_raw)):
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=series_raw,
                            mode="lines",
                            name=f"{label_base} raw",
                            line=dict(color=color, width=1, dash="dot"),
                        ),
                        secondary_y=secondary,
                    )
            if isinstance(baseline, pd.DataFrame) and col in baseline.columns:
                series_baseline = pd.to_numeric(baseline[col], errors="coerce")
                if np.any(np.isfinite(series_baseline)):
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=series_baseline,
                            mode="lines",
                            name=f"{label_base} baseline",
                            line=dict(color=color, width=1, dash="dash"),
                        ),
                        secondary_y=secondary,
                    )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=series_corrected,
                    mode="lines",
                    name=f"{label_base} corrected",
                    line=dict(color=color, width=2),
                ),
                secondary_y=secondary,
            )
    fig.update_xaxes(title_text="Time (min)")
    if left_label:
        fig.update_yaxes(title_text=left_label, secondary_y=False)
    if right_label:
        fig.update_yaxes(title_text=right_label, secondary_y=True)
    fig.update_layout(
        title=f"{sample_name} | Baseline subtraction overview",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )

    def _slug(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", text or "sample").strip("_") or "sample"

    fname = f"{_slug(sample_name)}_baseline.html"
    try:
        import plotly.io as pio
        pio.write_html(fig, str(plot_dir / fname), include_plotlyjs="cdn", auto_open=False)
    except Exception:
        pass
    if display_inline and _ipython_display is not None:
        try:
            _ipython_display(fig)
        except Exception:
            pass


def _baseline_correct_samples(
    samples: list[dict],
    *,
    start_window: tuple[float, float] = (7.0, 8.0),
    end_window: tuple[float, float | None] = (21.0, None),
    chromatogram_specs: tuple[dict[str, tuple[str, ...]], ...] | None = None,
    plot: bool = False,
    plot_dir: Path | None = None,
    name_getter: Callable[[dict], str] | None = None,
) -> int:
    """
    Iterate over parsed samples and attach baseline-corrected chromatograms (RID, DAD, ...).
    Returns the number of chromatograms successfully corrected.
    """
    if not samples:
        return 0
    specs = chromatogram_specs or (
        {"target_key": "chrom_dad_baseline_corrected", "source_keys": ("chrom_dad_aligned", "chrom_dad"), "label": "DAD", "plot_axis": "right"},
        {"target_key": "chrom_rid_baseline_corrected", "source_keys": ("chrom_rid",), "label": "RID", "plot_axis": "left"},
    )
    count = 0
    if plot and plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        sample_name = name_getter(sample) if name_getter else _sample_display_name(sample)
        per_sample_plots: dict[str, dict] | None = {} if (plot and plot_dir is not None) else None
        for spec in specs:
            target = spec.get("target_key")
            if not target:
                continue
            existing = sample.get(target)
            if isinstance(existing, pd.DataFrame) and not existing.empty:
                continue
            source_keys = spec.get("source_keys") or ()
            for key in source_keys:
                candidate = sample.get(key)
                if not isinstance(candidate, pd.DataFrame) or candidate.empty:
                    continue
                corrected, baseline_df, original_df = _baseline_correct_dataframe(
                    candidate,
                    start_window=start_window,
                    end_window=end_window,
                )
                if corrected is None:
                    continue
                sample[target] = corrected
                if per_sample_plots is not None and baseline_df is not None and original_df is not None:
                    chrom_label = spec.get("label") or target
                    per_sample_plots[chrom_label] = {
                        "original": original_df,
                        "baseline": baseline_df,
                        "corrected": corrected,
                        "axis": spec.get("plot_axis", "left"),
                    }
                count += 1
                break
        if per_sample_plots:
            _plot_baseline_subtractions_combined(
                sample_name,
                per_sample_plots,
                plot_dir,
                display_inline=plot,
            )
    return count


def _rid_plot_with_annotations(sample: dict, cal_df: pd.DataFrame, title: str):
    try:
        import plotly.graph_objs as go
    except Exception:
        return None
    rid = sample.get("chrom_rid_baseline_corrected")
    if (not isinstance(rid, pd.DataFrame)) or rid.empty:
        rid = sample.get("chrom_rid")
    if rid is None or not isinstance(rid, pd.DataFrame) or rid.empty:
        return None
    # ensure index time
    if rid.index.name is None or "time" not in str(rid.index.name).lower():
        rid = rid.set_index(rid.columns[0])
    # pick RID signal column
    col = None
    for c in rid.columns:
        if str(c).startswith("Signal (") and str(c).endswith(")"):
            col = c; break
    if col is None:
        col = rid.columns[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rid.index, y=rid[col], mode="lines", name="RID"))
    # annotations from cal_df
    for _, r in cal_df.iterrows():
        rt = float(r["Exp. RT (min)"])
        label = ("V0" if str(r.get("Vial",""))=="V0" else f"MW {r['MW']}")
        fig.add_shape(type="line", x0=rt, x1=rt, y0=0, y1=1, yref="paper", line=dict(color="red", dash="dot"))
        fig.add_annotation(x=rt, y=1.02, yref="paper", showarrow=False, text=label, xanchor="center", font=dict(size=10))
    fig.update_layout(title=title, xaxis_title="Time (min.)", yaxis_title=str(col))
    return fig


def _canon_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _color_for_sample(name: str, vial_key: str | None) -> str:
    """Return a consistent color per calibration sample.
    Rule: 'Red' -> red, 'Blue' -> blue, 'White' -> black/gray.
    Shade: H = darker, M = lighter. Fallback to gray.
    """
    key_src = vial_key or name
    z = _canon_text(key_src)
    is_h = 'psh' in z  # PS-H
    is_m = 'psm' in z  # PS-M
    if 'red' in z:
        return '#8B0000' if is_h else '#FF4D4D'  # darkred vs light red
    if 'blue' in z:
        return '#0B3D91' if is_h else '#1E90FF'  # dark blue vs dodger blue
    if 'white' in z:
        return '#000000' if is_h else '#555555'  # black vs dim gray
    # V0 or unknown
    if 'v0' in z and 'toluene' in z:
        return '#6A5ACD'  # slate blue for V0
    return '#888888'


def _flow_rate_from_sample(sample: dict) -> float | None:
    try:
        method_norm = sample.get("method_norm") or {}
        pump = method_norm.get("pump") or {}
        flow = pump.get("flow_ml_min")
        if flow is None:
            return None
        flow = float(flow)
        return flow if np.isfinite(flow) and flow > 0 else None
    except Exception:
        return None


def _color_with_alpha(color: str, alpha: float) -> str:
    """Convert a hex or rgb color string into an rgba() string with the requested alpha."""
    a = max(0.0, min(1.0, float(alpha)))
    value = str(color or "").strip()
    if value.startswith("#") and len(value) == 7:
        try:
            r = int(value[1:3], 16)
            g = int(value[3:5], 16)
            b = int(value[5:7], 16)
            return f"rgba({r},{g},{b},{a})"
        except ValueError:
            return value
    if value.lower().startswith("rgb(") and value.endswith(")"):
        try:
            parts = [int(float(p.strip())) for p in value[4:-1].split(",")[:3]]
            if len(parts) == 3:
                r, g, b = parts
                return f"rgba({r},{g},{b},{a})"
        except Exception:
            return value
    return value


def _select_primary_signal_column(df: pd.DataFrame | None) -> str | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for col in df.columns:
        text = str(col).lower()
        if "time" in text:
            continue
        if "signal" in text:
            return col
    for col in df.columns:
        if "time" not in str(col).lower():
            return col
    return df.columns[0]


def _get_chromatogram_from_sample(sample: dict, keys: tuple[str, ...]) -> tuple[pd.DataFrame | None, str | None]:
    if not isinstance(sample, dict):
        return (None, None)
    for key in keys:
        candidate = sample.get(key)
        if isinstance(candidate, pd.DataFrame) and not candidate.empty:
            return candidate, key
    return (None, None)


def _prepare_chromatogram(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Ensure a chromatogram is indexed by retention time and numerically typed.
    Returns a new DataFrame or None if preparation fails.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    working = df.copy()
    if working.index.name is None or "time" not in str(working.index.name).lower():
        time_cols = [c for c in working.columns if "time" in str(c).lower()]
        if time_cols:
            working = working.set_index(time_cols[0])
    working.index = pd.to_numeric(working.index, errors="coerce")
    working = working[~working.index.isna()]
    if working.empty:
        return None
    working = working.sort_index()
    working = working.apply(pd.to_numeric, errors="coerce")
    working = working.dropna(axis=1, how="all")
    working = working.dropna(axis=0, how="all")
    if working.empty:
        return None
    return working


def _attach_volume_index(df: pd.DataFrame | None, flow_rate: float | None) -> pd.DataFrame | None:
    """
    Convert a chromatogram from time-indexed (minutes) to volume-indexed (mL) using the flow rate.
    Retains the original time as a column for downstream consumers.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or not flow_rate or not np.isfinite(flow_rate):
        return df
    try:
        times = df.index.to_numpy(dtype=float)
    except Exception:
        return df
    if not np.any(np.isfinite(times)):
        return df
    volumes = times * float(flow_rate)
    updated = df.copy()
    updated.index = pd.Index(volumes, name="Volume (mL)")
    return updated


def _crop_by_volume(df: pd.DataFrame | None, v0: float | None, vc: float | None) -> pd.DataFrame | None:
    if not isinstance(df, pd.DataFrame) or df.empty or v0 is None or vc is None:
        return df
    lo = min(v0, vc)
    hi = max(v0, vc)
    try:
        mask = (df.index >= lo) & (df.index <= hi)
        if not mask.any():
            return df
        return df.loc[mask]
    except Exception:
        return df



def _attach_kav_index(df: pd.DataFrame | None, v0: float | None, vc: float | None) -> pd.DataFrame | None:
    """
    Convert a chromatogram indexed by volume (mL) into Kav space using provided V0/Vc.
    """
    if (
        not isinstance(df, pd.DataFrame)
        or df.empty
        or v0 is None
        or vc is None
        or v0 == vc
    ):
        return df
    try:
        volumes = df.index.to_numpy(dtype=float)
    except Exception:
        return df
    if not np.any(np.isfinite(volumes)):
        return df
    kav = (volumes - float(v0)) / (float(vc) - float(v0))
    updated = df.copy()
    updated.index = pd.Index(kav, name="Kav")
    return updated


def _volume_to_kav(volume: float | None, v0: float | None, vc: float | None) -> float | None:
    if (
        volume is None
        or v0 is None
        or vc is None
        or v0 == vc
    ):
        return None
    try:
        return (float(volume) - float(v0)) / (float(vc) - float(v0))
    except Exception:
        return None


def _crop_by_index_range(df: pd.DataFrame | None, lo: float, hi: float) -> pd.DataFrame | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    lo_val = min(lo, hi)
    hi_val = max(lo, hi)
    try:
        mask = (df.index >= lo_val) & (df.index <= hi_val)
        if not mask.any():
            return df
        return df.loc[mask]
    except Exception:
        return df


def _peak_regions_from_quickreport(
    cal_df: pd.DataFrame | None,
    *,
    reference_index: pd.Index | None = None,
    pad: float = 0.35,
    min_span: float = 0.05,
    flow_rate_ml_min: float | None = None,
) -> list[dict]:
    """
    Build a list of peak regions (left/right bounds) using QuickReport retention times.
    Bounds are midpoints between adjacent peaks, optionally clipped to the chromatogram domain.
    """
    if not isinstance(cal_df, pd.DataFrame) or cal_df.empty or "Exp. RT (min)" not in cal_df.columns:
        return []
    rt_series = pd.to_numeric(cal_df["Exp. RT (min)"], errors="coerce")
    mask = rt_series.notna()
    if not mask.any():
        return []
    ordered = cal_df.loc[mask].copy()
    ordered["_rt"] = rt_series[mask].astype(float)
    ordered.sort_values("_rt", inplace=True, kind="mergesort")

    ref_min = ref_max = None
    if reference_index is not None:
        try:
            ref_idx = pd.Index(reference_index)
            ref_numeric = pd.to_numeric(ref_idx, errors="coerce")
            ref_numeric = ref_numeric[ref_numeric.notna()]
            if len(ref_numeric):
                ref_min = float(ref_numeric.min())
                ref_max = float(ref_numeric.max())
        except Exception:
            ref_min = ref_max = None
    if ref_min is None or ref_max is None:
        ref_min = float(ordered["_rt"].min() - pad)
        ref_max = float(ordered["_rt"].max() + pad)
    if ref_max <= ref_min:
        ref_min -= pad
        ref_max += pad

    rt_values = ordered["_rt"].tolist()
    regions: list[dict] = []
    total = len(rt_values)
    for pos, (row_idx, row) in enumerate(ordered.iterrows()):
        center = float(row["_rt"])
        prev_rt = rt_values[pos - 1] if pos > 0 else None
        next_rt = rt_values[pos + 1] if pos < total - 1 else None
        left = center - pad if prev_rt is None else (prev_rt + center) / 2.0
        right = center + pad if next_rt is None else (center + next_rt) / 2.0
        left = max(ref_min, left)
        right = min(ref_max, right)
        if right - left < min_span:
            delta = (min_span - (right - left)) / 2.0
            left = max(ref_min, left - delta)
            right = min(ref_max, right + delta)
        payload = {
            "row_index": row_idx,
            "center": center,
            "left": float(left),
            "right": float(right),
            "mw": row.get("MW"),
            "vial": row.get("Vial"),
        }
        if flow_rate_ml_min and np.isfinite(flow_rate_ml_min):
            flow = float(flow_rate_ml_min)
            payload["center_vol"] = center * flow
            payload["left_vol"] = float(left) * flow
            payload["right_vol"] = float(right) * flow
        regions.append(payload)
    return regions


def _integrate_peak_regions(chrom_df: pd.DataFrame | None, regions: list[dict]) -> dict[str, dict[int, float]]:
    """
    Integrate each chromatogram column within the supplied regions.
    Returns mapping: column_name -> {row_index: area}.
    """
    if not isinstance(chrom_df, pd.DataFrame) or chrom_df.empty or not regions:
        return {}
    idx = chrom_df.index.to_numpy(dtype=float)
    if idx.size == 0:
        return {}
    out: dict[str, dict[int, float]] = {}
    for region in regions:
        left_in = region.get("left_vol")
        right_in = region.get("right_vol")
        left = float(left_in) if left_in is not None else float(region["left"])
        right = float(right_in) if right_in is not None else float(region["right"])
        window = chrom_df.loc[(chrom_df.index >= left) & (chrom_df.index <= right)]
        if window.empty:
            continue
        signal_cols = [c for c in window.columns if "time" not in str(c).lower()]
        if not signal_cols:
            continue
        window = window[signal_cols]
        x = window.index.to_numpy(dtype=float)
        y = window.to_numpy(dtype=float)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        y = np.nan_to_num(y, nan=0.0)
        areas = np.trapz(y, x, axis=0)
        for col, area in zip(window.columns, areas):
            by_col = out.setdefault(col, {})
            by_col[region["row_index"]] = float(area)
    return out


def _integrate_regions_on_axis(
    chrom_df: pd.DataFrame | None,
    regions_axis: list[dict],
    *,
    left_key: str = "left_axis",
    right_key: str = "right_axis",
) -> dict[str, dict[int, float]]:
    """
    Integrate using arbitrary axis coordinates already aligned with the chromatogram index.
    """
    if not isinstance(chrom_df, pd.DataFrame) or chrom_df.empty or not regions_axis:
        return {}
    out: dict[str, dict[int, float]] = {}
    for region in regions_axis:
        left = region.get(left_key)
        right = region.get(right_key)
        if left is None or right is None:
            continue
        window = chrom_df.loc[(chrom_df.index >= left) & (chrom_df.index <= right)]
        if window.empty:
            continue
        signal_cols = [c for c in window.columns if "time" not in str(c).lower()]
        if not signal_cols:
            continue
        window = window[signal_cols]
        x = window.index.to_numpy(dtype=float)
        y = window.to_numpy(dtype=float)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        y = np.nan_to_num(y, nan=0.0)
        areas = np.trapz(y, x, axis=0)
        for col, area in zip(window.columns, areas):
            by_col = out.setdefault(col, {})
            row_idx = region.get("row_index")
            if row_idx is None:
                continue
            by_col[int(row_idx)] = float(area)
    return out


def run_gpc_on_samples(samples: list[dict], out_dir: str = "results", signal_label: str = "RID", plot_calibration: bool = False):
    """
    Run GPC calibration on a provided list of parsed samples.
    - Tags samples: gpc_calibration / gpc_blank / gpc_sample
    - Writes per-sample CLBRTN tables and combined calibration file
    - Optionally displays and saves RID plots with annotations
    """
    out = Path(out_dir)
    out_cal = out / "calibration" / "per_cal_sample"
    out_cal.mkdir(parents=True, exist_ok=True)
    (out / "calibration").mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(samples)} parsed entries.")

    gpc = GPCCalibrationParser(signal_label=signal_label)
    calibration_set = []
    sample_set = []

    for s in samples:
        name = f"{s.get('barcode','')}_r{s.get('repeat',1)}"
        qdf = s.get("quickreport")
        # Prefer identity-based classification over peak-name heuristics
        vial_key = _guess_vial_key_from_identity(s, gpc)
        if vial_key is None:
            vial_key = _guess_vial_key_from_quickreport(qdf, gpc)
        inj_ts = s.get("inj_ts")
        if vial_key:
            s['sample_type'] = 'gpc_calibration'
            calibration_set.append({"name": name, "vial_key": vial_key, "quickreport": qdf, "sample": s, "inj_ts": inj_ts})
        elif _is_v0_sample(s):
            # Treat V0 (Toluene) as part of calibration
            s['sample_type'] = 'gpc_calibration'
            calibration_set.append({"name": name, "vial_key": None, "quickreport": qdf, "sample": s, "inj_ts": inj_ts})
        else:
            if _is_blank_sample(s):
                # mark blanks but do not count them in sample_set
                s['sample_type'] = 'gpc_blank'
            else:
                s['sample_type'] = s.get('sample_type', 'gpc_sample')
                sample_set.append({"name": name, "quickreport": qdf, "sample": s, "inj_ts": inj_ts})

    print(f"Calibration set: {len(calibration_set)}  |  Sample set: {len(sample_set)}")

    # Find preceding V0 for each cal-sample
    def _to_dt(ts):
        try:
            return pd.to_datetime(ts, format="%Y%m%d %H%M%S", errors="coerce")
        except Exception:
            return pd.NaT
    # Build V0 index from all samples
    v0s = []
    for s in samples:
        if _is_v0_sample(s):
            v0s.append({
                "sample": s,
                "quickreport": s.get("quickreport"),
                "inj_ts": s.get("inj_ts"),
            })
    for e in v0s:
        e["_dt"] = _to_dt(e.get("inj_ts"))
    v0s = sorted(v0s, key=lambda x: (x.get("_dt") if pd.notna(x.get("_dt")) else pd.Timestamp.min))

    per_tables: list[pd.DataFrame] = []
    plots_dir = (out / "calibration" / "plots")
    if plot_calibration:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Align DAD to RID on each V0 sample prior to calibration runs ---
    if v0s:
        try:
            analyzer = HPLCGroupAnalyzer(samples)
            for v in v0s:
                s_v0 = v.get("sample")
                name_v0 = f"{s_v0.get('barcode','')}_r{s_v0.get('repeat',1)}"
                try:
                    res = analyzer.align_dad_to_rid_by_peak(
                        s_v0,
                        time_range=(18.0, 22.0),
                        dad_nm=254.0,
                        smoothing=True,
                        smooth_window_pts=7,
                        smooth_method='median',
                        plot=plot_calibration,
                    )
                    # align_dad_to_rid_by_peak returns dt or (dt, fig) depending on plot
                    if isinstance(res, tuple):
                        dt, fig = res
                    else:
                        dt, fig = res, None
                    print(f"Aligned V0 sample {name_v0} dt= {dt}")
                    if fig is not None:
                        if _ipython_display is not None:
                            try:
                                _ipython_display(fig)
                            except Exception:
                                pass
                        try:
                            import plotly.io as pio
                            pio.write_html(fig, str(plots_dir / f"{name_v0}_alignment.html"), include_plotlyjs="cdn", auto_open=False)
                        except Exception:
                            pass
                except Exception:
                    # keep going if alignment fails
                    pass
        except Exception:
            pass

    baseline_plot_dir = (plots_dir / "baseline") if plot_calibration else None
    corrected_count = _baseline_correct_samples(
        samples,
        plot=plot_calibration,
        plot_dir=baseline_plot_dir,
        name_getter=_sample_display_name,
    )
    if corrected_count:
        print(f"Baseline corrected chromatograms: {corrected_count}")
    else:
        print("Baseline correction skipped (no valid chromatograms found).")

    for entry in sorted(calibration_set, key=lambda x: (_to_dt(x.get("inj_ts")) if x.get("inj_ts") else pd.Timestamp.min)):
        name = entry["name"]; vial_key = entry["vial_key"]; qdf = entry["quickreport"]; s_cal = entry["sample"]
        cal_df = gpc.from_quick_report_df(qdf, vial_key=vial_key)

        dt_cal = _to_dt(entry.get("inj_ts"))
        if v0s and pd.notna(dt_cal):
            candidates = [b for b in v0s if b.get("_dt") is not None and b.get("_dt") <= dt_cal]
            if candidates:
                v_prev = candidates[-1]
                qdf_v = v_prev.get("quickreport")
                try:
                    v_parsed = gpc.from_quick_report_df(qdf_v)
                    v0_df = v_parsed[v_parsed["Vial"].astype(str) == "V0"]
                    if not v0_df.empty:
                        cal_df = cal_df[cal_df["Vial"].astype(str) != "V0"]
                        cal_df = pd.concat([v0_df, cal_df], ignore_index=True)
                except Exception:
                    pass

        per_tables.append(cal_df.assign(_source=name))
        gpc.write_output(cal_df, str(out_cal / f"{name}_CLBRTN.csv"))

        if plot_calibration:
            fig = _rid_plot_with_annotations(s_cal, cal_df, title=f"{name} calibration")
            if fig is not None:
                if _ipython_display is not None:
                    try:
                        _ipython_display(fig)
                    except Exception:
                        pass
                try:
                    import plotly.io as pio
                    pio.write_html(fig, str(plots_dir / f"{name}.html"), include_plotlyjs="cdn", auto_open=False)
                except Exception:
                    pass

    combined = _combine_calibration_rows(per_tables)
    if not combined.empty:
        gpc.write_output(combined, str(out / "calibration" / "combined_CLBRTN.csv"))
        print("Wrote combined calibration ->", out / "calibration" / "combined_CLBRTN.csv")
    else:
        print("WARNING: No calibration rows found. Combined calibration not written.")

    sets_payload = {
        "calibration_set": [e["name"] for e in calibration_set],
        "sample_set": [e["name"] for e in sample_set],
    }
    (out / "sets.json").write_text(json.dumps(sets_payload, indent=2))
    print("Wrote set membership ->", out / "sets.json")

    return combined


def run_gpc_workflow(base_folder="data", out_dir="results", signal_label="RID", project_info=None, plot_calibration: bool = False):

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

    # 1) Load via base parser tagged as GPC, then delegate
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parser = HPLCBatchParser(base_folder=base_folder, workflow="GPC")
    samples = parser.parse_batch()
    run_gpc_on_samples(samples, out_dir=out_dir, signal_label=signal_label, plot_calibration=plot_calibration)


# ---------------- OO wrapper for stepwise control ----------------
class GPCWorkflow:
    """
    Stepwise GPC workflow runner.

    Usage:
      wf = GPCWorkflow(samples=samples, out_dir='results')
      wf.detect_sets()           # classify calibration/blank/sample
      wf.align_v0(plot=True)     # align V0 DAD->RID and plot if desired
      wf.baseline_correct_dad()  # subtract linear baseline from DAD channels
      wf.integrate_peaks(plot=True)        # integrate calibration peaks + visualize
      wf.determine_calibration_volumes()   # compute V0/Vc/Ve in mL
      wf.apply_calibration_volumes()       # translate axes to Kav & crop
      combined = wf.build_calibration_tables()  # write per-sample + combined tables and plots

    Or run all in one go:
      wf = GPCWorkflow(base_folder='data', out_dir='results')
      wf.run(plot=True)
    """

    def __init__(self, samples: list[dict] | None = None, base_folder: str = "data", out_dir: str = "results", signal_label: str = "RID"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.signal_label = signal_label
        if samples is None:
            parser = HPLCBatchParser(base_folder=base_folder, workflow="GPC")
            samples = parser.parse_batch()
        self.samples: list[dict] = samples
        self.calibration_set: list[dict] = []
        self.sample_set: list[dict] = []
        self.v0s: list[dict] = []
        self._gpc = GPCCalibrationParser(signal_label=signal_label)
        # in-memory outputs
        self.per_calibration_tables: dict[str, pd.DataFrame] = {}
        self.combined_calibration: pd.DataFrame | None = None
        self.calibration_fits: dict[str, dict] = {}
        self.calibration_fit: dict | None = None
        self._calibration_state: dict[str, Any] = {}
        self.peak_metrics: pd.DataFrame = pd.DataFrame()

    def detect_sets(self) -> tuple[int, int]:
        cal, sam = [], []
        for s in self.samples:
            name = f"{s.get('barcode','')}_r{s.get('repeat',1)}"
            qdf = s.get("quickreport")
            vial_key = _guess_vial_key_from_identity(s, self._gpc)
            if vial_key is None:
                vial_key = _guess_vial_key_from_quickreport(qdf, self._gpc)
            inj_ts = s.get("inj_ts")
            if vial_key:
                s['sample_type'] = 'gpc_calibration'
                cal.append({"name": name, "vial_key": vial_key, "quickreport": qdf, "sample": s, "inj_ts": inj_ts})
            elif _is_v0_sample(s):
                s['sample_type'] = 'gpc_calibration'
                cal.append({"name": name, "vial_key": None, "quickreport": qdf, "sample": s, "inj_ts": inj_ts})
            else:
                if _is_blank_sample(s):
                    s['sample_type'] = 'gpc_blank'
                else:
                    s['sample_type'] = s.get('sample_type', 'gpc_sample')
                    sam.append({"name": name, "quickreport": qdf, "sample": s, "inj_ts": inj_ts})
        # Build V0 index from all samples
        v0s = []
        for s in self.samples:
            if _is_v0_sample(s):
                v0s.append({"sample": s, "quickreport": s.get("quickreport"), "inj_ts": s.get("inj_ts")})
        def _to_dt_local(ts):
            try:
                return pd.to_datetime(ts, format="%Y%m%d %H%M%S", errors="coerce")
            except Exception:
                return pd.NaT
        for e in v0s:
            e["_dt"] = _to_dt_local(e.get("inj_ts"))
        v0s = sorted(v0s, key=lambda x: (x.get("_dt") if pd.notna(x.get("_dt")) else pd.Timestamp.min))

        self.calibration_set, self.sample_set, self.v0s = cal, sam, v0s
        print(f"Calibration set: {len(cal)}  |  Sample set: {len(sam)}")
        return len(cal), len(sam)

    def align_v0(self, plot: bool = False):
        if not self.v0s:
            # ensure sets are detected
            self.detect_sets()
        if not self.v0s:
            return
        plots_dir = self.out_dir / "calibration" / "plots"
        if plot:
            plots_dir.mkdir(parents=True, exist_ok=True)
        analyzer = HPLCGroupAnalyzer(self.samples)
        for v in self.v0s:
            s_v0 = v.get("sample")
            name_v0 = f"{s_v0.get('barcode','')}_r{s_v0.get('repeat',1)}"
            try:
                res = analyzer.align_dad_to_rid_by_peak(
                    s_v0,
                    time_range=(18.0, 22.0),
                    dad_nm=254.0,
                    smoothing=True,
                    smooth_window_pts=7,
                    smooth_method='median',
                    plot=plot,
                )
                dt, fig = (res if isinstance(res, tuple) else (res, None))
                print(f"Aligned V0 sample {name_v0} dt= {dt}")
                if fig is not None:
                    if _ipython_display is not None:
                        try:
                            _ipython_display(fig)
                        except Exception:
                            pass
                    try:
                        import plotly.io as pio
                        pio.write_html(fig, str(plots_dir / f"{name_v0}_alignment.html"), include_plotlyjs="cdn", auto_open=False)
                    except Exception:
                        pass
            except Exception:
                continue

    def baseline_correct_chromatograms(
        self,
        *,
        start_window: tuple[float, float] = (7.0, 8.0),
        end_window: tuple[float, float | None] = (21.0, None),
        chromatogram_specs: tuple[dict[str, tuple[str, ...]], ...] | None = None,
        plot: bool = False,
        plot_dir: Path | None = None,
    ) -> int:
        """
        Subtract a linear baseline from every available chromatogram (RID, DAD, ...) across all samples.
        Returns the number of chromatograms corrected.
        """
        if plot and plot_dir is None:
            plot_dir = self.out_dir / "calibration" / "baseline_plots"
        count = _baseline_correct_samples(
            self.samples,
            start_window=start_window,
            end_window=end_window,
            chromatogram_specs=chromatogram_specs,
            plot=plot,
            plot_dir=plot_dir,
            name_getter=_sample_display_name,
        )
        if count:
            print(f"Baseline corrected chromatograms: {count}")
        else:
            print("Baseline correction skipped (no valid chromatograms found).")
        return count

    def baseline_correct_dad(
        self,
        *,
        start_window: tuple[float, float] = (7.0, 8.0),
        end_window: tuple[float, float | None] = (21.0, None),
        chromatogram_specs: tuple[dict[str, tuple[str, ...]], ...] | None = None,
        plot: bool = False,
        plot_dir: Path | None = None,
    ) -> int:
        """
        Backwards-compatible alias for baseline_correct_chromatograms.
        """
        return self.baseline_correct_chromatograms(
            start_window=start_window,
            end_window=end_window,
            chromatogram_specs=chromatogram_specs,
            plot=plot,
            plot_dir=plot_dir,
        )
        return count

    def _require_calibration_state(self) -> dict[str, Any]:
        state = getattr(self, "_calibration_state", None)
        if not state:
            raise RuntimeError("integrate_peaks() must be called before this step.")
        return state

    def integrate_peaks(self, plot: bool = False) -> None:
        if not self.calibration_set:
            self.detect_sets()

        out_cal = self.out_dir / "calibration" / "per_cal_sample"
        out_cal.mkdir(parents=True, exist_ok=True)
        plots_dir = self.out_dir / "calibration" / "plots"
        if plot:
            plots_dir.mkdir(parents=True, exist_ok=True)

        entries_sorted = self._sorted_calibration_entries()
        peak_payload = self._collect_calibration_peak_data(
            entries_sorted=entries_sorted,
            out_cal=out_cal,
            plot=plot,
            overview_fig=None,
        )

        prepared_raw = self._prepare_plot_payloads_raw(peak_payload["plot_payloads"])
        if plot and prepared_raw:
            fig = self._init_calibration_overview(
                plot=True,
                entries=entries_sorted,
                height_per_subplot=130,
            )
            self._render_plot_payloads(prepared_raw, fig)
            if fig is not None:
                if _ipython_display is not None:
                    try:
                        _ipython_display(fig)
                    except Exception:
                        pass
                else:  # pragma: no cover
                    try:
                        fig.show()
                    except Exception:
                        pass

        self._calibration_state = {
            "per_map": peak_payload["per_map"],
            "output_paths": peak_payload["output_paths"],
            "plot_payloads": peak_payload["plot_payloads"],
            "volume_records": peak_payload["volume_records"],
            "plots_dir": plots_dir,
            "plot_flag": plot,
            "entries_sorted": entries_sorted,
            "overview_fig": None,
        }

    def determine_calibration_volumes(
        self,
        peak_targets: Sequence[Any] | None = None,
        bound_preferences: Sequence[str] | None = None,
    ) -> dict[str, float | None]:
        state = self._require_calibration_state()
        global_v0_volume, global_vc_volume = self._determine_calibration_volumes(state["volume_records"])

        def _match_record(target):
            if target is None:
                return None
            if isinstance(target, str):
                tgt = target.strip()
                if not tgt:
                    return None
                if tgt.lower().startswith("v0"):
                    for rec in state["volume_records"]:
                        vial = str(rec.get("vial") or "").strip().upper()
                        if vial == "V0":
                            return rec
                try:
                    val = float(tgt)
                except ValueError:
                    return None
            else:
                try:
                    val = float(target)
                except (TypeError, ValueError):
                    return None
            best_rec = None
            best_diff = None
            for rec in state["volume_records"]:
                mw_val = rec.get("mw")
                if mw_val is None or not np.isfinite(mw_val):
                    continue
                diff = abs(float(mw_val) - val)
                if (best_diff is None) or (diff < best_diff):
                    best_diff = diff
                    best_rec = rec
            return best_rec

        def _volume_from_record(rec, prefer: str) -> float | None:
            if rec is None:
                return None
            order_map = {
                "left": ("left_vol", "center_vol", "right_vol"),
                "right": ("right_vol", "center_vol", "left_vol"),
                "center": ("center_vol", "left_vol", "right_vol"),
            }
            keys = order_map.get(prefer.lower(), order_map["center"])
            for key in keys:
                val = rec.get(key)
                if val is not None:
                    return float(val)
            return None

        default_targets = ["6085000", "V0"]
        targets = list(peak_targets) if peak_targets else default_targets
        prefs = list(bound_preferences) if bound_preferences else ["center", "center"]
        while len(targets) < 2:
            targets.append(default_targets[len(targets)])
        while len(prefs) < 2:
            prefs.append("center")

        rec = _match_record(targets[0])
        manual_v0 = _volume_from_record(rec, prefs[0] if prefs else "center")
        if manual_v0 is not None:
            global_v0_volume = manual_v0

        rec = _match_record(targets[1])
        manual_vc = _volume_from_record(rec, prefs[1] if len(prefs) > 1 else "center")
        if manual_vc is not None:
            global_vc_volume = manual_vc

        state["global_v0_volume"] = global_v0_volume
        state["global_vc_volume"] = global_vc_volume
        self.calibration_volumes = {
            "V0_mL": global_v0_volume,
            "Vc_mL": global_vc_volume,
        }
        return self.calibration_volumes

    def apply_calibration_volumes(self, print_summary: bool = False) -> None:
        state = self._require_calibration_state()
        if state.get("global_v0_volume") is None or state.get("global_vc_volume") is None:
            self.determine_calibration_volumes()
            state = self._calibration_state

        prepared = self._prepare_plot_payloads_kav(
            plot_payloads=state.get("plot_payloads", []),
            global_v0_volume=state.get("global_v0_volume"),
            global_vc_volume=state.get("global_vc_volume"),
        )
        state["prepared_plot_payloads"] = prepared
        if print_summary:
            v0 = state.get("global_v0_volume")
            vc = state.get("global_vc_volume")
            window_txt = f"[{v0:.3f}, {vc:.3f}] mL" if v0 is not None and vc is not None else "N/A"
            print(f"Applied calibration volumes across {len(prepared)} chromatograms (window {window_txt}).")

    def build_calibration_tables(self, plot: bool = False) -> pd.DataFrame:
        if not getattr(self, "_calibration_state", None):
            self.integrate_peaks(plot=False)
            self.determine_calibration_volumes()
            self.apply_calibration_volumes()
        state = self._require_calibration_state()
        if not state.get("per_map"):
            return pd.DataFrame()

        if state.get("global_v0_volume") is None or state.get("global_vc_volume") is None:
            self.determine_calibration_volumes()
            state = self._calibration_state

        if "prepared_plot_payloads" not in state:
            self.apply_calibration_volumes()
            state = self._calibration_state

        state["plot_flag"] = plot

        self._reintegrate_after_axis_change(
            state["per_map"],
            state["prepared_plot_payloads"],
            state.get("global_v0_volume"),
            state.get("global_vc_volume"),
        )

        if state.get("plot_flag"):
            state["overview_fig"] = self._init_calibration_overview(
                plot=True,
                entries=state.get("entries_sorted", []),
                height_per_subplot=130,
            )
        else:
            state["overview_fig"] = None

        combined = self._finalize_calibration_output(
            per_map=state["per_map"],
            output_paths=state["output_paths"],
            plots_dir=state.get("plots_dir", self.out_dir / "calibration" / "plots"),
            overview_fig=state.get("overview_fig"),
            plot=state.get("plot_flag", False),
            global_v0_volume=state.get("global_v0_volume"),
            global_vc_volume=state.get("global_vc_volume"),
            plot_payloads=state.get("prepared_plot_payloads", []),
        )

        return combined

    def _sorted_calibration_entries(self) -> list[dict]:
        def _to_dt_local(ts):
            try:
                return pd.to_datetime(ts, format="%Y%m%d %H%M%S", errors="coerce")
            except Exception:
                return pd.NaT

        return sorted(
            self.calibration_set,
            key=lambda x: (_to_dt_local(x.get("inj_ts")) if x.get("inj_ts") else pd.Timestamp.min),
        )

    def _init_calibration_overview(
        self,
        *,
        plot: bool,
        entries: list[dict],
        height_per_subplot: int = 260,
    ):
        if not plot:
            return None
        try:
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=len(entries) or 1,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.06,
                subplot_titles=[e["name"] for e in entries] or ["Calibration overview"],
            )
            total_rows = len(entries) or 1
            fig.update_layout(height=max(360, height_per_subplot * total_rows))
            return fig
        except Exception:
            return None

    def _collect_calibration_peak_data(
        self,
        *,
        entries_sorted: list[dict],
        out_cal: Path,
        plot: bool,
        overview_fig,
    ) -> dict:
        per_map: dict[str, pd.DataFrame] = {}
        output_paths: dict[str, Path] = {}
        plot_payloads: list[dict] = []
        volume_records: list[dict] = []

        def _to_dt_local(ts):
            try:
                return pd.to_datetime(ts, format="%Y%m%d %H%M%S", errors="coerce")
            except Exception:
                return pd.NaT

        for row_idx, entry in enumerate(entries_sorted, start=1):
            name = entry["name"]
            vial_key = entry["vial_key"]
            qdf = entry["quickreport"]
            s_cal = entry["sample"]
            cal_df = (
                self._gpc.from_quick_report_df(qdf, vial_key=vial_key)
                if vial_key
                else self._gpc.from_quick_report_df(qdf)
            )
            dt_cal = _to_dt_local(entry.get("inj_ts"))
            if self.v0s and pd.notna(dt_cal):
                candidates = [b for b in self.v0s if b.get("_dt") is not None and b.get("_dt") <= dt_cal]
                if candidates:
                    v_prev = candidates[-1]
                    try:
                        v_parsed = self._gpc.from_quick_report_df(v_prev.get("quickreport"))
                        v0_df = v_parsed[v_parsed["Vial"].astype(str) == "V0"]
                        if not v0_df.empty:
                            cal_df = cal_df[cal_df["Vial"].astype(str) != "V0"]
                            cal_df = pd.concat([v0_df, cal_df], ignore_index=True)
                    except Exception:
                        pass

            flow_rate = _flow_rate_from_sample(s_cal)
            rid_raw, _ = _get_chromatogram_from_sample(
                s_cal,
                ("chrom_rid_baseline_corrected", "chrom_rid"),
            )
            dad_raw, _ = _get_chromatogram_from_sample(
                s_cal,
                ("chrom_dad_baseline_corrected", "chrom_dad_aligned", "chrom_dad"),
            )
            rid_prepared = _prepare_chromatogram(rid_raw)
            dad_prepared = _prepare_chromatogram(dad_raw)
            rid_time_index = rid_prepared.index if isinstance(rid_prepared, pd.DataFrame) else None
            reference_index = rid_time_index
            if reference_index is None and isinstance(dad_prepared, pd.DataFrame):
                reference_index = dad_prepared.index
            rid_volume = _attach_volume_index(rid_prepared, flow_rate) if isinstance(rid_prepared, pd.DataFrame) else None
            dad_volume = _attach_volume_index(dad_prepared, flow_rate) if isinstance(dad_prepared, pd.DataFrame) else None
            rid_for_ops = rid_volume if isinstance(rid_volume, pd.DataFrame) else rid_prepared
            dad_for_ops = dad_volume if isinstance(dad_volume, pd.DataFrame) else dad_prepared

            regions = _peak_regions_from_quickreport(
                cal_df,
                reference_index=reference_index,
                flow_rate_ml_min=flow_rate,
            )
            region_map = {
                reg["row_index"]: {
                    "center": reg["center"],
                    "left": reg["left"],
                    "right": reg["right"],
                    "center_vol": reg.get("center_vol"),
                    "left_vol": reg.get("left_vol"),
                    "right_vol": reg.get("right_vol"),
                    "mw": reg.get("mw"),
                    "vial": reg.get("vial"),
                }
                for reg in regions
            } if regions else {}
            if "Peak Area (QuickReport)" not in cal_df.columns:
                cal_df["Peak Area (QuickReport)"] = cal_df["Peak Area"]
            if "V_e (mL)" not in cal_df.columns:
                cal_df["V_e (mL)"] = np.nan

            rid_integrations = _integrate_peak_regions(rid_for_ops, regions) if isinstance(rid_for_ops, pd.DataFrame) else {}
            dad_integrations = _integrate_peak_regions(dad_for_ops, regions) if isinstance(dad_for_ops, pd.DataFrame) else {}
            rid_signal = _select_primary_signal_column(rid_for_ops) if isinstance(rid_for_ops, pd.DataFrame) else None
            rid_peak_map = rid_integrations.get(rid_signal) if rid_signal else None
            if rid_peak_map is None and rid_integrations:
                first_col = next(iter(rid_integrations.keys()))
                rid_peak_map = rid_integrations.get(first_col)
                if rid_signal is None:
                    rid_signal = first_col
            if rid_peak_map:
                for df_idx, area in rid_peak_map.items():
                    if df_idx in cal_df.index:
                        cal_df.loc[df_idx, "Peak Area"] = area
            for channel_name, mapping in dad_integrations.items():
                col_label = f"Peak Area [{channel_name}]"
                if col_label not in cal_df.columns:
                    cal_df[col_label] = np.nan
                for df_idx, area in mapping.items():
                    if df_idx in cal_df.index:
                        cal_df.loc[df_idx, col_label] = area

            for reg in regions:
                idx = reg["row_index"]
                if idx not in cal_df.index:
                    continue
                center_vol = reg.get("center_vol")
                if center_vol is not None:
                    cal_df.loc[idx, "V_e (mL)"] = center_vol
                try:
                    mw_val = float(str(cal_df.loc[idx, "MW"]).replace(",", ""))
                except Exception:
                    mw_val = np.nan
                volume_records.append(
                    {
                        "mw": mw_val,
                        "vial": cal_df.loc[idx, "Vial"],
                        "center_vol": center_vol,
                        "left_vol": reg.get("left_vol"),
                        "right_vol": reg.get("right_vol"),
                    }
                )

            peak_payload: dict[str, dict] = {}
            if region_map:
                peak_payload["regions"] = region_map
                s_cal["calibration_peak_regions"] = region_map
            channel_payload: dict[str, dict] = {}
            for integration_map in (rid_integrations, dad_integrations):
                for channel_name, mapping in integration_map.items():
                    channel_payload[channel_name] = {int(idx): float(val) for idx, val in mapping.items()}
            if channel_payload:
                peak_payload["channels"] = channel_payload
            if peak_payload:
                s_cal["peak_integrations"] = peak_payload

            per_map[name] = cal_df
            output_paths[name] = out_cal / f"{name}_CLBRTN.csv"

            if isinstance(rid_for_ops, pd.DataFrame) and not rid_for_ops.empty:
                color = _color_for_sample(name, entry.get("vial_key"))
                signal_col = _select_primary_signal_column(rid_for_ops)
                plot_payloads.append(
                    {
                        "row": row_idx,
                        "name": name,
                        "color": color,
                        "raw_df": rid_for_ops,
                        "regions": regions,
                        "signal_col": signal_col,
                    }
                )

        return {
            "per_map": per_map,
            "output_paths": output_paths,
            "plot_payloads": plot_payloads,
            "volume_records": volume_records,
        }

    def _determine_calibration_volumes(self, volume_records: list[dict]) -> tuple[float | None, float | None]:
        global_v0_volume = None
        global_v0_mw = None
        global_vc_volume = None

        for rec in volume_records:
            mw_val = rec.get("mw")
            center_vol = rec.get("center_vol")
            candidate_vol = center_vol if center_vol is not None else rec.get("right_vol")
            if mw_val is None or not np.isfinite(mw_val) or candidate_vol is None:
                continue
            if (global_v0_mw is None) or (mw_val > global_v0_mw):
                global_v0_mw = mw_val
                global_v0_volume = float(candidate_vol)

        for rec in volume_records:
            vial = str(rec.get("vial") or "").strip().upper()
            if vial != "V0":
                continue
            center_vol = rec.get("center_vol")
            candidate_vol = center_vol if center_vol is not None else rec.get("left_vol")
            if candidate_vol is None:
                continue
            vol_center = float(candidate_vol)
            if (global_vc_volume is None) or (vol_center < global_vc_volume):
                global_vc_volume = vol_center

        return global_v0_volume, global_vc_volume

    def _prepare_plot_payloads_raw(self, plot_payloads: list[dict]) -> list[dict]:
        prepared: list[dict] = []
        if not plot_payloads:
            return prepared
        for payload in plot_payloads:
            raw_df = payload.get("raw_df")
            if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
                continue
            axis_name = str(raw_df.index.name or "")
            use_volume_axis = "volume" in axis_name.lower()
            axis_label = "Volume (mL)" if use_volume_axis else "Time (min)"
            axis_mode = "volume" if use_volume_axis else "time"
            regions_axis: list[dict] = []
            for region in payload.get("regions") or []:
                regions_axis.append(
                    {
                        "row_index": region.get("row_index"),
                        "left_axis": region.get("left_vol") if use_volume_axis else region.get("left"),
                        "right_axis": region.get("right_vol") if use_volume_axis else region.get("right"),
                        "center_axis": region.get("center_vol") if use_volume_axis else region.get("center"),
                        "left_time": region.get("left"),
                        "right_time": region.get("right"),
                        "center_time": region.get("center"),
                        "left_volume": region.get("left_vol"),
                        "right_volume": region.get("right_vol"),
                        "center_volume": region.get("center_vol"),
                    }
                )
            prepared.append(
                {
                    "row": payload.get("row"),
                    "name": payload.get("name"),
                    "color": payload.get("color"),
                    "plot_df": raw_df,
                    "axis_label": axis_label,
                    "axis_mode": axis_mode,
                    "regions_axis": regions_axis,
                    "signal_col": payload.get("signal_col"),
                    "raw_df": payload.get("raw_df"),
                }
            )
        return prepared

    def _prepare_plot_payloads_kav(
        self,
        *,
        plot_payloads: list[dict],
        global_v0_volume: float | None,
        global_vc_volume: float | None,
    ) -> list[dict]:
        if (
            global_v0_volume is None
            or global_vc_volume is None
            or global_v0_volume == global_vc_volume
        ):
            return self._prepare_plot_payloads_raw(plot_payloads)

        prepared: list[dict] = []
        for payload in plot_payloads:
            raw_df = payload.get("raw_df")
            if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
                continue
            axis_name = str(raw_df.index.name or "")
            use_volume_axis = "volume" in axis_name.lower()
            regions = payload.get("regions") or []
            converted_regions: list[dict] = []
            plot_df = raw_df
            axis_label = "Time (min)"
            axis_mode = "time"

            if use_volume_axis:
                attached = _attach_kav_index(plot_df, global_v0_volume, global_vc_volume)
                if isinstance(attached, pd.DataFrame) and not attached.empty:
                    plot_df = attached
                cropped = _crop_by_index_range(plot_df, 0.0, 1.0)
                if isinstance(cropped, pd.DataFrame) and not cropped.empty:
                    plot_df = cropped
                axis_label = "Kav"
                axis_mode = "kav"
                for region in regions:
                    converted_regions.append(
                        {
                            "row_index": region.get("row_index"),
                            "left_axis": _volume_to_kav(region.get("left_vol"), global_v0_volume, global_vc_volume),
                            "right_axis": _volume_to_kav(region.get("right_vol"), global_v0_volume, global_vc_volume),
                            "center_axis": _volume_to_kav(region.get("center_vol"), global_v0_volume, global_vc_volume),
                            "left_time": region.get("left"),
                            "right_time": region.get("right"),
                            "center_time": region.get("center"),
                            "left_volume": region.get("left_vol"),
                            "right_volume": region.get("right_vol"),
                            "center_volume": region.get("center_vol"),
                        }
                    )
            else:
                for region in regions:
                    converted_regions.append(
                        {
                            "row_index": region.get("row_index"),
                            "left_axis": region.get("left"),
                            "right_axis": region.get("right"),
                            "center_axis": region.get("center"),
                            "left_time": region.get("left"),
                            "right_time": region.get("right"),
                            "center_time": region.get("center"),
                            "left_volume": region.get("left_vol"),
                            "right_volume": region.get("right_vol"),
                            "center_volume": region.get("center_vol"),
                        }
                    )

            prepared.append(
                {
                    "row": payload.get("row"),
                    "name": payload.get("name"),
                    "color": payload.get("color"),
                    "plot_df": plot_df,
                    "axis_label": axis_label,
                    "axis_mode": axis_mode,
                    "regions_axis": converted_regions,
                    "signal_col": payload.get("signal_col"),
                    "raw_df": payload.get("raw_df"),
                }
            )

        return prepared

    def _render_plot_payloads(self, payloads: list[dict], overview_fig) -> None:
        if overview_fig is None or not payloads:
            return
        try:
            import plotly.graph_objs as go
        except Exception:
            return

        for payload in payloads:
            plot_df = payload.get("plot_df")
            if not isinstance(plot_df, pd.DataFrame) or plot_df.empty:
                continue
            signal_col = payload.get("signal_col")
            if signal_col is None or signal_col not in plot_df.columns:
                signal_col = _select_primary_signal_column(plot_df)
            if signal_col is None:
                continue
            color = payload.get("color")
            overview_fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[signal_col],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=payload.get("name"),
                    showlegend=False,
                ),
                row=payload.get("row", 1),
                col=1,
            )
            regions_axis = payload.get("regions_axis") or []
            if regions_axis:
                fill_color = _color_with_alpha(color, 0.25)
                for region in regions_axis:
                    left_val = region.get("left_axis")
                    right_val = region.get("right_axis")
                    if left_val is None or right_val is None:
                        continue
                    seg = plot_df.loc[(plot_df.index >= left_val) & (plot_df.index <= right_val), signal_col]
                    if seg.empty:
                        continue
                    overview_fig.add_trace(
                        go.Scatter(
                            x=seg.index,
                            y=seg.values,
                            mode="lines",
                            line=dict(color=fill_color, width=0),
                            fill="tozeroy",
                            fillcolor=fill_color,
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=payload.get("row", 1),
                        col=1,
                    )
                for region in regions_axis:
                    axis_val = region.get("center_axis")
                    if axis_val is None:
                        continue
                    overview_fig.add_vline(
                        x=axis_val,
                        line_width=1,
                        line_dash="dot",
                        line_color="#444444",
                        row=payload.get("row", 1),
                        col=1,
                    )
            overview_fig.update_xaxes(title_text=payload.get("axis_label", "Time (min)"), row=payload.get("row", 1), col=1)

    def _reintegrate_after_axis_change(
        self,
        per_map: dict[str, pd.DataFrame],
        prepared_payloads: list[dict],
        global_v0_volume: float | None,
        global_vc_volume: float | None,
    ) -> None:
        if not prepared_payloads:
            self.peak_metrics = pd.DataFrame()
            return
        global_slope_map = self._compute_logmw_slope_global(per_map, global_v0_volume, global_vc_volume)
        metrics_rows: list[dict] = []
        for payload in prepared_payloads:
            name = payload.get("name")
            cal_df = per_map.get(name)
            if cal_df is None:
                continue
            plot_df = payload.get("plot_df")
            if not isinstance(plot_df, pd.DataFrame) or plot_df.empty:
                continue
            signal_col = payload.get("signal_col")
            if signal_col is None or signal_col not in plot_df.columns:
                signal_col = _select_primary_signal_column(plot_df)
            raw_df = payload.get("raw_df")
            integration = {}
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                raw_signal_col = signal_col if signal_col in raw_df.columns else _select_primary_signal_column(raw_df)
                if raw_signal_col:
                    integration = _integrate_regions_on_axis(
                        raw_df[[raw_signal_col]],
                        payload.get("regions_axis") or [],
                        left_key="left_volume",
                        right_key="right_volume",
                    )
                    if not integration:
                        integration = _integrate_regions_on_axis(
                            raw_df[[raw_signal_col]],
                            payload.get("regions_axis") or [],
                            left_key="left_time",
                            right_key="right_time",
                        )
            if not integration:
                integration = _integrate_regions_on_axis(plot_df, payload.get("regions_axis") or [])
            if integration:
                for col, mapping in integration.items():
                    if col == signal_col:
                        for idx, area in mapping.items():
                            if idx in cal_df.index:
                                cal_df.loc[idx, "Peak Area"] = area
                    else:
                        label = f"Peak Area [{col}]"
                        if label not in cal_df.columns:
                            cal_df[label] = np.nan
                        for idx, area in mapping.items():
                            if idx in cal_df.index:
                                cal_df.loc[idx, label] = area
            sample_metrics = self._collect_peak_metrics_for_payload(
                payload=payload,
                cal_df=cal_df,
                slope_map=global_slope_map,
                global_v0_volume=global_v0_volume,
                global_vc_volume=global_vc_volume,
            )
            metrics_rows.extend(sample_metrics)
        try:
            self.peak_metrics = pd.DataFrame(metrics_rows)
        except Exception:
            self.peak_metrics = pd.DataFrame(metrics_rows or [])

    def _compute_logmw_slope_global(
        self,
        per_map: dict[str, pd.DataFrame],
        global_v0_volume: float | None,
        global_vc_volume: float | None,
    ) -> dict[int, float]:
        frames = []
        for df in per_map.values():
            if isinstance(df, pd.DataFrame):
                working = df.copy()
                if "Kav" in working.columns:
                    kav = pd.to_numeric(working["Kav"], errors="coerce")
                else:
                    kav = None
                if (kav is None or (kav.isna().all() if hasattr(kav, "isna") else True)) and "V_e (mL)" in working.columns:
                    if (
                        global_v0_volume is not None
                        and global_vc_volume is not None
                        and global_vc_volume != global_v0_volume
                    ):
                        ve = pd.to_numeric(working["V_e (mL)"], errors="coerce")
                        kav = (ve - global_v0_volume) / (global_vc_volume - global_v0_volume)
                if kav is None:
                    continue
                temp = pd.DataFrame(
                    {
                        "MW": working.get("MW"),
                        "Kav": kav,
                    }
                )
                frames.append(temp)
        if not frames:
            return {}
        data = pd.concat(frames, copy=True)
        if "MW" not in data.columns or "Kav" not in data.columns:
            return {}
        data["MW"] = pd.to_numeric(data["MW"], errors="coerce")
        data["Kav"] = pd.to_numeric(data["Kav"], errors="coerce")
        mask = data["MW"].notna() & data["Kav"].notna() & (data["MW"] > 0)
        data = data[mask]
        if data.empty:
            return {}
        data = data.assign(logMW=np.log10(data["MW"]))
        data = data.sort_values("Kav")
        kav = data["Kav"].to_numpy(dtype=float)
        logmw = data["logMW"].to_numpy(dtype=float)
        idxs = data.index.to_numpy()
        slopes: dict[int, float] = {}
        n = len(kav)
        for i, idx in enumerate(idxs):
            if n == 1:
                slopes[int(idx)] = np.nan
                continue
            if i == 0:
                denom = kav[i + 1] - kav[i]
                slopes[int(idx)] = (logmw[i + 1] - logmw[i]) / denom if denom else np.nan
            elif i == n - 1:
                denom = kav[i] - kav[i - 1]
                slopes[int(idx)] = (logmw[i] - logmw[i - 1]) / denom if denom else np.nan
            else:
                denom = kav[i + 1] - kav[i - 1]
                slopes[int(idx)] = (logmw[i + 1] - logmw[i - 1]) / denom if denom else np.nan
        return slopes

    @staticmethod
    def _peak_position(df: pd.DataFrame | None, column: str | None, left: float | None, right: float | None) -> float | None:
        if (
            not isinstance(df, pd.DataFrame)
            or column is None
            or column not in df.columns
            or left is None
            or right is None
        ):
            return None
        lo = min(left, right)
        hi = max(left, right)
        try:
            seg = df.loc[(df.index >= lo) & (df.index <= hi), column]
        except Exception:
            return None
        if seg.empty:
            return None
        apex = seg.idxmax()
        try:
            return float(apex)
        except Exception:
            return None

    @staticmethod
    def _interp_half_position(x_vals: np.ndarray, y_vals: np.ndarray, target: float, search_left: bool) -> float | None:
        if x_vals.size < 2 or not np.isfinite(target):
            return None
        if search_left:
            rng = range(len(x_vals) - 1, 0, -1)
        else:
            rng = range(0, len(x_vals) - 1)
        for i in rng:
            idx0 = i - 1 if search_left else i
            idx1 = i if search_left else i + 1
            x0, x1 = x_vals[idx0], x_vals[idx1]
            y0, y1 = y_vals[idx0], y_vals[idx1]
            if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)):
                continue
            if (y0 - target) == 0:
                return x0
            if (y1 - target) == 0:
                return x1
            if (y0 - target) * (y1 - target) <= 0:
                # linear interpolation
                denom = (y1 - y0)
                if denom == 0:
                    return float(x0)
                frac = (target - y0) / denom
                return float(x0 + frac * (x1 - x0))
        return None

    @staticmethod
    def _volume_from_kav(kav: float | None, v0: float | None, vc: float | None) -> float | None:
        if kav is None or v0 is None or vc is None or vc == v0:
            return None
        return float(v0 + kav * (vc - v0))

    @staticmethod
    def _kav_from_volume(volume: float | None, v0: float | None, vc: float | None) -> float | None:
        if volume is None or v0 is None or vc is None or vc == v0:
            return None
        return float((volume - v0) / (vc - v0))

    def _collect_peak_metrics_for_payload(
        self,
        *,
        payload: dict,
        cal_df: pd.DataFrame,
        slope_map: dict[int, float],
        global_v0_volume: float | None,
        global_vc_volume: float | None,
    ) -> list[dict]:
        plot_df = payload.get("plot_df")
        if not isinstance(plot_df, pd.DataFrame) or plot_df.empty:
            return []
        signal_col = payload.get("signal_col")
        if signal_col is None or signal_col not in plot_df.columns:
            signal_col = _select_primary_signal_column(plot_df)
        if signal_col is None:
            return []
        axis_mode = str(payload.get("axis_mode", "time")).lower()
        raw_df = payload.get("raw_df")
        raw_signal_col = signal_col if isinstance(raw_df, pd.DataFrame) and signal_col in raw_df.columns else None
        if raw_signal_col is None and isinstance(raw_df, pd.DataFrame):
            raw_signal_col = _select_primary_signal_column(raw_df)
        results: list[dict] = []
        for region in payload.get("regions_axis") or []:
            row_idx = region.get("row_index")
            left = region.get("left_axis")
            right = region.get("right_axis")
            if row_idx is None or left is None or right is None:
                continue
            try:
                seg = plot_df.loc[(plot_df.index >= left) & (plot_df.index <= right), signal_col]
            except Exception:
                continue
            if seg.empty:
                continue
            x_vals = seg.index.to_numpy(dtype=float)
            y_vals = seg.to_numpy(dtype=float)
            if x_vals.size == 0 or not np.isfinite(y_vals).any():
                continue
            max_idx = int(np.nanargmax(y_vals))
            apex_axis = float(x_vals[max_idx])
            apex_height = float(y_vals[max_idx])
            half_height = apex_height / 2.0
            left_half = self._interp_half_position(x_vals[: max_idx + 1], y_vals[: max_idx + 1], half_height, search_left=True)
            right_half = self._interp_half_position(x_vals[max_idx:], y_vals[max_idx:], half_height, search_left=False)
            if left_half is None:
                left_half = float(x_vals[0])
            if right_half is None:
                right_half = float(x_vals[-1])
            fwhm_axis = float(right_half - left_half) if np.isfinite(right_half - left_half) else np.nan
            delta_axis = fwhm_axis / 2.355 if np.isfinite(fwhm_axis) else np.nan
            symmetry = np.nan
            denom = apex_axis - left_half
            if denom not in (None, 0):
                symmetry = (right_half - apex_axis) / denom
            kav_val = None
            volume_val = None
            time_val = None
            if axis_mode == "kav":
                kav_val = apex_axis
                volume_val = self._volume_from_kav(kav_val, global_v0_volume, global_vc_volume)
            elif axis_mode == "volume":
                volume_val = apex_axis
                kav_val = self._kav_from_volume(volume_val, global_v0_volume, global_vc_volume)
            else:
                time_val = apex_axis
            # refine using raw data if available
            if isinstance(raw_df, pd.DataFrame) and raw_signal_col in raw_df.columns:
                vol_apex = self._peak_position(raw_df, raw_signal_col, region.get("left_volume"), region.get("right_volume"))
                if vol_apex is not None:
                    volume_val = vol_apex
                    kav_val = self._kav_from_volume(volume_val, global_v0_volume, global_vc_volume)
                time_apex = self._peak_position(raw_df, raw_signal_col, region.get("left_time"), region.get("right_time"))
                if time_apex is not None:
                    time_val = time_apex
            span = None
            if (
                global_v0_volume is not None
                and global_vc_volume is not None
                and global_vc_volume != global_v0_volume
            ):
                span = global_vc_volume - global_v0_volume
            fwhm_kav = None
            fwhm_volume = None
            if np.isfinite(fwhm_axis):
                if axis_mode == "kav":
                    fwhm_kav = fwhm_axis
                    if span is not None:
                        fwhm_volume = fwhm_kav * span
                elif axis_mode == "volume":
                    fwhm_volume = fwhm_axis
                    if span is not None:
                        fwhm_kav = fwhm_volume / span
                else:
                    # time axis; leave conversions as None
                    pass
            delta_kav = None
            if np.isfinite(delta_axis):
                if axis_mode == "kav":
                    delta_kav = delta_axis
                elif axis_mode == "volume" and span is not None:
                    delta_kav = delta_axis / span
            mw_val = cal_df.loc[row_idx]["MW"] if row_idx in cal_df.index and "MW" in cal_df.columns else np.nan
            try:
                mw_val = float(str(mw_val).replace(",", ""))
            except Exception:
                mw_val = np.nan
            vial_val = cal_df.loc[row_idx]["Vial"] if row_idx in cal_df.index and "Vial" in cal_df.columns else None
            slope_val = slope_map.get(row_idx)
            rsp = np.nan
            if delta_kav and slope_val not in (None, 0):
                rsp = 0.25 / (abs(slope_val) * delta_kav) if delta_kav > 0 else np.nan
            results.append(
                {
                    "sample": payload.get("name"),
                    "row_index": row_idx,
                    "MW": mw_val,
                    "Vial": vial_val,
                    "apex_axis": apex_axis,
                    "axis_mode": axis_mode,
                    "axis_label": payload.get("axis_label"),
                    "apex_height": apex_height,
                    "apex_kav": kav_val,
                    "apex_volume_mL": volume_val,
                    "apex_time_min": time_val,
                    "fwhm_axis": fwhm_axis,
                    "fwhm_kav": fwhm_kav,
                    "fwhm_volume_mL": fwhm_volume,
                    "delta_axis": delta_axis,
                    "delta_kav": delta_kav,
                    "symmetry": symmetry,
                    "logmw_slope": slope_val,
                    "Rsp": rsp,
                }
            )
        return results

    def _finalize_calibration_output(
        self,
        *,
        per_map: dict[str, pd.DataFrame],
        output_paths: dict[str, Path],
        plots_dir: Path,
        overview_fig,
        plot: bool,
        global_v0_volume: float | None,
        global_vc_volume: float | None,
        plot_payloads: list[dict],
    ) -> pd.DataFrame:
        denominator = None
        if (
            global_v0_volume is not None
            and global_vc_volume is not None
            and global_v0_volume != global_vc_volume
        ):
            denominator = global_vc_volume - global_v0_volume

        per_tables: list[pd.DataFrame] = []
        for name, cal_df in per_map.items():
            if "V_e (mL)" not in cal_df.columns:
                cal_df["V_e (mL)"] = np.nan
            cal_df["V_0 (mL)"] = global_v0_volume if global_v0_volume is not None else np.nan
            cal_df["V_c (mL)"] = global_vc_volume if global_vc_volume is not None else np.nan
            if denominator:
                cal_df["Kav"] = (cal_df["V_e (mL)"] - global_v0_volume) / denominator
            else:
                cal_df["Kav"] = np.nan
            per_tables.append(cal_df.assign(_source=name))
            out_path = output_paths.get(name)
            if out_path is not None:
                self._gpc.write_output(cal_df, str(out_path))

        self.calibration_volumes = {
            "V0_mL": global_v0_volume,
            "Vc_mL": global_vc_volume,
        }

        if plot and overview_fig is not None and plot_payloads:
            self._render_plot_payloads(plot_payloads, overview_fig)

        combined = _combine_calibration_rows(per_tables)
        self.per_calibration_tables = per_map
        self.combined_calibration = combined

        sets_payload = {
            "calibration_set": [e["name"] for e in self.calibration_set],
            "sample_set": [e["name"] for e in self.sample_set],
        }
        (self.out_dir / "sets.json").write_text(json.dumps(sets_payload, indent=2))

        if plot and overview_fig is not None:
            plots_dir.mkdir(parents=True, exist_ok=True)
            try:
                import plotly.io as pio

                pio.write_html(
                    overview_fig,
                    str(plots_dir / "calibration_overview.html"),
                    include_plotlyjs="cdn",
                    auto_open=False,
                )
                if _ipython_display is not None:
                    try:
                        _ipython_display(overview_fig)
                    except Exception:
                        pass
            except Exception:
                pass

        return combined

    def run(self, plot: bool = False) -> pd.DataFrame:
        self.detect_sets()
        self.align_v0(plot=plot)
        self.baseline_correct_chromatograms(plot=plot)
        self.integrate_peaks(plot=plot)
        self.determine_calibration_volumes()
        self.apply_calibration_volumes()
        return self.build_calibration_tables(plot=plot)

    # ---- Step 3: fit calibration curves (exp decay) ----
    def fit_calibration_curves(self, plot: bool = False,
                               upper_bound_da: float = 2_000_000.0,
                               lower_bound_candidates: tuple[float, float] = (162.0, 200.0),
                               iv_upper_bound: float | None = None,
                               iv_lower_bound_candidates: tuple[float, ...] = (0.03, 0.05, 0.1)) -> None:
        """
        Fit a single calibration curve over the combined calibration table.
        Model: log(value) = a + b * x (exponential decay in MW or IV vs elution axis),
        where x defaults to Kav (if present) otherwise retention time.
        - For MW (Mp): filters to MW <= upper_bound_da and MW >= chosen lower bound.
        - For IV: filters to supplied upper bound (optional) and lower bound candidates.
        - Tries the provided lower-bound candidates for each quantity, keeps the one with lowest log-RMSE.
        - Stores MW fit in self.calibration_fit (backwards compatible) and both fits in self.calibration_fits.
        - Optional plotting: two subplots per quantity (value/log(value) vs chosen axis) mirroring previous behaviour.
        """
        import numpy as _np
        try:
            import plotly.graph_objs as go
            from plotly.subplots import make_subplots
        except Exception:
            go = None
            make_subplots = None

        # Ensure combined calibration is available
        if self.combined_calibration is None or self.combined_calibration.empty:
            self.build_calibration_tables()
        df = self.combined_calibration
        if df is None or df.empty:
            self.calibration_fit = None
            return

        if "Kav" in df.columns:
            x_column = "Kav"
            x_label = "Kav"
        else:
            x_column = "Exp. RT (min)"
            x_label = "Time (min.)"

        def _prepare_numeric(column: str) -> pd.DataFrame:
            if column not in df.columns:
                return pd.DataFrame(columns=[x_column, column])
            data = df[[x_column, column]].copy()
            data[x_column] = pd.to_numeric(data[x_column], errors="coerce")
            data[column] = pd.to_numeric(data[column], errors="coerce")
            mask = _np.isfinite(data[x_column]) & _np.isfinite(data[column]) & (data[column] > 0)
            return data[mask]

        def _fit_log_curve(value_col: str,
                           label: str,
                           upper_bound: float | None,
                           lower_candidates: tuple[float, ...] | None) -> dict | None:
            data_all = _prepare_numeric(value_col)
            if data_all.empty:
                return None
            if upper_bound is not None:
                data = data_all[data_all[value_col] <= float(upper_bound)]
            else:
                data = data_all.copy()
            if data.empty:
                return None
            lb_candidates = lower_candidates if lower_candidates else (float(data[value_col].min()),)
            best_fit = None
            for lb in lb_candidates:
                sub = data[data[value_col] >= float(lb)]
                if len(sub) < 2:
                    continue
                x = _np.asarray(sub[x_column], dtype=float)
                y = _np.asarray(sub[value_col], dtype=float)
                logy = _np.log(y)
                X = _np.vstack([_np.ones_like(x), x]).T
                try:
                    beta, *_ = _np.linalg.lstsq(X, logy, rcond=None)
                    a, b = float(beta[0]), float(beta[1])
                except Exception:
                    continue
                logy_hat = a + b * x
                rmse = float(_np.sqrt(_np.mean((logy - logy_hat) ** 2)))
                cand = {
                    "a": a,
                    "b": b,
                    "lower_bound": float(lb),
                    "upper_bound": float(upper_bound) if upper_bound is not None else float(data[value_col].max()),
                    "rmse": rmse,
                    "n": int(len(sub)),
                    "x_min": float(_np.min(x)),
                    "x_max": float(_np.max(x)),
                    "model": f"log({label}) = a + b * {x_label}",
                    "value_column": value_col,
                    "label": label,
                }
                if (best_fit is None) or (cand["rmse"] < best_fit["rmse"]):
                    best_fit = cand

            if best_fit is None:
                return None

            if upper_bound is not None:
                try:
                    base = data[data[value_col] >= best_fit["lower_bound"]].copy()
                    xw = _np.asarray(base[x_column], dtype=float)
                    yw = _np.asarray(base[value_col], dtype=float)
                    candidates = data_all[data_all[value_col] > best_fit["upper_bound"]].sort_values(value_col)
                    a, b = best_fit["a"], best_fit["b"]
                    for _, row in candidates.iterrows():
                        rt = float(row[x_column])
                        val = float(row[value_col])
                        if not _np.isfinite(rt) or not _np.isfinite(val) or val <= 0:
                            continue
                        pred = float(_np.exp(a + b * rt))
                        perc_err = abs(pred - val) / val * 100.0
                        if perc_err <= 3.0:
                            xw = _np.concatenate([xw, _np.array([rt])])
                            yw = _np.concatenate([yw, _np.array([val])])
                            logy = _np.log(yw)
                            Xw = _np.vstack([_np.ones_like(xw), xw]).T
                            beta, *_ = _np.linalg.lstsq(Xw, logy, rcond=None)
                            a, b = float(beta[0]), float(beta[1])
                            logy_hat = a + b * xw
                            rmse = float(_np.sqrt(_np.mean((logy - logy_hat) ** 2)))
                            best_fit.update({
                                "a": a,
                                "b": b,
                                "upper_bound": val,
                                "rmse": rmse,
                                "n": int(len(xw)),
                                "x_min": float(_np.min(xw)),
                                "x_max": float(_np.max(xw)),
                            })
                        else:
                            break
                except Exception:
                    pass
            return best_fit

        def _plot_fit_curve(fit: dict | None,
                            value_col: str,
                            y_label: str,
                            log_y_label: str,
                            html_name: str) -> None:
            if not fit or go is None or make_subplots is None:
                return
            a, b = fit["a"], fit["b"]
            # Determine plotting range
            x_ranges = []
            for _, tdf in self.per_calibration_tables.items():
                if value_col not in tdf.columns:
                    continue
                times = pd.to_numeric(tdf.get(x_column, tdf.get("Exp. RT (min)")), errors="coerce")
                vals = pd.to_numeric(tdf[value_col], errors="coerce")
                mask = _np.isfinite(times) & _np.isfinite(vals) & (vals > 0)
                if mask.any():
                    arr = times[mask].to_numpy(dtype=float)
                    x_ranges.append((arr.min(), arr.max()))
            if x_ranges:
                x_min = float(min(v[0] for v in x_ranges))
                x_max = float(max(v[1] for v in x_ranges))
            else:
                x_min = float(fit["x_min"])
                x_max = float(fit["x_max"])
            xs = _np.linspace(x_min, x_max, 200)
            ys = _np.exp(a + b * xs)

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=(f"{y_label} vs {x_label} (combined)", f"{log_y_label} vs {x_label} (combined)"))
            for name, tdf in self.per_calibration_tables.items():
                if value_col not in tdf.columns:
                    continue
                try:
                    vial_key = None
                    for e in self.calibration_set:
                        if e["name"] == name:
                            vial_key = e.get("vial_key")
                            break
                    color = _color_for_sample(name, vial_key)
                    times = pd.to_numeric(tdf.get(x_column, tdf.get("Exp. RT (min)")), errors="coerce")
                    vals = pd.to_numeric(tdf[value_col], errors="coerce")
                    mask = _np.isfinite(times) & _np.isfinite(vals) & (vals > 0)
                    x = times[mask].to_numpy(dtype=float)
                    y = vals[mask].to_numpy(dtype=float)
                    if x.size == 0:
                        continue
                    logy = _np.log(y)
                    logy_hat = a + b * x
                    err = _np.abs(logy - logy_hat)
                    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=color), name=name), row=1, col=1)
                    fig.add_trace(go.Scatter(x=x, y=logy, mode="markers", marker=dict(color=color),
                                             error_y=dict(type="data", array=err, visible=True),
                                             name=name, showlegend=False), row=1, col=2)
                except Exception:
                    continue

            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Fit", line=dict(color="#222222", width=2)),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=(a + b * xs), mode="lines", name="log Fit",
                                     line=dict(color="#222222", width=2), showlegend=False),
                          row=1, col=2)
            for col_idx in (1, 2):
                for bound in (fit["x_min"], fit["x_max"]):
                    fig.add_vline(x=bound, line_width=1, line_dash="dash", line_color="gray", row=1, col=col_idx)
            fig.update_xaxes(title_text=x_label, row=1, col=1)
            fig.update_yaxes(title_text=y_label, row=1, col=1)
            fig.update_xaxes(title_text=x_label, row=1, col=2)
            fig.update_yaxes(title_text=log_y_label, row=1, col=2)
            ub_val = fit.get("upper_bound")
            lb_str = f"{fit['lower_bound']:.4g}"
            ub_str = f"{ub_val:.4g}" if (ub_val is not None and _np.isfinite(ub_val)) else "n/a"
            fig.update_layout(title=f"{y_label} calibration fit | lower={lb_str} | upper={ub_str} | rmse={fit['rmse']:.4f}")
            try:
                if _ipython_display is not None:
                    _ipython_display(fig)
            except Exception:
                pass
            try:
                import plotly.io as pio
                plots_dir = self.out_dir / "calibration" / "fit_plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                pio.write_html(fig, str(plots_dir / html_name), include_plotlyjs="cdn", auto_open=False)
            except Exception:
                pass

        mw_fit = _fit_log_curve("MW", "MW (Da)", upper_bound_da, lower_bound_candidates)
        iv_fit = _fit_log_curve("iv_dl_per_g", "IV (dL/g)", iv_upper_bound, iv_lower_bound_candidates)

        self.calibration_fit = mw_fit
        fits: dict[str, dict] = {}
        if mw_fit:
            fits["combined"] = mw_fit
            fits["mw"] = mw_fit
        if iv_fit:
            fits["iv_dl_per_g"] = iv_fit
        self.calibration_fits = fits

        if plot:
            _plot_fit_curve(mw_fit, "MW", "MW (Da)", "log(MW)", "combined_fit_mw.html")
            _plot_fit_curve(iv_fit, "iv_dl_per_g", "IV (dL/g)", "log(IV)", "combined_fit_iv.html")
            def _plot_rsp_curve():
                if go is None:
                    return
                peak_df = getattr(self, "peak_metrics", None)
                if peak_df is None or peak_df.empty:
                    return
                data = peak_df.copy()
                try:
                    data["MW"] = pd.to_numeric(data["MW"], errors="coerce")
                    data["Rsp"] = pd.to_numeric(data["Rsp"], errors="coerce")
                except Exception:
                    return
                mask = data["MW"].notna() & data["Rsp"].notna() & (data["Rsp"] > 0)
                if not mask.any():
                    return
                plot_data = data.loc[mask].sort_values("MW")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=plot_data["MW"],
                        y=plot_data["Rsp"],
                        mode="lines+markers",
                        text=plot_data["sample"],
                        hovertemplate="MW=%{x:.3g}<br>Rsp=%{y:.3f}<extra></extra>",
                    )
                )
                fig.update_xaxes(title_text="MW (Da)", type="log")
                fig.update_yaxes(title_text="Specific Resolution (Rsp)")
                fig.update_layout(title="Specific Resolution vs MW")
                try:
                    if _ipython_display is not None:
                        _ipython_display(fig)
                except Exception:
                    pass
                try:
                    import plotly.io as pio
                    plots_dir = self.out_dir / "calibration" / "fit_plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    pio.write_html(fig, str(plots_dir / "resolution_curve.html"), include_plotlyjs="cdn", auto_open=False)
                except Exception:
                    pass
            _plot_rsp_curve()

        # ---- Area vs Mass linear calibration (robust to outliers) ----
        # Build combined dataset of (Mass, Peak Area) with sample names
        if not self.per_calibration_tables:
            self.build_calibration_tables()
        import numpy as _np
        import pandas as _pd
        rows = []
        for name, df_part in self.per_calibration_tables.items():
            try:
                part = df_part.copy()
                area_col = "Peak Area" if "Peak Area" in part.columns else "Peak Area (QuickReport)"
                part["Mass"] = _pd.to_numeric(part["Mass"], errors="coerce")
                part["_AreaFit"] = _pd.to_numeric(part[area_col], errors="coerce")
                part["MW_numeric"] = _pd.to_numeric(part.get("MW"), errors="coerce")
                part = part[_np.isfinite(part["Mass"]) & _np.isfinite(part["_AreaFit"])]
                part = part[(part["Mass"] > 0) & (part["_AreaFit"] > 0)]
                part = part[~_np.isclose(part["MW_numeric"], 162.0, rtol=0, atol=1e-6)]
                if part.empty:
                    continue
                vial_key = None
                for e in self.calibration_set:
                    if e['name'] == name:
                        vial_key = e.get('vial_key'); break
                for _, r in part.iterrows():
                    rows.append({'name': name, 'vial_key': vial_key,
                                 'Mass': float(r['Mass']), 'Area': float(r['_AreaFit'])})
            except Exception:
                continue
        if rows:
            adat = _pd.DataFrame(rows)
        else:
            adat = _pd.DataFrame(columns=['name','vial_key','Mass','Area'])

        if not adat.empty:
            x = _np.asarray(adat['Mass'], dtype=float)
            y = _np.asarray(adat['Area'], dtype=float)
            X = _np.vstack([_np.ones_like(x), x]).T
            try:
                beta, *_ = _np.linalg.lstsq(X, y, rcond=None)
                alpha, slope = float(beta[0]), float(beta[1])
            except Exception:
                alpha, slope = 0.0, 0.0
            resid = y - (alpha + slope * x)
            rmse = float(_np.sqrt(_np.mean(resid**2))) if y.size > 0 else 0.0
            inlier_mask = _np.ones_like(y, dtype=bool)

            self.area_fit = {
                'alpha': alpha,
                'slope': slope,
                'rmse': rmse,
                'n_used': int(inlier_mask.sum()),
                'n_outliers': int((~inlier_mask).sum()),
                'model': 'Area = alpha + slope * Mass'
            }

            # Optional plotting: points colored by sample, outliers marked, fit line
            if plot:
                try:
                    from plotly.subplots import make_subplots
                    import plotly.graph_objs as go
                    # Mass range for line
                    x_min = float(x.min()) if x.size else 0.0
                    x_max = float(x.max()) if x.size else 1.0
                    xs = _np.linspace(max(0.0, x_min*0.9), x_max*1.1, 200)
                    ys = alpha + slope * xs
                    fig = make_subplots(rows=1, cols=1, subplot_titles=("Area vs Mass (combined)",))
                    # per-sample points
                    for name in sorted(adat['name'].unique()):
                        sub = adat[adat['name'] == name]
                        vial_key = None
                        for e in self.calibration_set:
                            if e['name'] == name:
                                vial_key = e.get('vial_key'); break
                        color = _color_for_sample(name, vial_key)
                        xi = sub['Mass'].to_numpy(dtype=float)
                        yi = sub['Area'].to_numpy(dtype=float)
                        if xi.size:
                            fig.add_trace(go.Scatter(x=xi, y=yi, mode='markers', marker=dict(color=color),
                                                     name=name))
                    # fit line
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='Fit', line=dict(color='#222222', width=2)))
                    fig.update_xaxes(title_text='Mass (mg)')
                    fig.update_yaxes(title_text='Peak Area')
                    fig.update_layout(title=f"Area calibration | alpha={alpha:.3g}, slope={slope:.3g}, rmse={rmse:.3g}")
                    if _ipython_display is not None:
                        try:
                            _ipython_display(fig)
                        except Exception:
                            pass
                    try:
                        import plotly.io as pio
                        plots_dir = self.out_dir / "calibration" / "fit_plots"
                        plots_dir.mkdir(parents=True, exist_ok=True)
                        pio.write_html(fig, str(plots_dir / f"area_fit.html"), include_plotlyjs='cdn', auto_open=False)
                    except Exception:
                        pass
                except Exception:
                    pass

    # ---- Step 4: add MW index column to chromatograms ----
    def add_mw_index_to_chromatograms(self, column_name: str = "MW PS Equivalent (Da)") -> None:
        """
        Using the fitted combined calibration curve, compute MW for each RT in both RID and DAD chromatograms
        and add it as a new column named `column_name`. Leaves the existing time index intact.
        Requires that `fit_calibration_curves` has been run to populate `self.calibration_fit`.
        """
        import numpy as _np
        if not self.calibration_fit:
            # Ensure a fit is available
            self.fit_calibration_curves(plot=False)
        fit = self.calibration_fit
        if not fit:
            return
        a, b = float(fit['a']), float(fit['b'])

        def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            if df.index.name is None or 'time' not in str(df.index.name).lower():
                return df.set_index(df.columns[0])
            return df

        for s in self.samples:
            # RID (prefer baseline-corrected)
            if isinstance(s.get('chrom_rid_baseline_corrected'), pd.DataFrame) and not s['chrom_rid_baseline_corrected'].empty:
                rid = s['chrom_rid_baseline_corrected']
                rid_key = 'chrom_rid_baseline_corrected'
            else:
                rid = s.get('chrom_rid')
                rid_key = 'chrom_rid'
            if isinstance(rid, pd.DataFrame) and not rid.empty:
                rid_idx = _ensure_time_index(rid)
                try:
                    rt = _np.asarray(rid_idx.index, dtype=float)
                    mw = _np.exp(a + b * rt)
                    rid_idx[column_name] = mw
                    s[rid_key] = rid_idx
                except Exception:
                    pass
            # DAD (prefer aligned)
            if isinstance(s.get('chrom_dad_baseline_corrected'), pd.DataFrame) and not s['chrom_dad_baseline_corrected'].empty:
                dad = s['chrom_dad_baseline_corrected']
                key = 'chrom_dad_baseline_corrected'
            elif isinstance(s.get('chrom_dad_aligned'), pd.DataFrame) and not s['chrom_dad_aligned'].empty:
                dad = s['chrom_dad_aligned']
                key = 'chrom_dad_aligned'
            else:
                dad = s.get('chrom_dad')
                key = 'chrom_dad'
            if isinstance(dad, pd.DataFrame) and not dad.empty:
                dad_idx = _ensure_time_index(dad)
                try:
                    rt = _np.asarray(dad_idx.index, dtype=float)
                    mw = _np.exp(a + b * rt)
                    dad_idx[column_name] = mw
                    s[key] = dad_idx
                except Exception:
                    pass
            try:
                import plotly.io as pio
                plots_dir = self.out_dir / "calibration" / "fit_plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                pio.write_html(fig, str(plots_dir / f"combined_fit.html"), include_plotlyjs='cdn', auto_open=False)
            except Exception:
                pass
