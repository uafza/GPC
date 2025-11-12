# workflow_gpc.py
from __future__ import annotations
from pathlib import Path
import re
import json
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


def _rid_plot_with_annotations(sample: dict, cal_df: pd.DataFrame, title: str):
    try:
        import plotly.graph_objs as go
    except Exception:
        return None
    rid = sample.get("chrom_rid")
    if rid is None or rid.empty:
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
      combined = wf.build_calibration_tables(plot=True)  # write per-sample + combined tables and plots

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

    def build_calibration_tables(self, plot: bool = False) -> pd.DataFrame:
        if not self.calibration_set:
            self.detect_sets()
        out = self.out_dir
        out_cal = out / "calibration" / "per_cal_sample"
        out_cal.mkdir(parents=True, exist_ok=True)
        plots_dir = out / "calibration" / "plots"
        if plot:
            plots_dir.mkdir(parents=True, exist_ok=True)

        per_tables: list[pd.DataFrame] = []
        per_map: dict[str, pd.DataFrame] = {}
        def _to_dt_local(ts):
            try:
                return pd.to_datetime(ts, format="%Y%m%d %H%M%S", errors="coerce")
            except Exception:
                return pd.NaT
        entries_sorted = sorted(self.calibration_set, key=lambda x: (_to_dt_local(x.get("inj_ts")) if x.get("inj_ts") else pd.Timestamp.min))

        # Prepare a single overview figure if plotting is requested
        overview_fig = None
        if plot:
            try:
                from plotly.subplots import make_subplots
                import plotly.graph_objs as go
                overview_fig = make_subplots(rows=len(entries_sorted) or 1, cols=1, shared_xaxes=False,
                                             vertical_spacing=0.06,
                                             subplot_titles=[e["name"] for e in entries_sorted] or ["Calibration overview"])
            except Exception:
                overview_fig = None

        for row_idx, entry in enumerate(entries_sorted, start=1):
            name = entry["name"]; vial_key = entry["vial_key"]; qdf = entry["quickreport"]; s_cal = entry["sample"]
            cal_df = self._gpc.from_quick_report_df(qdf, vial_key=vial_key) if vial_key else self._gpc.from_quick_report_df(qdf)
            # prepend nearest preceding V0 row if present
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
            per_tables.append(cal_df.assign(_source=name))
            per_map[name] = cal_df
            self._gpc.write_output(cal_df, str(out_cal / f"{name}_CLBRTN.csv"))
            # Add to single overview figure instead of per-sample figures
            if plot and overview_fig is not None:
                try:
                    import plotly.graph_objs as go
                    rid = s_cal.get("chrom_rid")
                    if isinstance(rid, pd.DataFrame) and not rid.empty:
                        if rid.index.name is None or "time" not in str(rid.index.name).lower():
                            rid = rid.set_index(rid.columns[0])
                        # pick RID signal column
                        col = None
                        for c in rid.columns:
                            if str(c).startswith("Signal (") and str(c).endswith(")"):
                                col = c; break
                        if col is None:
                            col = rid.columns[0]
                        # trace
                        color = _color_for_sample(name, entry.get('vial_key'))
                        overview_fig.add_trace(go.Scatter(x=rid.index, y=rid[col], mode="lines",
                                                          line=dict(color=color, width=2),
                                                          name=name, showlegend=False),
                                               row=row_idx, col=1)
                        # annotations (vertical dashed lines)
                        for _, r in cal_df.iterrows():
                            rt = float(r["Exp. RT (min)"])
                            overview_fig.add_vline(x=rt, line_width=1, line_dash='dot', line_color='#444444', row=row_idx, col=1)
                except Exception:
                    pass
        combined = _combine_calibration_rows(per_tables)
        # keep in memory instead of writing a separate combined file
        self.per_calibration_tables = per_map
        self.combined_calibration = combined
        print(f"Combined calibration rows: {len(combined)}")
        # Save sets
        sets_payload = {
            "calibration_set": [e["name"] for e in self.calibration_set],
            "sample_set": [e["name"] for e in self.sample_set],
        }
        (out / "sets.json").write_text(json.dumps(sets_payload, indent=2))
        print("Wrote set membership ->", out / "sets.json")
        # finalize plotting: write a single overview file
        if plot and overview_fig is not None:
            try:
                import plotly.io as pio
                pio.write_html(overview_fig, str(plots_dir / "calibration_overview.html"), include_plotlyjs='cdn', auto_open=False)
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
        return self.build_calibration_tables(plot=plot)

    # ---- Step 3: fit calibration curves (exp decay) ----
    def fit_calibration_curves(self, plot: bool = False,
                               upper_bound_da: float = 2_000_000.0,
                               lower_bound_candidates: tuple[float, float] = (162.0, 200.0)) -> None:
        """
        Fit a single calibration curve over the combined calibration table.
        Model: log(MW) = a + b * RT (exponential decay in MW vs RT).
        - Filters to MW <= upper_bound_da and MW >= chosen lower bound.
        - Tries lower bounds (162, 200) and selects the one with lower RMSE (log-space).
        - Stores result in self.calibration_fit.
        - Optional plotting: two subplots (MW vs RT, log(MW) vs RT) with fitted curve,
          residual error bars on log plot, and vertical dashed lines at RT bounds.
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
            self.build_calibration_tables(plot=False)
        df = self.combined_calibration
        if df is None or df.empty:
            self.calibration_fit = None
            return

        # Select rows with MW (exclude V0) and apply bounds
        tbl = df.copy()
        tbl = tbl[~tbl['MW'].astype(str).eq("")].copy()
        tbl['MW'] = _np.asarray(tbl['MW'], dtype=float)
        tbl['Exp. RT (min)'] = _np.asarray(tbl['Exp. RT (min)'], dtype=float)
        tbl = tbl[_np.isfinite(tbl['MW']) & (tbl['MW'] <= float(upper_bound_da))]
        if tbl.empty:
            self.calibration_fit = None
            return

        best = None
        for lb in lower_bound_candidates:
            sub = tbl[tbl['MW'] >= float(lb)].copy()
            if len(sub) < 2:
                continue
            x = _np.asarray(sub['Exp. RT (min)'], dtype=float)
            y = _np.asarray(sub['MW'], dtype=float)
            logy = _np.log(y)
            X = _np.vstack([_np.ones_like(x), x]).T
            try:
                beta, *_ = _np.linalg.lstsq(X, logy, rcond=None)
                a, b = float(beta[0]), float(beta[1])
            except Exception:
                continue
            logy_hat = a + b * x
            rmse = float(_np.sqrt(_np.mean((logy - logy_hat)**2)))
            cand = {
                'a': a, 'b': b,
                'lower_bound': float(lb), 'upper_bound': float(upper_bound_da),
                'rmse': rmse,
                'n': int(len(sub)),
                'rt_min': float(_np.min(x)), 'rt_max': float(_np.max(x)),
                'model': 'logMW = a + b * RT',
            }
            if (best is None) or (cand['rmse'] < best['rmse']):
                best = cand

        # Attempt to extend upper bound by admitting higher-MW points with small percent error
        if best is not None:
            try:
                # base dataset at chosen lower bound (initial UB already applied in tbl)
                base = tbl[tbl['MW'] >= best['lower_bound']].copy()
                xw = _np.asarray(base['Exp. RT (min)'], dtype=float)
                yw = _np.asarray(base['MW'], dtype=float)
                # candidates from the full combined table above current UB, sorted by MW ascending
                all_tbl = df.copy()
                all_tbl = all_tbl[~all_tbl['MW'].astype(str).eq("")].copy()
                all_tbl['MW'] = _np.asarray(all_tbl['MW'], dtype=float)
                all_tbl['Exp. RT (min)'] = _np.asarray(all_tbl['Exp. RT (min)'], dtype=float)
                candidates = all_tbl[_np.isfinite(all_tbl['MW']) & (all_tbl['MW'] > best['upper_bound'])]
                candidates = candidates.sort_values('MW')
                # current fit params
                a, b = best['a'], best['b']
                for _, row in candidates.iterrows():
                    rt = float(row['Exp. RT (min)']); mw = float(row['MW'])
                    if not _np.isfinite(rt) or not _np.isfinite(mw) or mw <= 0:
                        continue
                    mw_hat = float(_np.exp(a + b * rt))
                    perc_err = abs(mw_hat - mw) / mw * 100.0
                    if perc_err <= 3.0:
                        # include point and refit in log-space
                        xw = _np.concatenate([xw, _np.array([rt])])
                        yw = _np.concatenate([yw, _np.array([mw])])
                        logy = _np.log(yw)
                        Xw = _np.vstack([_np.ones_like(xw), xw]).T
                        beta, *_ = _np.linalg.lstsq(Xw, logy, rcond=None)
                        a, b = float(beta[0]), float(beta[1])
                        logy_hat = a + b * xw
                        rmse = float(_np.sqrt(_np.mean((logy - logy_hat)**2)))
                        best.update({
                            'a': a, 'b': b,
                            'upper_bound': mw,
                            'rmse': rmse,
                            'n': int(len(xw)),
                            'rt_min': float(_np.min(xw)), 'rt_max': float(_np.max(xw)),
                        })
                    else:
                        break
            except Exception:
                pass

        self.calibration_fit = best
        self.calibration_fits = {'combined': best} if best is not None else {}

        # Plotting for combined fit (include all points, even outside fit bounds)
        if plot and best is not None and go is not None and make_subplots is not None:
            a, b = best['a'], best['b']
            lb, ub = best['lower_bound'], best['upper_bound']
            # establish x-range for fitted curve from all points
            x_all = []
            for _, tdf in self.per_calibration_tables.items():
                t = _np.asarray(tdf[~tdf['MW'].astype(str).eq("")]['Exp. RT (min)'], dtype=float)
                t = t[_np.isfinite(t)]
                if t.size:
                    x_all.append((t.min(), t.max()))
            if x_all:
                x_min = float(min(v[0] for v in x_all)); x_max = float(max(v[1] for v in x_all))
            else:
                x_min = float(best['rt_min']); x_max = float(best['rt_max'])
            xs = _np.linspace(x_min, x_max, 200)
            ys = _np.exp(a + b * xs)

            fig = make_subplots(rows=1, cols=2, subplot_titles=("MW vs RT (combined)", "log(MW) vs RT (combined)"))
            # add per-sample points with consistent colors on both subplots
            for name, tdf in self.per_calibration_tables.items():
                try:
                    # Determine color from sample name/vial
                    vial_key = None
                    for e in self.calibration_set:
                        if e['name'] == name:
                            vial_key = e.get('vial_key'); break
                    color = _color_for_sample(name, vial_key)
                    sub = tdf[~tdf['MW'].astype(str).eq("")].copy()
                    x = _np.asarray(sub['Exp. RT (min)'], dtype=float)
                    y = _np.asarray(sub['MW'], dtype=float)
                    mask = _np.isfinite(x) & _np.isfinite(y) & (y > 0)
                    x = x[mask]; y = y[mask]
                    if x.size == 0:
                        continue
                    logy = _np.log(y)
                    logy_hat = a + b * x
                    err = _np.abs(logy - logy_hat)
                    # left (linear MW)
                    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color=color), name=name), row=1, col=1)
                    # right (log MW) with error bars
                    fig.add_trace(go.Scatter(x=x, y=logy, mode='markers', marker=dict(color=color),
                                             error_y=dict(type='data', array=err, visible=True),
                                             name=name, showlegend=False), row=1, col=2)
                except Exception:
                    continue
            # fitted curve overlay
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='Fit', line=dict(color='#222222', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=(a + b * xs), mode='lines', name='log Fit', line=dict(color='#222222', width=2), showlegend=False), row=1, col=2)
            for col in (1, 2):
                for bound in (best['rt_min'], best['rt_max']):
                    fig.add_vline(x=bound, line_width=1, line_dash='dash', line_color='gray', row=1, col=col)
            fig.update_xaxes(title_text='Time (min.)', row=1, col=1)
            fig.update_yaxes(title_text='MW (Da)', row=1, col=1)
            fig.update_xaxes(title_text='Time (min.)', row=1, col=2)
            fig.update_yaxes(title_text='log(MW)', row=1, col=2)
            fig.update_layout(title=f"Combined calibration fit | lower={int(lb)} Da, upper={int(ub)} Da | rmse={best['rmse']:.4f}")
            try:
                if _ipython_display is not None:
                    _ipython_display(fig)
            except Exception:
                pass

        # ---- Area vs Mass linear calibration (robust to outliers) ----
        # Build combined dataset of (Mass, Peak Area) with sample names
        if not self.per_calibration_tables:
            self.build_calibration_tables(plot=False)
        import numpy as _np
        import pandas as _pd
        rows = []
        for name, df_part in self.per_calibration_tables.items():
            try:
                part = df_part.copy()
                # coerce numeric
                part['Mass'] = _pd.to_numeric(part['Mass'], errors='coerce')
                part['Peak Area'] = _pd.to_numeric(part['Peak Area'], errors='coerce')
                part = part[_np.isfinite(part['Mass']) & _np.isfinite(part['Peak Area'])]
                part = part[(part['Mass'] > 0) & (part['Peak Area'] > 0)]
                if part.empty:
                    continue
                vial_key = None
                for e in self.calibration_set:
                    if e['name'] == name:
                        vial_key = e.get('vial_key'); break
                for _, r in part.iterrows():
                    rows.append({'name': name, 'vial_key': vial_key,
                                 'Mass': float(r['Mass']), 'Area': float(r['Peak Area'])})
            except Exception:
                continue
        if rows:
            adat = _pd.DataFrame(rows)
        else:
            adat = _pd.DataFrame(columns=['name','vial_key','Mass','Area'])

        if not adat.empty:
            x = _np.asarray(adat['Mass'], dtype=float)
            y = _np.asarray(adat['Area'], dtype=float)
            # initial LS fit with intercept: y = alpha + beta * x
            X = _np.vstack([_np.ones_like(x), x]).T
            try:
                beta, *_ = _np.linalg.lstsq(X, y, rcond=None)
                alpha, slope = float(beta[0]), float(beta[1])
            except Exception:
                alpha, slope = 0.0, 0.0
            resid = y - (alpha + slope * x)
            # robust scale via MAD
            mad = _np.median(_np.abs(resid - _np.median(resid)))
            sigma = 1.4826 * mad if mad > 0 else (_np.std(resid) if resid.size > 1 else 0.0)
            if sigma == 0:
                inlier_mask = _np.ones_like(y, dtype=bool)
            else:
                inlier_mask = _np.abs(resid) <= 3.0 * sigma
            # refit with inliers
            if inlier_mask.any():
                Xi = X[inlier_mask]
                yi = y[inlier_mask]
                beta2, *_ = _np.linalg.lstsq(Xi, yi, rcond=None)
                alpha, slope = float(beta2[0]), float(beta2[1])
                resid_i = yi - (alpha + slope * Xi[:,1])
                rmse = float(_np.sqrt(_np.mean(resid_i**2))) if yi.size > 0 else 0.0
            else:
                rmse = float(_np.sqrt(_np.mean(resid**2))) if y.size > 0 else 0.0

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
                        mask = inlier_mask[adat['name'].values == name]
                        vial_key = None
                        for e in self.calibration_set:
                            if e['name'] == name:
                                vial_key = e.get('vial_key'); break
                        color = _color_for_sample(name, vial_key)
                        # inliers
                        xi = sub['Mass'].values[mask]
                        yi = sub['Area'].values[mask]
                        if xi.size:
                            fig.add_trace(go.Scatter(x=xi, y=yi, mode='markers', marker=dict(color=color),
                                                     name=f"{name} (used)"))
                        # outliers
                        xo = sub['Mass'].values[~mask]
                        yo = sub['Area'].values[~mask]
                        if xo.size:
                            fig.add_trace(go.Scatter(x=xo, y=yo, mode='markers', marker=dict(color=color, symbol='x', size=10),
                                                     name=f"{name} (outlier)"))
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
            # RID
            rid = s.get('chrom_rid')
            if isinstance(rid, pd.DataFrame) and not rid.empty:
                rid_idx = _ensure_time_index(rid)
                try:
                    rt = _np.asarray(rid_idx.index, dtype=float)
                    mw = _np.exp(a + b * rt)
                    rid_idx[column_name] = mw
                    s['chrom_rid'] = rid_idx
                except Exception:
                    pass
            # DAD (prefer aligned)
            dad = s.get('chrom_dad_aligned') if isinstance(s.get('chrom_dad_aligned'), pd.DataFrame) else s.get('chrom_dad')
            key = 'chrom_dad_aligned' if isinstance(s.get('chrom_dad_aligned'), pd.DataFrame) else 'chrom_dad'
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
