# src/parser/viz.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


SignalMode = Union[str, Tuple[str, float]]  # "RID" or ("DAD", 230.0)


class HPLCVisualizer:
    """
    Visualization utilities for HPLC/GPC chromatograms.

    Assumptions:
      - Chromatograms were relabeled already (chrom_labeling.relabel_chromatograms):
          * index is time (minutes) with name "Time (min.)"
          * RID column is "Signal (<unit>)", e.g. "Signal (nRIU)"
          * DAD columns are "Signal {wl} nm (<unit>)", e.g. "Signal 230 nm (mAU)"
      - Each sample is a dict with keys like: barcode, repeat, chrom_rid, chrom_dad, method_struct/method_norm
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
        # remember last requested signal mode to allow auto-titling in make_figure
        self._last_signal: Optional[SignalMode] = None
        # track effective wavelength chosen from available DAD channels across samples
        self._effective_wavelength_nm: Optional[float] = None

    # ---------- utilities ----------
    @staticmethod
    def _nested_get(d: Dict[str, Any], dotted: str, default=None):
        cur = d
        for p in dotted.split("."):
            if not isinstance(cur, dict):
                return default
            cur = cur.get(p, default)
        return cur

    @staticmethod
    def _unique_sample_id(s: Dict[str, Any]) -> str:
        bc = str(s.get("barcode", "NA"))
        rp = s.get("repeat", 1)
        return f"{bc}#{rp}"

    # ---------- column helpers ----------
    @staticmethod
    def _rid_column(df: pd.DataFrame) -> Optional[str]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        # prefer exact "Signal (unit)" pattern
        for c in df.columns:
            if str(c).startswith("Signal (") and str(c).endswith(")"):
                return c
        # fallback: single column case
        if df.shape[1] == 1:
            return df.columns[0]
        return None

    @staticmethod
    def _dad_columns(df: pd.DataFrame) -> List[Tuple[float, str]]:
        """
        Return list of (wavelength_nm, column_name), sorted by wavelength.
        Expects columns like 'Signal 230 nm (mAU)'.
        """
        out: List[Tuple[float, str]] = []
        if not isinstance(df, pd.DataFrame) or df.empty:
            return out
        for c in df.columns:
            txt = str(c).lower()
            # parse 'signal {wl} nm ('...
            if txt.startswith("signal ") and " nm (" in txt:
                try:
                    mid = txt.split("signal ", 1)[1]
                    wl = float(mid.split(" nm", 1)[0].strip())
                    out.append((wl, str(c)))
                except Exception:
                    continue
        out.sort(key=lambda x: x[0])
        return out

    @staticmethod
    def _closest_wavelength(columns: List[Tuple[float, str]], target_nm: float) -> Optional[Tuple[float, str]]:
        if not columns:
            return None
        arr = np.array([wl for wl, _ in columns], dtype=float)
        idx = int(np.argmin(np.abs(arr - target_nm)))
        return columns[idx]

    # ---------- trace builder ----------
    def build_traces_grouped(
        self,
        *,
        signal: SignalMode = "RID",
        group_keys: Tuple[str, str] = ("sample_type", "barcode"),
        offset_factor: float = 0.35,
        group_gap: float = 10_000.0,
        robust_scale: bool = True,
        line_width: int = 2,
        color_cycle: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[Any, List[go.Scatter]]], List[float]]:
        """
        Build Plotly traces grouped in subplots and vertically offset within groups.

        Parameters
        ----------
        signal:
            "RID"  -> prefer sample['chrom_rid_baseline_corrected'] (fallback to 'chrom_rid')
            ("DAD", 230.0) -> prefer baseline/aligned DAD column closest to that wavelength [nm]
        group_keys:
            (subplot_key_path, group_key_path) dotted keys within sample dict.
            Example: ("sample_type", "metadata.Sample Name")
        offset_factor:
            Vertical spacing factor per sample within a group.
            If robust_scale=True, based on IQR; else based on max.
        group_gap:
            Extra spacing between groups inside a subplot (same unit as y).
        robust_scale:
            Use IQR-based amplitude per sample for more stable spacing.
        """
        # remember requested signal for downstream figure title
        self._last_signal = signal

        if color_cycle is None:
            # plotly qualitative + repeat
            from plotly.colors import qualitative
            color_cycle = qualitative.Plotly * 10

        # split into subplots by group_keys[0]
        subplot_map: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for s in self.samples:
            key1 = self._nested_get(s, group_keys[0], "NA")
            subplot_map[key1].append(s)

        subplot_traces: List[Tuple[Any, List[go.Scatter]]] = []
        y_max_by_subplot: List[float] = []

        # collect effective wavelengths seen (for DAD mode)
        effective_wls: List[float] = []

        for subplot_key, sub_samples in sorted(subplot_map.items(), key=lambda kv: str(kv[0])):
            # within each subplot, group by group_keys[1]
            groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
            for s in sub_samples:
                key2 = self._nested_get(s, group_keys[1], "NA")
                groups[key2].append(s)

            traces: List[go.Scatter] = []
            base_offset = 0.0

            for _, g_samples in sorted(groups.items(), key=lambda kv: str(kv[0])):
                # stable colors per barcode
                for idx, s in enumerate(g_samples):
                    sid = self._unique_sample_id(s)
                    name = str(s.get("barcode", "Sample"))
                    time, y, label, color, eff_wl = self._extract_series(s, signal, color_cycle[idx % len(color_cycle)])

                    if time is None or y is None:
                        continue

                    if eff_wl is not None:
                        effective_wls.append(float(eff_wl))

                    # compute spacing amplitude
                    if robust_scale:
                        q = np.nanpercentile(y, [5, 95]) if y.size else [0, 1]
                        amp = max(1e-9, (q[1] - q[0]))
                    else:
                        amp = max(1e-9, np.nanmax(y) - np.nanmin(y))

                    y_plot = y + base_offset + idx * offset_factor * amp

                    traces.append(
                        go.Scatter(
                            x=time,
                            y=y_plot,
                            name=name,
                            mode="lines",
                            line=dict(width=line_width, color=color),
                            legendgroup=name,
                            hovertemplate=f"ID: {sid}<br>Subplot: {subplot_key}"
                                          f"<br>Time (min.): %{{x:.3f}}"
                                          f"<br>Signal: %{{y:.3f}}<extra></extra>",
                            customdata=[sid],  # minimal
                            showlegend=False,
                        )
                    )

                # bump base offset for next group
                base_offset += (len(g_samples) * offset_factor + 0.5) * (amp if 'amp' in locals() else 1.0) + group_gap

            subplot_traces.append((subplot_key, traces))
            y_max_by_subplot.append(base_offset)

        # summarize effective wavelength across samples for DAD mode
        if isinstance(signal, tuple) and len(signal) == 2 and str(signal[0]).upper() == "DAD" and effective_wls:
            try:
                arr = np.round(np.array(effective_wls, dtype=float)).astype(int)
                vals, counts = np.unique(arr, return_counts=True)
                self._effective_wavelength_nm = float(vals[int(np.argmax(counts))])
            except Exception:
                self._effective_wavelength_nm = None
        else:
            self._effective_wavelength_nm = None

        return subplot_traces, y_max_by_subplot

    # ---------- figure creator ----------
    def make_figure(
        self,
        subplot_traces: List[Tuple[Any, List[go.Scatter]]],
        *,
        title: Optional[str] = None,
        x_title: str = "Time (min.)",
        y_title: str = "Signal (offset)",
        height_per_subplot: int = 360,
        width: int = 1400,
    ) -> go.Figure:
        # Auto-title if not provided, based on the last requested signal in build_traces_grouped
        if title is None:
            sig = getattr(self, "_last_signal", None)
            if sig == "RID":
                title = "RID Chromatograms"
            elif isinstance(sig, tuple) and len(sig) == 2 and str(sig[0]).upper() == "DAD":
                try:
                    eff = self._effective_wavelength_nm
                    nm_val = int(round(float(eff))) if eff is not None else int(round(float(sig[1])))
                    title = f"DAD {nm_val} nm"
                except Exception:
                    title = "DAD Chromatograms"
            else:
                title = "Chromatograms"
        else:
            # If a title is given and the signal is DAD, replace any embedded
            # 'DAD <number> nm' with the actual requested wavelength to avoid stale titles.
            sig = getattr(self, "_last_signal", None)
            if isinstance(sig, tuple) and len(sig) == 2 and str(sig[0]).upper() == "DAD":
                import re as _re
                try:
                    eff = self._effective_wavelength_nm
                    nm_txt = f"{int(round(float(eff)))}" if eff is not None else f"{int(round(float(sig[1])))}"
                    title = _re.sub(r"(?i)DAD\s+\d+(?:\.\d+)?\s*nm",
                                    f"DAD {nm_txt} nm", str(title))
                except Exception:
                    pass

        n = len(subplot_traces)
        fig = make_subplots(
            rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=[str(k) for k, _ in subplot_traces]
        )
        for r, (_, traces) in enumerate(subplot_traces, start=1):
            for tr in traces:
                fig.add_trace(tr, row=r, col=1)

        fig.update_layout(
            height=max(360, height_per_subplot * max(1, n)),
            width=width,
            title=title,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        fig.update_xaxes(title_text=x_title, row=n, col=1)
        fig.update_yaxes(title_text=y_title, row=int(np.ceil(n/2)), col=1)
        return fig

    # ---------- (optional) DAD heatmap waterfall (matplotlib-free, plotly) ----------
    def make_dad_heatmap_waterfall(
        self,
        *,
        sample_pad: float = 1.0,
        cmap: str = "Viridis",
        title: str = "All DAD Channels (waterfall heatmap)",
        width: int = 1200,
        height_per_band: Optional[int] = None,
        band_height_fraction: float = 0.05,
        min_height: int = 300,
    ) -> Optional[go.Figure]:
        """
        Plot DAD bands stacked per sample (each channel = one band).
        - Figure height scales with bands:
            * If height_per_band is provided: height = height_per_band * number_of_bands
            * Else: height = band_height_fraction * width * number_of_bands (default keeps prior behavior at 5%)
          In both cases, min_height caps the minimum height.
        - Y-axis labeled by wavelength [nm] at the center of each band.
        """
        all_traces: List[go.Heatmap] = []
        tick_text: List[str] = []
        tick_pos: List[float] = []
        y_base = 0.0
        total_bands = 0  # for height calculation

        for s in self.samples:
            # Prefer processed versions (baseline -> aligned -> raw)
            dad = s.get("chrom_dad_baseline_corrected")
            if (not isinstance(dad, pd.DataFrame)) or dad.empty:
                dad = s.get("chrom_dad_aligned")
            if (not isinstance(dad, pd.DataFrame)) or dad.empty:
                dad = s.get("chrom_dad")
            if not isinstance(dad, pd.DataFrame) or dad.empty:
                continue

            # ensure index is time
            if dad.index.name is None or "time" not in str(dad.index.name).lower():
                dad = dad.set_index(dad.columns[0])

            # sorted by wavelength: [(wl, colname), ...]
            ch_cols = self._dad_columns(dad)
            if not ch_cols:
                continue

            times = dad.index.values
            Z = np.array([dad[col].to_numpy(dtype=float) for (_, col) in ch_cols], dtype=float)

            # y coordinates for each band in this sample
            y_coords = np.arange(Z.shape[0]) + y_base
            all_traces.append(
                go.Heatmap(
                    x=times,
                    y=y_coords,
                    z=Z,
                    colorscale=cmap,
                    colorbar=dict(title="DAD Intensity"),
                    showscale=(len(all_traces) == 0),  # single colorbar
                )
            )

            # wavelength ticks at the center of each band
            for j, (wl, _) in enumerate(ch_cols):
                tick_pos.append(y_base + j + 0.5)
                tick_text.append(f"{int(round(wl))} nm")

            # advance base for next sample
            y_base += Z.shape[0] + sample_pad
            total_bands += Z.shape[0] + sample_pad  # count pads as band-heights for final sizing

        if not all_traces:
            return None

        # Determine figure height per band
        if height_per_band is not None:
            per_band_px = float(height_per_band)
        else:
            per_band_px = float(width) * float(band_height_fraction)
        height = max(int(min_height), int(per_band_px * max(1, total_bands)))

        fig = go.Figure(data=all_traces)
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            xaxis_title="Time (min.)",
            yaxis=dict(
                title="Wavelength (nm)",
                tickmode="array",
                tickvals=tick_pos,
                ticktext=tick_text,
            ),
        )
        return fig

    # ---------- internals ----------
    def _extract_series(
        self,
        sample: Dict[str, Any],
        signal: SignalMode,
        color: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, str, Optional[float]]:
        """
        Returns (time, y, label, color) for requested signal.
        """
        if signal == "RID":
            df = sample.get("chrom_rid_baseline_corrected")
            if (not isinstance(df, pd.DataFrame)) or df.empty:
                df = sample.get("chrom_rid")
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None, None, "", color, None
            # ensure time in index
            if df.index.name is None or "time" not in str(df.index.name).lower():
                df = df.set_index(df.columns[0])
            time = df.index.to_numpy(dtype=float)
            col = self._rid_column(df)
            if col is None:
                return None, None, "", color, None
            y = df[col].to_numpy(dtype=float)
            label = f"{sample.get('barcode','NA')} (RID)"
            return time, y, label, color, None

        # DAD @ wavelength
        if isinstance(signal, tuple) and len(signal) == 2 and str(signal[0]).upper() == "DAD":
            target_nm = float(signal[1])
            # Prefer processed DAD if present
            df = sample.get("chrom_dad_baseline_corrected")
            if (not isinstance(df, pd.DataFrame)) or df.empty:
                df = sample.get("chrom_dad_aligned")
            if (not isinstance(df, pd.DataFrame)) or df.empty:
                df = sample.get("chrom_dad")
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None, None, "", color, None
            if df.index.name is None or "time" not in str(df.index.name).lower():
                df = df.set_index(df.columns[0])
            cols = self._dad_columns(df)  # [(wl, colname), ...]
            if not cols:
                return None, None, "", color, None
            wl_col = self._closest_wavelength(cols, target_nm)
            if wl_col is None:
                return None, None, "", color, None
            wl, colname = wl_col
            time = df.index.to_numpy(dtype=float)
            y = df[colname].to_numpy(dtype=float)
            label = f"{sample.get('barcode','NA')} (DAD {int(round(wl))} nm)"
            return time, y, label, color, float(wl)

        # unknown mode
        return None, None, "", color, None
