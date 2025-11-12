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
            "RID"  -> use sample['chrom_rid'] with its single 'Signal (unit)' column
            ("DAD", 230.0) -> use sample['chrom_dad'] column closest to that wavelength [nm]
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
                    time, y, label, color = self._extract_series(s, signal, color_cycle[idx % len(color_cycle)])

                    if time is None or y is None:
                        continue

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

        return subplot_traces, y_max_by_subplot

    # ---------- figure creator ----------
    def make_figure(
        self,
        subplot_traces: List[Tuple[Any, List[go.Scatter]]],
        *,
        title: str = "Chromatograms",
        x_title: str = "Time (min.)",
        y_title: str = "Signal (offset)",
        height_per_subplot: int = 360,
        width: int = 1400,
    ) -> go.Figure:
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
    ) -> Optional[go.Figure]:
        """
        Plot DAD bands stacked per sample (each channel a band). Uses plotly heatmap for interactivity.
        """
        all_traces: List[go.Heatmap] = []
        y_ticks = []
        y_tickpos = []
        y_base = 0.0

        for i, s in enumerate(self.samples):
            dad = s.get("chrom_dad")
            if not isinstance(dad, pd.DataFrame) or dad.empty:
                continue

            # ensure index is time
            if dad.index.name is None or "time" not in str(dad.index.name).lower():
                dad = dad.set_index(dad.columns[0])

            ch_cols = self._dad_columns(dad)  # sorted (wl, col)
            if not ch_cols:
                continue

            times = dad.index.values
            Z = []
            for j, (_, col) in enumerate(ch_cols):
                Z.append(dad[col].to_numpy(dtype=float))
            Z = np.array(Z, dtype=float)  # shape: (n_ch, n_time)

            # construct y-coordinates per channel band
            y_coords = np.arange(Z.shape[0]) + y_base
            all_traces.append(
                go.Heatmap(
                    x=times,
                    y=y_coords,
                    z=Z,
                    colorscale=cmap,
                    colorbar=dict(title="DAD Intensity"),
                    showscale=(len(all_traces) == 0),  # one colorbar
                )
            )
            y_ticks.append(str(s.get("barcode", f"Sample {i+1}")))
            y_tickpos.append(y_base + (Z.shape[0] - 1) / 2.0)
            y_base += Z.shape[0] + sample_pad

        if not all_traces:
            return None

        fig = go.Figure(data=all_traces)
        fig.update_layout(
            title=title,
            xaxis_title="Time (min.)",
            yaxis=dict(
                title="Sample",
                tickmode="array",
                tickvals=y_tickpos,
                ticktext=y_ticks,
            ),
            width=1200,
            height=max(400, int(70 * y_base)),
        )
        return fig

    # ---------- internals ----------
    def _extract_series(
        self,
        sample: Dict[str, Any],
        signal: SignalMode,
        color: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, str]:
        """
        Returns (time, y, label, color) for requested signal.
        """
        if signal == "RID":
            df = sample.get("chrom_rid")
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None, None, "", color
            # ensure time in index
            if df.index.name is None or "time" not in str(df.index.name).lower():
                df = df.set_index(df.columns[0])
            time = df.index.to_numpy(dtype=float)
            col = self._rid_column(df)
            if col is None:
                return None, None, "", color
            y = df[col].to_numpy(dtype=float)
            label = f"{sample.get('barcode','NA')} (RID)"
            return time, y, label, color

        # DAD @ wavelength
        if isinstance(signal, tuple) and len(signal) == 2 and str(signal[0]).upper() == "DAD":
            target_nm = float(signal[1])
            df = sample.get("chrom_dad")
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None, None, "", color
            if df.index.name is None or "time" not in str(df.index.name).lower():
                df = df.set_index(df.columns[0])
            cols = self._dad_columns(df)  # [(wl, colname), ...]
            if not cols:
                return None, None, "", color
            wl_col = self._closest_wavelength(cols, target_nm)
            if wl_col is None:
                return None, None, "", color
            wl, colname = wl_col
            time = df.index.to_numpy(dtype=float)
            y = df[colname].to_numpy(dtype=float)
            label = f"{sample.get('barcode','NA')} (DAD {int(round(wl))} nm)"
            return time, y, label, color

        # unknown mode
        return None, None, "", color
