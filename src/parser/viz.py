import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from dash import dcc, html, Input, Output, Dash
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import defaultdict
from collections import defaultdict
import plotly.colors


class GCMSVisualizer:
    def __init__(self, samples):
        """
        samples: list of dicts, each with a 'sample_info' and 'ms_chromatogram' (from GCMSBatchParser)
        """
        self.samples = samples

    def plot_all_chromatograms_waterfall(self, time_col='Ret.Time', intensity_col='Absolute Intensity', offset=0.2, figsize=(10,8)):
        """
        Plot all chromatograms in a waterfall style.
        - time_col: column in DataFrame for x-axis (default: 'Ret.Time')
        - intensity_col: column in DataFrame for y-axis (default: 'Absolute Intensity')
        - offset: vertical offset between samples (default: 0.2 times max intensity)
        """
        plt.figure(figsize=figsize)
        n = len(self.samples)
        if n == 0:
            print("No samples to plot.")
            return
        for i, sample in enumerate(self.samples):
            chrom = sample.get('ms_chromatogram')
            if chrom is None or chrom.empty:
                continue
            y = chrom[intensity_col] + i * offset * chrom[intensity_col].max()
            plt.plot(chrom[time_col], y, label=sample['sample_info'].get('sample_name', f"Sample {i+1}"))
        plt.xlabel('Time [min]')
        plt.ylabel('Absolute Intensity + offset')
        plt.title('GCMS Chromatograms (Waterfall Plot)')
        plt.legend(fontsize=8, loc='upper right')
        plt.tight_layout()
        plt.show()


# --- Your HPLCVisualizer class (use the latest version you made) ---
class HPLCVisualizer:
    def __init__(self, samples):
        self.samples = samples

    def get_nested_keys(self, sample, keys, default=None):
        if isinstance(keys, str):
            keys = [keys]
        result = []
        for key in keys:
            value = sample
            for k in key.split('.'):
                if isinstance(value, dict):
                    value = value.get(k, default)
                else:
                    value = default
            result.append(value if value is not None else default)
        return tuple(result)

    def get_plotly_traces_grouped(
        self,
        group_keys=None,  # List of 2: first = subplot key, second = group-within-subplot key
        time_col=0,
        signal_col=1,
        offset=0.2,
        group_gap=20000
    ):

        subplot_dict = defaultdict(list)
        for sample in self.samples:
            key1 = self.get_nested_keys(sample, [group_keys[0]])[0]
            subplot_dict[key1].append(sample)

        color_palette = plotly.colors.qualitative.Plotly * 10
        subplot_traces = []
        y_max_by_subplot = []

        for subplot_idx, (subplot_key, group_samples) in enumerate(sorted(subplot_dict.items(), key=lambda x: str(x[0]))):
            groups = defaultdict(list)
            for sample in group_samples:
                key2 = self.get_nested_keys(sample, [group_keys[1]])[0]
                groups[key2].append(sample)

            base_offset = 0
            traces = []
            max_y = 0
            for g_idx, (group, samples_in_group) in enumerate(sorted(groups.items(), key=lambda x: str(x[0]))):
                group_max = 0
                for s_idx, sample in enumerate(samples_in_group):
                    chrom = sample.get('chrom_rid')
                    if chrom is None or chrom.empty:
                        continue
                    x = chrom.iloc[:, time_col]
                    y_raw = chrom.iloc[:, signal_col]
                    y = y_raw + base_offset + s_idx * offset * y_raw.max()
                    barcode = sample.get('barcode', f'Sample {s_idx+1}')
                    label = f"{group}: {barcode}"
                    color = color_palette[s_idx % len(color_palette)]
                    traces.append(go.Scatter(
                        x=x, y=y,
                        name=label,
                        mode='lines',
                        line=dict(color=color, width=2),
                        legendgroup=label,
                        customdata=[barcode]*len(x),
                        hovertemplate=f"Barcode: {barcode}<br>Group: {group}<br>Subplot: {subplot_key}<br>Time: %{{x}}<br>Signal: %{{y}}<extra></extra>",
                        showlegend=False,  # We'll use our own legend in Dash
                    ))
                    group_max = max(group_max, y_raw.max())
                base_offset += group_max * offset * len(samples_in_group) + group_gap
            subplot_traces.append((subplot_key, traces))
            y_max_by_subplot.append(base_offset)
        return subplot_traces, y_max_by_subplot
    

    @staticmethod
    def get_channel_wavelength_map(method):
        """
        Returns dict like {'DAD_1A': 254, ...} for any Agilent method block structure.
        """
        # Try to locate a "3.2 Signals" section
        for sec, block in method.items():
            if "3.2 Signals" in sec:
                tbl = block.get('3.2 Signals') if isinstance(block, dict) and '3.2 Signals' in block else block
                # 1. If it's a DataFrame (your old case)
                if hasattr(tbl, "iterrows"):
                    mapping = {}
                    for i, row in tbl.iterrows():
                        sig = row['Signal']
                        wl = float(str(row['Wavelength']).replace('nm','').strip())
                        ch = 'DAD_1' + sig.split()[-1]  # Signal A -> DAD_1A
                        mapping[ch] = wl
                    return mapping
                # 2. If it's a dict (your current case)
                if isinstance(tbl, dict):
                    # Find all dict values starting with "Signal "
                    mapping = {}
                    for key, val in tbl.items():
                        if key.startswith('Yes') or key.startswith('No') or key.startswith('Acquire'):
                            # Expect a value like: 'Signal H,250.0 nm,4.0 nm,Yes,360.0 nm,100.0 nm'
                            parts = val.split(',')
                            if parts[0].startswith('Signal '):
                                sig_letter = parts[0].replace('Signal ','').strip()
                                wl = float(parts[1].replace('nm','').strip())
                                ch = f'DAD_1{sig_letter}'
                                mapping[ch] = wl
                    return mapping
        # fallback: try generic
        return {}

    def plot_all_dad_heatmap_waterfall(self, offset=1.2, sample_pad=8, figsize=(14, 12), cmap='viridis'):
        """
        For all samples, plot DAD chromatogram as a stacked waterfall heatmap.
        Each 'band' is a sample (8 channels stacked), color is intensity.
        x: RT (time), y: stacked bands (channels per sample + spacing), color: intensity.
        """


        fig, ax = plt.subplots(figsize=figsize)
        y_base = 0
        ytick_labels = []
        ytick_pos = []
        for i, sample in enumerate(self.samples):
            chrom_dad = sample.get('chrom_dad')
            method = sample.get('method', {})
            if chrom_dad is None or len(chrom_dad) == 0:
                continue
            # Map DAD_1A, DAD_1B, ... to nm
            channel_map = self.get_channel_wavelength_map(method)
            if not channel_map:
                channel_map = {col: 200 + 10 * idx for idx, col in enumerate(chrom_dad.columns)}
            channels = [ch for ch in chrom_dad.columns if ch in channel_map]
            if not channels:
                continue
            rt = chrom_dad.index.values
            n_ch = len(channels)
            for j, ch in enumerate(channels):
                y_pos = y_base + j
                ax.pcolormesh(rt, [y_pos, y_pos+1], np.vstack([chrom_dad[ch].values, chrom_dad[ch].values]),
                            shading='auto', cmap=cmap)
            # Mark sample center for ytick
            ytick_labels.append(str(sample.get('barcode', f'Sample {i+1}')))
            ytick_pos.append(y_base + n_ch / 2)
            y_base += n_ch + sample_pad  # more space between samples

        if not ytick_labels:
            print("No DAD data to plot.")
            return
        ax.set_xlabel('Retention Time [min]')
        ax.set_ylabel('Sample')
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels, fontsize=8)
        plt.title('All DAD Channels (waterfall heatmap)')
        plt.tight_layout()
        plt.colorbar(ax.collections[0], ax=ax, label='DAD Intensity')
        plt.show()

    def plot_dad_peak_spectra(self, hplc_samples, title="Averaged DAD spectra for RID peaks"):
        """
        For all samples, plot the averaged DAD spectrum extracted at each RID peak.
        Each sample gets its own subplot; X-axis is wavelength, Y is absorbance.
        """


        # Filter samples with spectra
        samples_with_spectra = [s for s in hplc_samples if s.get('dad_peak_spectra')]
        if not samples_with_spectra:
            print("No DAD spectra to plot.")
            return

        n_samples = len(samples_with_spectra)
        fig, axs = plt.subplots(n_samples, 1, figsize=(9, 3 * n_samples), squeeze=False, sharex=True)
        fig.suptitle(title, fontsize=14)

        for i, sample in enumerate(samples_with_spectra):
            ax = axs[i, 0]
            barcode = sample.get('barcode', f'Sample {i+1}')
            spectra_list = sample['dad_peak_spectra']
            for peak in spectra_list:
                spectrum = peak['mean_spectrum']
                wl = np.array(peak['wavelengths'])
                # sort wavelengths
                sort_idx = np.argsort(wl)
                wl_sorted = wl[sort_idx]
                y_sorted = np.array(spectrum)[sort_idx]
                label = f"RT={peak['rt']:.2f} min, w={peak['width']:.2f}"
                ax.plot(wl_sorted, y_sorted, marker='o', label=label, alpha=0.75)
            ax.set_ylabel("Abs (a.u.)")
            ax.set_title(f"Sample {barcode} (n_peaks={len(spectra_list)})")
            ax.legend(fontsize=7, loc='upper right', ncol=2)
        axs[-1, 0].set_xlabel("Wavelength [nm]")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()


    def show_dash_selector(self, group_keys):
        subplot_traces, y_max_by_subplot = self.get_plotly_traces_grouped(group_keys=group_keys)

        n_subplots = len(subplot_traces)
        fig = make_subplots(
            rows=n_subplots, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=[f"{group_keys[0]}: {key}" for key, _ in subplot_traces]
        )

        barcode_to_traceidx = {}
        legend_items = []

        for idx, (subplot_key, traces) in enumerate(subplot_traces):
            for t_idx, trace in enumerate(traces):
                fig.add_trace(trace, row=idx+1, col=1)
                barcode = trace.customdata[0]
                legend_items.append({'label': trace.name, 'value': barcode})
                barcode_to_traceidx[barcode] = len(fig.data) - 1

        fig.update_layout(
            height=400 * n_subplots,
            width=1400,
            title="Click a curve in the dropdown to highlight, barcode added below",
            xaxis_title='Time [min]',
            yaxis_title='RID Signal (arbitrary offset)',
            legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='closest'
        )

        app = Dash(__name__)

        app.layout = html.Div([
            dcc.Graph(id='chrom_plot', figure=fig, style={'height': f"{400 * n_subplots}px"}),
            html.Br(),
            html.Label("Select curves to highlight and add barcode to exclusion list:"),
            dcc.Dropdown(
                id='curve_selector',
                options=[{'label': item['label'], 'value': item['value']} for item in legend_items],
                multi=True,
                placeholder="Select by barcode...",
                style={'width': '60%'}
            ),
            html.Div(id='selected_barcodes', style={'marginTop': '20px', 'fontSize': 18}),
        ])

        @app.callback(
            Output('chrom_plot', 'figure'),
            Output('selected_barcodes', 'children'),
            Input('curve_selector', 'value')
        )
        def highlight_curves(selected_barcodes):
            fig2 = go.Figure(fig)  # CORRECT: use go.Figure to copy
            selected_barcodes = selected_barcodes or []
            for idx, trace in enumerate(fig2.data):
                barcode = trace.customdata[0]
                if selected_barcodes and barcode not in selected_barcodes:
                    trace.line.color = 'lightgray'
                    trace.line.width = 1
                elif barcode in selected_barcodes:
                    trace.line.width = 3
            sel_txt = "Selected barcodes for exclusion: " + ", ".join(selected_barcodes) if selected_barcodes else "No barcodes selected."
            return fig2, sel_txt

        app.run(debug=True)  # Do not use mode="inline" with Dash>=3; works in Jupyter too
