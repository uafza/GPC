import pandas as pd
import re
from tqdm.notebook import tqdm
from collections import defaultdict, Counter
import numpy as np
import pubchempy as pcp
import time
import pubchempy as pcp
from tqdm import tqdm


class HPLCGroupAnalyzer:
    def __init__(self, samples, rt_tolerance=0.05):
        self.samples = samples
        self.rt_tolerance = rt_tolerance

    def match_qual_quant(self, sample):
        qual = sample.get('qual_results', pd.DataFrame()).copy()
        quant = sample.get('quant_results', pd.DataFrame()).copy()
        if qual.empty or quant.empty:
            return qual  # Nothing to match

        if 'RT (min)' not in qual.columns:
            print(f"qual_results columns: {qual.columns}")
            raise KeyError("qual_results missing 'RT (min)' column")
        if 'RT [min]' not in quant.columns:
            print(f"quant_results columns: {quant.columns}")
            return qual

        # Convert RT columns to numeric and drop NaNs
        qual['RT (min)'] = pd.to_numeric(qual['RT (min)'], errors='coerce')
        quant['RT [min]'] = pd.to_numeric(quant['RT [min]'], errors='coerce')
        qual_peaks = qual[qual['RT (min)'].notnull()].reset_index(drop=True)
        quant_peaks = quant[quant['RT [min]'].notnull()].reset_index(drop=True)

        # Only drop summary rows if column exists
        if 'Signal Description' in qual_peaks.columns:
            qual_peaks = qual_peaks[~qual_peaks['Signal Description'].astype(str).str.contains('Sum|Peak Details', case=False, na=False)]
        if 'Width (min)' in qual_peaks.columns:
            qual_peaks = qual_peaks[pd.to_numeric(qual_peaks['Width (min)'], errors='coerce').notnull()]

        if 'Signal Description' in quant_peaks.columns:
            quant_peaks = quant_peaks[~quant_peaks['Signal Description'].astype(str).str.contains('Sum|Peak Details', case=False, na=False)]
        if 'Width (min)' in quant_peaks.columns:
            quant_peaks = quant_peaks[pd.to_numeric(quant_peaks['Width (min)'], errors='coerce').notnull()]

        # For each quant peak, find closest qual peak within tolerance
        merged = qual_peaks.copy()
        for i, row in quant_peaks.iterrows():
            q_rt = row['RT [min]']
            diffs = (qual_peaks['RT (min)'] - q_rt).abs()
            min_idx = diffs.idxmin()
            min_diff = diffs[min_idx]
            if min_diff <= self.rt_tolerance:
                for col in quant.columns:
                    merged.loc[min_idx, f'quant_{col}'] = row[col]
        # (Optional) final brute-force filter: only keep rows where RT and Width are real numbers
        merged = merged[pd.to_numeric(merged['RT (min)'], errors='coerce').notnull()]
        if 'Width (min)' in merged.columns:
            merged = merged[pd.to_numeric(merged['Width (min)'], errors='coerce').notnull()]
        return merged

    
    def extract_dad_peak_spectra(self, hplc_sample, merged_peaks, rt_shift=-0.043):
        """
        For each RID peak (from merged_peaks DataFrame), extract mean DAD spectrum in the RT window.
        Saves list of dicts to hplc_sample['dad_peak_spectra'].
        Each dict contains: {'peak_idx', 'rt', 'width', 'mean_spectrum', 'channels', 'rt_mask'}
        """
        chrom_dad = hplc_sample.get('chrom_dad')
        if chrom_dad is None or chrom_dad.empty:
            hplc_sample['dad_peak_spectra'] = []
            return

        # For channel mapping (optional, for readability)
        method = hplc_sample.get('method', {})
        channel_map = HPLCVisualizer.get_channel_wavelength_map(method)
        channels = list(chrom_dad.columns)
        wavelengths = [channel_map.get(ch, np.nan) for ch in channels]

        results = []
        for idx, row in merged_peaks.iterrows():
            rt_rid = float(row['RT (min)'])
            width = float(row['Width (min)'])
            # Adjust for the DAD being downstream (shift peak to earlier RT)
            rt_dad_center = rt_rid + rt_shift
            half_width = width / 2
            # Window for the peak (could also use ±width, but usually FWHM/2 is best)
            rt_min = rt_dad_center - half_width
            rt_max = rt_dad_center + half_width

            # Find rows in DAD where index is within this RT window
            idx_mask = (chrom_dad.index >= rt_min) & (chrom_dad.index <= rt_max)
            if np.any(idx_mask):
                slice_dad = chrom_dad.loc[idx_mask, :]
                mean_spectrum = slice_dad.mean(axis=0).values  # mean for each channel (A-H)
            else:
                mean_spectrum = np.full(len(channels), np.nan)
            results.append({
                'peak_idx': idx,
                'rt': rt_rid,
                'width': width,
                'mean_spectrum': mean_spectrum,
                'channels': channels,
                'wavelengths': wavelengths,
                'rt_mask': idx_mask,  # for trace/debug
            })
        hplc_sample['dad_peak_spectra'] = results

    def remove_samples_by_barcodes(self, exclude_barcodes):
        norm_exclude = set(str(int(b)).zfill(10) for b in exclude_barcodes)
        filtered = []
        for s in self.samples:
            barcode = s.get('barcode')
            if barcode is None:
                filtered.append(s)
                continue
            try:
                norm_barcode = str(int(barcode)).zfill(10)
            except Exception:
                norm_barcode = str(barcode)
            if norm_barcode not in norm_exclude:
                filtered.append(s)
        self.samples = filtered

    def filter_all_samples_by_area_percent(self, min_area_percent=1.0, store_key='filtered_matched_peaks'):
        """
        For each sample in self.samples:
            - Runs match_qual_quant
            - Filters by min_area_percent
            - Stores result in sample[store_key]
        """
        for sample in self.samples:
            merged = self.match_qual_quant(sample)
            filtered = self.filter_matched_peaks_by_area_percent(merged, min_area_percent=min_area_percent)
            sample[store_key] = filtered
    
    def filter_matched_peaks_by_area_percent(self, merged_df, min_area_percent=1.0):
        """
        Remove peaks < min_area_percent of total area in this sample's merged table.
        """
        if merged_df.empty or 'Area' not in merged_df.columns:
            return merged_df
        merged_df['Area'] = pd.to_numeric(merged_df['Area'], errors='coerce').fillna(0)
        total_area = merged_df['Area'].sum()
        keep = merged_df['Area'] / total_area * 100 >= min_area_percent if total_area > 0 else False
    
        return merged_df[keep]
    
    def align_peaks_by_retention_time(self, rt_tolerance=0.05, peak_table_key='filtered_matched_peaks'):
        """
        Align peaks by retention time across all samples and assign a cluster id.
        Returns a DataFrame where each row is a peak from a sample with a 'peak_cluster' column.
        """
        all_peaks = []

        for sample in self.samples:
            sample_name = sample.get('barcode') or sample.get('sample_name', '')
            peaks = sample.get(peak_table_key)

            # Skip if no peaks or empty
            if peaks is None or isinstance(peaks, float):
                continue
            if not isinstance(peaks, pd.DataFrame) or peaks.empty:
                continue

            # Work on a copy and coerce key columns to numeric
            p = peaks.copy()
            # Normalize the column name (in case of weird spacing)
            if 'RT (min)' not in p.columns:
                # try to find a close match like "RT (min) " etc.
                rt_cols = [c for c in p.columns if str(c).strip().lower() == 'rt (min)']
                if rt_cols:
                    p.rename(columns={rt_cols[0]: 'RT (min)'}, inplace=True)

            if 'RT (min)' not in p.columns:
                # Can't use this sample's peaks
                continue

            p['RT (min)'] = pd.to_numeric(p['RT (min)'], errors='coerce')

            # Drop rows with non-numeric RT (summary/footer lines)
            p = p[p['RT (min)'].notna()].reset_index(drop=True)

            if p.empty:
                continue

            # Attach sample metadata
            p['sample_name'] = sample_name
            p['barcode'] = sample.get('barcode')
            p['group'] = sample.get('group', None)

            all_peaks.append(p)

        if not all_peaks:
            print("No peaks to align!")
            return pd.DataFrame()

        df = pd.concat(all_peaks, ignore_index=True)

        # Ensure RT is numeric and sorted
        df['RT (min)'] = pd.to_numeric(df['RT (min)'], errors='coerce')
        df = df[df['RT (min)'].notna()].sort_values('RT (min)').reset_index(drop=True)

        if df.empty:
            print("No valid numeric RTs to align!")
            return df

        # ---- Cluster by RT tolerance ----
        clusters = []
        current_cluster = 0
        cluster_ref_rt = float(df.iloc[0]['RT (min)'])
        clusters.append(current_cluster)

        for rt in df['RT (min)'].iloc[1:].astype(float).values:
            if abs(rt - cluster_ref_rt) <= rt_tolerance:
                clusters.append(current_cluster)
            else:
                current_cluster += 1
                clusters.append(current_cluster)
                cluster_ref_rt = rt

        df['peak_cluster'] = clusters
        return df

        
    def compute_pivot_table(self, cluster_df, value_cols=None, sample_id_col='barcode',
                            label_col='quant_Name', rt_col='RT (min)', cluster_col='peak_cluster'):
        if value_cols is None:
            qual_cols = [c for c in cluster_df.columns if c in {'Area', 'Area%'}]
            quant_cols = [c for c in cluster_df.columns if c.startswith('quant_')]
            value_cols = qual_cols + quant_cols
        peak_labels = self.build_peak_labels(cluster_df, name_col=label_col, rt_col=rt_col, cluster_col=cluster_col)
        result = {}
        for value_col in value_cols:
            pivot_table = cluster_df.pivot_table(
                index=sample_id_col,
                columns=cluster_col,
                values=value_col,
                aggfunc='first'
            )
            # Set nice headers
            pivot_table.columns = [peak_labels.get(c, c) for c in pivot_table.columns]
            result[value_col] = pivot_table
        return result

        
    def build_peak_labels(self, cluster_df, name_col='quant_Name', rt_col='RT (min)', cluster_col='peak_cluster'):
        peak_labels = {}
        for cluster in cluster_df[cluster_col].unique():
            peaks = cluster_df[cluster_df[cluster_col] == cluster]
            mean_rt = peaks[rt_col].mean()
            # Most common (non-empty) name if available
            most_common_name = peaks[name_col].dropna().astype(str).replace('nan','').mode()
            if not most_common_name.empty and most_common_name.iloc[0]:
                label = f"{cluster}: {mean_rt:.2f} min, {most_common_name.iloc[0]}"
            else:
                label = f"{cluster}: {mean_rt:.2f} min"
            peak_labels[cluster] = label
        return peak_labels


    def collapse_same_name_columns(self,
                                wide_table: pd.DataFrame,
                                combine_istd: bool = False) -> pd.DataFrame:
        """
        Collapse (sum) columns within each group (Area, Area%, quant_*)
        when they have the SAME compound name in their header, even if they
        came from different RT clusters.

        Expected column patterns:
        - "Area|<idx>: <rt> min, <Name>"
        - "Area%|<idx>: <rt> min, <Name>"
        - "quant_Concentration [g/L]|<idx>: <rt> min, <Name>"
        - or already name-only: "...|<Name>"

        By default ISTD-like names are NOT collapsed (set combine_istd=True to enable).
        """
        df = wide_table.copy()

        def is_group_col(c: str, prefix: str) -> bool:
            return str(c).startswith(prefix)

        def extract_label(c: str) -> str:
            return c.split("|", 1)[1] if "|" in str(c) else str(c)

        def extract_name_from_label(label: str) -> str | None:
            # Try "... min, <Name>"
            m = re.search(r"min,\s*(.+)$", label, flags=re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                if any(ch.isalpha() for ch in name):
                    return name
            # If no "min", treat full label as name if it looks like a chemical string
            if "min" not in label.lower() and any(ch.isalpha() for ch in label):
                return label.strip()
            return None

        def is_istd_name(name: str) -> bool:
            s = name.lower()
            return any(t in s for t in ("istd", "internal standard", "internal standart", "internal std", "triglyme"))

        # Define groups in table-order (Area, Area%, quant_*)
        group_specs = [
            ("Area|", "Area"),
            ("Area%|", "Area%"),
            ("quant_", "quant"),   # we'll preserve the exact 'quant_*' prefix per column
        ]

        out_series: list[pd.Series] = []
        out_names:  list[str] = []

        for prefix, key in group_specs:
            group_cols = [c for c in df.columns if is_group_col(c, prefix)]
            if not group_cols:
                continue

            # Map column -> parsed name (or None)
            names_for_col: dict[str, str | None] = {}
            for c in group_cols:
                label = extract_label(str(c))
                names_for_col[c] = extract_name_from_label(label)

            # Buckets: canonical name -> list of member columns (order preserved)
            buckets: dict[str, list[str]] = {}
            for c in group_cols:
                nm = names_for_col[c]
                if nm and (combine_istd or not is_istd_name(nm)):
                    buckets.setdefault(nm.lower(), []).append(c)

            # Emit columns for this block in left-to-right order
            for c in group_cols:
                nm = names_for_col[c]

                # unnamed column → pass through unchanged
                if not nm:
                    out_series.append(df[c])
                    out_names.append(c)
                    continue

                # ISTD (kept separate by default)
                if (not combine_istd) and is_istd_name(nm):
                    out_series.append(df[c])
                    out_names.append(c)
                    continue

                key_l = nm.lower()
                members = buckets.get(key_l, [c])

                # If multiple members share the same name, only the FIRST position
                # emits the SUM; later duplicates are skipped.
                if len(members) > 1:
                    if c != members[0]:
                        continue
                    s = df[members].sum(axis=1, skipna=True)
                    # keep NaN where all members are NaN
                    s[df[members].notna().sum(axis=1) == 0] = np.nan

                    if key == "quant":
                        qprefix = str(members[0]).split("|", 1)[0]  # e.g. "quant_Concentration [g/L]"
                        out_name = f"{qprefix}|{nm}"
                    else:
                        out_name = f"{key}|{nm}"

                    out_series.append(s)
                    out_names.append(out_name)
                else:
                    # unique named column → keep as-is
                    out_series.append(df[c])
                    out_names.append(c)

        # Build the new frame in one go to avoid fragmentation
        new_df = pd.concat(out_series, axis=1)
        new_df.columns = out_names
        return new_df
