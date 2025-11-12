import pandas as pd
import re
from tqdm.notebook import tqdm
from collections import defaultdict, Counter
import numpy as np
import pubchempy as pcp
import time
import pubchempy as pcp
from tqdm import tqdm

class GCMSGroupAnalyzer:
    def __init__(self, samples):
        self.samples = samples

    def get_group(self, group_name=None):
        if group_name is None:
            return self.samples
        # Use metadata['Substrate'] for grouping, fallback to group if metadata is missing
        return [
            s for s in self.samples
            if (
                'metadata' in s and s['metadata'].get('Substrate') == group_name
            ) or (
                s.get('group') == group_name  # fallback for backward compatibility
            )
        ]


    def align_peaks_by_retention_time(self, group_name=None, rt_tolerance=0.05):
        group_samples = self.get_group(group_name)
        peak_rows = []
        for sample in group_samples:
            sname = sample['sample_info'].get('sample_name')
            for peak in sample['peaks']:
                lib_hit_names = [
                    hit['names'][0] if hit.get('names') and len(hit['names']) > 0 else ''
                    for hit in peak.get('library_hits', [])
                ]
                row = {
                    'sample_name': sname,
                    'peak_number': peak.get('peak_number'),
                    'retention_time': peak.get('retention_time'),
                    'library_hit_names': lib_hit_names,
                }
                peak_rows.append(row)
        df = pd.DataFrame(peak_rows)
        df = df.sort_values('retention_time')
        clusters = []
        if not df.empty:
            current_cluster = 0
            cluster_rt = df.iloc[0]['retention_time']
            clusters.append(current_cluster)
            for rt in df['retention_time'][1:]:
                if abs(rt - cluster_rt) <= rt_tolerance:
                    clusters.append(current_cluster)
                else:
                    current_cluster += 1
                    clusters.append(current_cluster)
                    cluster_rt = rt
            df['peak_cluster'] = clusters
        return df

    def print_clustered_peak_names(self, group_name=None, rt_tolerance=0.05):
        df = self.align_peaks_by_retention_time(group_name, rt_tolerance)
        if df.empty:
            print("No peaks found.")
            return
        for cid, g in df.groupby('peak_cluster'):
            print(f"\nPeak cluster {cid} (RT ≈ {g['retention_time'].mean():.2f}):")
            print("  Peak numbers:", set(g['peak_number']))
            hits = set(h for hitlist in g['library_hit_names'] for h in hitlist if h)
            print("  Lib hit names:", hits)

    @staticmethod
    def plain_formula(formula):
        if not formula:
            return None
        formula = re.sub(r'<sub>(.*?)</sub>', r'\1', formula)
        formula = re.sub(r'<sup>(.*?)</sup>', r'\1', formula)
        formula = re.sub(r'<[^>]+>', '', formula)
        return formula.replace(' ', '')

    @staticmethod
    def is_CHO_formula(formula):
        if not formula:
            return False
        atoms = re.findall(r'[A-Z][a-z]?', formula)
        return all(a in {'C', 'H', 'O'} for a in atoms)

    def annotate_CHO_hits(self):
        """
        Annotate all library hits in all samples with 'is_CHO'
        using only the molecular formula from each hit.
        No name or CAS lookup is performed.
        """
        print("Annotating all library hits with C/H/O filter using table formulas...")
        for sample in tqdm(self.samples, desc="Samples"):
            for peak in tqdm(sample['peaks'], leave=False, desc="Peaks"):
                for hit in peak.get('library_hits', []):
                    formula = self.plain_formula(hit.get('molecular_formula'))
                    is_CHO = self.is_CHO_formula(formula)
                    hit['molecular_formula_lookup'] = formula
                    hit['is_CHO'] = is_CHO
        print("Done annotating C/H/O hits.")

    def print_all_peaks_and_hits(self):
        for sidx, sample in enumerate(self.samples):
            print(f"\nSample {sidx}: {sample['sample_info'].get('sample_name', '')}")
            for peak in sample['peaks']:
                print(f"  Peak {peak.get('peak_number')} @ RT {peak.get('retention_time')}:")
                for hit in peak.get('library_hits', []):
                    syns = ' $$ '.join(hit.get('names', []))
                    print(f"    Hit: {syns} (CAS {hit.get('cas_number','')}), "
                          f"CHO: {hit.get('is_CHO')}, Formula: {hit.get('molecular_formula_lookup')}")

    def filter_library_hits_by_CHO(self):
        """
        For each peak in all samples, keep only those library hits where is_CHO is True.
        """
        for sample in self.samples:
            for peak in sample['peaks']:
                peak['library_hits'] = [hit for hit in peak.get('library_hits', []) if hit.get('is_CHO')]

    def filter_library_hits_by_max_c(self, max_c_atoms=6):
        """
        Remove all library hits from all peaks where the number of carbon atoms exceeds max_c_atoms.
        """
        c_pattern = re.compile(r'C(\d*)')
        for sample in self.samples:
            for peak in sample['peaks']:
                filtered_hits = []
                for hit in peak['library_hits']:
                    formula = hit.get('molecular_formula_lookup') or hit.get('molecular_formula')
                    n_c = 0
                    if formula:
                        m = c_pattern.search(formula)
                        if m:
                            n_c = int(m.group(1) or 1)
                    if n_c <= max_c_atoms:
                        filtered_hits.append(hit)
                peak['library_hits'] = filtered_hits

    def filter_peaks_by_min_area_percent(self, min_area_percent=0.5):
        """
        Keep only peaks where the area as a percent of total peak area in the sample >= min_area_percent.
        Assumes peak['area'] is present.
        """
        for sample in self.samples:
            total_area = sum(p.get('area', 0) or 0 for p in sample['peaks'])
            filtered_peaks = []
            for peak in sample['peaks']:
                area = peak.get('area', 0) or 0
                area_percent = (area / total_area) * 100 if total_area else 0
                if area_percent >= min_area_percent:
                    filtered_peaks.append(peak)
            sample['peaks'] = filtered_peaks

    def cross_correlate_peaks(self, rt_tolerance=0.05, min_similarity=0, use_formular_first=True):
        all_peaks = []
        for s_idx, sample in enumerate(self.samples):
            sname = sample['sample_info'].get('sample_name', f'sample_{s_idx}')
            for peak in sample['peaks']:
                rt = peak.get('retention_time')
                if rt is None:
                    continue
                for hit in peak.get('library_hits', []):
                    all_peaks.append({
                        'sample': sname,
                        'peak_number': peak.get('peak_number'),
                        'retention_time': rt,
                        'hit_names': hit.get('names', []),
                        'hit_formula': hit.get('molecular_formula', ''),
                        'hit_cas': hit.get('cas_number', ''),
                        'hit_si': hit.get('similarity_index', None),
                    })

        all_peaks_df = pd.DataFrame(all_peaks)
        if all_peaks_df.empty:
            print("No peaks found.")
            return pd.DataFrame()

        # --- Cluster by retention time ---
        sorted_rts = np.sort(all_peaks_df['retention_time'].unique())
        clusters = []
        if len(sorted_rts) == 0:
            return pd.DataFrame()
        current_cluster = 0
        cluster_rt = sorted_rts[0]
        clusters_dict = {cluster_rt: current_cluster}
        for rt in sorted_rts[1:]:
            if abs(rt - cluster_rt) <= rt_tolerance:
                clusters_dict[rt] = current_cluster
            else:
                current_cluster += 1
                clusters_dict[rt] = current_cluster
                cluster_rt = rt
        all_peaks_df['peak_cluster'] = all_peaks_df['retention_time'].map(clusters_dict)

        rows = []
        for cid, group in all_peaks_df.groupby('peak_cluster'):
            # Group all hits for this cluster
            # Best: CAS, then name, then formula, as you requested!
            cas_scores = {}
            for _, row in group.iterrows():
                cas = row['hit_cas']
                si = row['hit_si']
                if cas:
                    if cas not in cas_scores or (si is not None and si > cas_scores[cas][0]):
                        cas_scores[cas] = (si, row['hit_names'], row['hit_formula'])
            # Pick best by CAS, fallback on name, fallback on formula
            if cas_scores:
                best_cas, (max_si, best_names, best_formula) = max(cas_scores.items(), key=lambda item: item[1][0] if item[1][0] is not None else -1)
                best_name = best_names[0] if best_names else ""
            else:
                # fallback by formula
                formula_scores = {}
                for _, row in group.iterrows():
                    formula = row['hit_formula']
                    si = row['hit_si']
                    if formula:
                        if formula not in formula_scores or (si is not None and si > formula_scores[formula][0]):
                            formula_scores[formula] = (si, row['hit_names'])
                if formula_scores:
                    best_formula, (max_si, best_names) = max(formula_scores.items(), key=lambda item: item[1][0] if item[1][0] is not None else -1)
                    best_name = best_names[0] if best_names else ""
                else:
                    # fallback: just any name/SI
                    si = group['hit_si'].max()
                    hit_row = group[group['hit_si'] == si].iloc[0]
                    best_formula = hit_row['hit_formula']
                    best_name = hit_row['hit_names'][0] if hit_row['hit_names'] else ""

            sis = [x for x in group['hit_si'] if x is not None]
            row_out = {
                'peak_cluster': cid,
                'mean_retention_time': round(group['retention_time'].mean(),3),
                'best_formula': best_formula,
                'best_name': best_name,
                'max_si': max(sis) if sis else None,
                'avg_si': round(np.mean(sis),1) if sis else None,
            }
            rows.append(row_out)

        return pd.DataFrame(rows)
    
    def fill_best_names_by_formula(self, df=None):
        if df is None:
            df = self.results  # or raise error if self.results is not defined
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filling best names"):
            if (not row['best_name'] or pd.isna(row['best_name'])) and row['best_formula']:
                try:
                    # Use formula search, pick first compound
                    compounds = pcp.get_compounds(row['best_formula'], 'formula')
                    if compounds:
                        # Try IUPAC name first, fallback to synonyms, then formula
                        best_name = getattr(compounds[0], 'iupac_name', None)
                        if not best_name and getattr(compounds[0], 'synonyms', None):
                            best_name = compounds[0].synonyms[0]
                        if not best_name:
                            best_name = compounds[0].molecular_formula
                        df.at[idx, 'best_name'] = best_name
                except Exception as e:
                    print(f"Failed formula lookup for {row['best_formula']}: {e}")
        return df
    
    def fill_best_names_by_pubchem(self, df=None):
        if df is None:
            raise ValueError("Pass a DataFrame to fill_best_names_by_formula")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filling best names"):
            if (not row['best_name'] or pd.isna(row['best_name'])):
                best_name = None
                # (1) Try PubChem by CAS
                cas = None
                for c in ['best_cas', 'CAS', 'cas']:  # in case your DataFrame changes
                    if c in row and row[c]:
                        cas = row[c]
                        break
                if not cas:
                    # Try to get CAS from all_hits
                    all_hits_cols = [c for c in df.columns if c.endswith('_all_hits')]
                    for col in all_hits_cols:
                        hit_info = row[col]
                        if isinstance(hit_info, str) and "CAS" in hit_info:
                            m = re.search(r'CAS ([0-9\- ]+)', hit_info)
                            if m:
                                cas = m.group(1).strip()
                                break
                if cas and cas != '0 - 00 - 0':
                    try:
                        compounds = pcp.get_compounds(cas, 'name')
                        if compounds and compounds[0].iupac_name:
                            best_name = compounds[0].iupac_name
                    except Exception as e:
                        print(f"Failed CAS lookup for {cas}: {e}")
                # (2) Fallback: take first synonym from the sample with max SI
                if not best_name:
                    all_hits_cols = [c for c in df.columns if c.endswith('_all_hits')]
                    found = False
                    for col in all_hits_cols:
                        hit_info = row[col]
                        if isinstance(hit_info, str) and "SI" in hit_info:
                            # First synonym before ' (SI ...'
                            best_name = hit_info.split(' (SI')[0]
                            found = True
                            break
                # (3) Fallback: use best_formula
                if not best_name:
                    best_name = row['best_formula']
                df.at[idx, 'best_name'] = best_name
        return df

    def compute_istd_corrected_area_table(self, peak_table_df, istd_retention_time, rt_tolerance=0.05):
        """
        For all samples, compute a table:
        - rows: samples
        - columns: best_name (from cross-correlation table)
        - values: area percent for the identified peak (ISTD corrected, if ISTD present), else NaN
        Also adds group, barcode, and SI scores.

        Args:
            peak_table_df (pd.DataFrame): Output table from cross_correlate_peaks()
            istd_retention_time (float): Retention time for ISTD
            rt_tolerance (float): Tolerance for RT matching

        Returns:
            pd.DataFrame: rows = samples, columns = best_name + group + barcode + SI columns, values = ISTD-corrected area %
        """
        sample_names = []
        group_names = []
        barcodes = []
        peak_names = list(peak_table_df['best_name'])

        # Find ISTD cluster number based on closest RT
        ist_rt_diffs = abs(peak_table_df['mean_retention_time'] - istd_retention_time)
        istd_peak_cluster = peak_table_df.iloc[ist_rt_diffs.idxmin()]['peak_cluster']

        area_matrix = []
        si_matrix = []

        # Map cluster to best_formula for later
        cluster_to_formula = dict(zip(peak_table_df['peak_cluster'], peak_table_df['best_formula']))
        cluster_to_name = dict(zip(peak_table_df['peak_cluster'], peak_table_df['best_name']))
        cluster_to_mean_rt = dict(zip(peak_table_df['peak_cluster'], peak_table_df['mean_retention_time']))

        for sample in self.samples:
            sname = sample['sample_info'].get('sample_name', '')
            group = sample.get('group', '')
            barcode = sample['sample_info'].get('barcode', '')
            sample_names.append(sname)
            group_names.append(group)
            barcodes.append(barcode)

            # Build a map from cluster number to area percent (raw), and SI
            cluster_area = {}
            cluster_si = {}
            for peak in sample['peaks']:
                rt = peak.get('retention_time')
                area = peak.get('area', 0.0)
                # Find which cluster this peak matches (by RT within tolerance)
                matched_cluster = None
                for cluster, mean_rt in zip(peak_table_df['peak_cluster'], peak_table_df['mean_retention_time']):
                    if abs(rt - mean_rt) <= rt_tolerance:
                        matched_cluster = cluster
                        break
                if matched_cluster is not None:
                    cluster_area[matched_cluster] = area
                    # Find SI for this peak (best SI among library hits)
                    best_si = None
                    for hit in peak.get('library_hits', []):
                        si = hit.get('similarity_index', None)
                        if best_si is None or (si is not None and si > best_si):
                            best_si = si
                    cluster_si[matched_cluster] = best_si

            # Get ISTD area for this sample
            istd_area = cluster_area.get(istd_peak_cluster, None)

            # Prepare row: for each peak_name, compute ISTD-corrected area%
            row = []
            si_row = []
            for cluster, name in zip(peak_table_df['peak_cluster'], peak_table_df['best_name']):
                raw_area = cluster_area.get(cluster, None)
                si_value = cluster_si.get(cluster, None)
                if istd_area is not None and raw_area is not None:
                    value = raw_area / istd_area if istd_area != 0 else None
                else:
                    value = None
                row.append(value)
                si_row.append(si_value)
            area_matrix.append(row)
            si_matrix.append(si_row)

        # Build the dataframe
        area_df = pd.DataFrame(area_matrix, columns=peak_names, index=sample_names)
        si_cols = [f"{name}_SI" for name in peak_names]
        si_df = pd.DataFrame(si_matrix, columns=si_cols, index=sample_names)
        area_df['group'] = group_names
        area_df['barcode'] = barcodes
        # Concatenate SI columns to the end
        final_df = pd.concat([area_df, si_df], axis=1)
        return final_df




class HPLCGroupAnalyzer:
    def __init__(self, samples, rt_tolerance=0.05):
        self.samples = samples
        self.rt_tolerance = rt_tolerance

    # -------- convenience: sample lookup by barcode/name --------
    @staticmethod
    def _norm_barcode_key(x):
        s = str(x).strip()
        # zero-pad pure digits to 10 chars (common barcode format)
        if s.replace(" ", "").isdigit():
            s = s.replace(" ", "")
            try:
                s = str(int(s)).zfill(10)
            except Exception:
                pass
        return s.lower()

    def get_sample(self, key: str, repeat: int | None = None):
        """
        Return the sample dict whose 'barcode' (or descriptive name) matches 'key'.
        - Matching is case-insensitive; numeric strings are zero-padded to 10 digits.
        - If repeat is provided, also require sample['repeat'] == repeat.
        - Returns the first match if multiple repeats exist and repeat is None.
        - Returns None if not found.

        Examples
        --------
        s = analyzer.get_sample('V0 Toluene')
        s = analyzer.get_sample('4000000020')
        s = analyzer.get_sample('4000000020', repeat=1)
        s = analyzer.get_sample('PS-M Blue', repeat=3)
        """
        want = self._norm_barcode_key(key)
        cand = None
        for s in self.samples:
            bc = s.get('barcode')
            if bc is None:
                continue
            if self._norm_barcode_key(bc) == want:
                if repeat is None or int(s.get('repeat', 1)) == int(repeat):
                    return s
                # remember first matching barcode in case repeat not found
                if cand is None:
                    cand = s
        return cand

    def match_qual_quant(self, sample):
        """
        Match QuickReport peaks (or legacy quant_results) to QualResults by nearest RT.

        Inputs from sample dict (as produced by batch_parser):
          - qual_results: optional DataFrame parsed from QualResults CSV (may be empty)
          - quickreport:  DataFrame with columns ['Name','rt_min','area']

        Fallback to legacy 'quant_results' if 'quickreport' is missing.
        """
        qual = sample.get('qual_results', pd.DataFrame()).copy()
        quant = sample.get('quickreport', pd.DataFrame()).copy()
        if quant.empty:
            quant = sample.get('quant_results', pd.DataFrame()).copy()
        if qual.empty or quant.empty:
            return qual  # Nothing to match

        # Normalize Qual RT column
        if 'RT (min)' not in qual.columns and 'RT [min]' in qual.columns:
            qual = qual.rename(columns={'RT [min]': 'RT (min)'})
        if 'RT (min)' not in qual.columns:
            print(f"qual_results columns: {qual.columns}")
            raise KeyError("qual_results missing 'RT (min)' column")

        # Normalize QuickReport/Quant columns into a common shape
        q = quant.copy()
        cols_l = {str(c).strip().lower(): c for c in q.columns}
        # Map rt column
        if 'rt [min]' in q.columns:
            rt_col = 'RT [min]'
        elif 'rt [min]' in cols_l:  # unlikely
            rt_col = cols_l['rt [min]']
        elif 'rt (min)' in q.columns:
            rt_col = 'RT (min)'
        elif 'rt (min)' in cols_l:
            rt_col = cols_l['rt (min)']
        elif 'rt_min' in cols_l:
            rt_col = cols_l['rt_min']
        elif 'rt' in cols_l:
            rt_col = cols_l['rt']
        else:
            # cannot proceed
            print(f"quant/quickreport missing RT column. Columns: {list(q.columns)}")
            return qual

        # Name and Area columns
        name_col = cols_l.get('name', next((c for c in q.columns if str(c).strip().lower()=="name"), None))
        area_col = cols_l.get('area', next((c for c in q.columns if str(c).strip().lower()=="area"), None))

        # Create a canonical view with RT [min]
        q_can = q.copy()
        if str(rt_col) != 'RT [min]':
            q_can = q_can.rename(columns={rt_col: 'RT [min]'})
        if name_col is not None and str(name_col) != 'Name':
            q_can = q_can.rename(columns={name_col: 'Name'})
        if area_col is not None and str(area_col) != 'Area':
            q_can = q_can.rename(columns={area_col: 'Area'})

        # Convert RT columns to numeric and drop NaNs
        qual['RT (min)'] = pd.to_numeric(qual['RT (min)'], errors='coerce')
        q_can['RT [min]'] = pd.to_numeric(q_can['RT [min]'], errors='coerce')
        qual_peaks = qual[qual['RT (min)'].notnull()].reset_index(drop=True)
        quant_peaks = q_can[q_can['RT [min]'].notnull()].reset_index(drop=True)

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
                for col in quant_peaks.columns:
                    merged.loc[min_idx, f'quant_{col}'] = row[col]
        # (Optional) final brute-force filter: only keep rows where RT and Width are real numbers
        merged = merged[pd.to_numeric(merged['RT (min)'], errors='coerce').notnull()]
        if 'Width (min)' in merged.columns:
            merged = merged[pd.to_numeric(merged['Width (min)'], errors='coerce').notnull()]
        return merged

    def align_dad_to_rid_by_peak(
        self,
        sample: dict,
        time_range: tuple,
        dad_nm: float,
        *,
        smoothing: bool = True,
        smooth_window_pts: int = 7,
        smooth_method: str = "median",  # 'median' or 'mean'
        plot: bool = False,
    ):
        """
        Align DAD to RID by matching the biggest peak (by height) within a time window.

        - Keeps RID times unchanged and shifts all DAD channels by a single offset.
        - Writes aligned DAD to sample['chrom_dad_aligned'] and metadata to
          sample['alignment'] = {'mode': 'peak', 'window': (tmin, tmax), 'dad_nm': <effective>, 'dt': dt}

        Parameters
        ----------
        sample : dict
            HPLC/GPC sample dict containing 'chrom_rid' and 'chrom_dad' DataFrames (relabelled).
        time_range : (float, float)
            Time window in minutes (tmin, tmax) used to find the biggest peak.
        dad_nm : float
            Target DAD wavelength (nm); the closest available channel is used.
        smoothing : bool
            Apply light smoothing before peak detection.
        smooth_window_pts : int
            Rolling window size in points for smoothing.
        smooth_method : str
            'median' (default) or 'mean'.
        plot : bool
            If True, returns a plotly Figure overlaying RID and DAD in the window with peak markers.

        Returns
        -------
        dt : float or None
            The applied time offset (RID_peak_time - DAD_peak_time). None if alignment failed.
        fig : plotly.graph_objects.Figure or None
            Returned only if plot=True; otherwise None.
        """
        import re as _re

        rid = sample.get('chrom_rid')
        dad = sample.get('chrom_dad')
        if not isinstance(rid, pd.DataFrame) or rid.empty or not isinstance(dad, pd.DataFrame) or dad.empty:
            return None, None if plot else None

        # Ensure indexes are time in minutes
        def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
            if df.index.name is None or 'time' not in str(df.index.name).lower():
                return df.set_index(df.columns[0])
            return df

        rid = _ensure_time_index(rid).copy()
        dad = _ensure_time_index(dad).copy()
        rid.index = pd.to_numeric(rid.index, errors='coerce')
        dad.index = pd.to_numeric(dad.index, errors='coerce')
        # drop rows where index is NaN
        rid = rid[~pd.isna(rid.index)]
        dad = dad[~pd.isna(dad.index)]

        # Pick RID series (prefer 'Signal (... )' or single column)
        rid_col = None
        for c in rid.columns:
            cs = str(c)
            if cs.startswith('Signal (') and cs.endswith(')'):
                rid_col = c
                break
        if rid_col is None and rid.shape[1] == 1:
            rid_col = rid.columns[0]
        if rid_col is None:
            # fallback: first numeric column
            num_cols = [c for c in rid.columns if pd.api.types.is_numeric_dtype(rid[c])]
            rid_col = num_cols[0] if num_cols else rid.columns[0]

        # Pick DAD column closest to dad_nm using names like 'Signal 254 nm (...)'
        dad_cols = []  # list of (wl, col)
        for c in dad.columns:
            m = _re.search(r"(?i)signal\s+(\d+(?:\.\d+)?)\s*nm", str(c))
            if m:
                try:
                    dad_cols.append((float(m.group(1)), c))
                except Exception:
                    continue
        effective_nm = None
        if dad_cols:
            arr = np.array([w for (w, _) in dad_cols], dtype=float)
            idx = int(np.argmin(np.abs(arr - float(dad_nm))))
            effective_nm, dad_col = dad_cols[idx]
        else:
            # fallback: single column or first numeric column
            dad_col = dad.columns[0]

        # Slice to time window
        tmin, tmax = float(time_range[0]), float(time_range[1])
        rid_slice = rid.loc[(rid.index >= tmin) & (rid.index <= tmax), rid_col].astype(float)
        dad_slice = dad.loc[(dad.index >= tmin) & (dad.index <= tmax), dad_col].astype(float)
        if rid_slice.empty or dad_slice.empty:
            return None, None if plot else None

        # Optional smoothing
        if smoothing and smooth_window_pts and smooth_window_pts > 1:
            win = int(max(1, smooth_window_pts))
            if smooth_method.lower() == 'mean':
                rid_s = rid_slice.rolling(window=win, center=True, min_periods=max(1, win // 2)).mean()
                dad_s = dad_slice.rolling(window=win, center=True, min_periods=max(1, win // 2)).mean()
            else:  # median
                rid_s = rid_slice.rolling(window=win, center=True, min_periods=max(1, win // 2)).median()
                dad_s = dad_slice.rolling(window=win, center=True, min_periods=max(1, win // 2)).median()
            # fallback to unsmoothed if all-NaN
            if rid_s.notna().any():
                rid_slice = rid_s
            if dad_s.notna().any():
                dad_slice = dad_s

        # Find peak maxima (by height)
        try:
            t_rid_peak = float(rid_slice.idxmax())
        except Exception:
            t_rid_peak = None
        try:
            t_dad_peak = float(dad_slice.idxmax())
        except Exception:
            t_dad_peak = None
        if t_rid_peak is None or t_dad_peak is None:
            return None, None if plot else None

        dt = float(t_rid_peak - t_dad_peak)

        # Apply shift to all DAD channels (global offset)
        dad_aligned = dad.copy()
        try:
            new_index = dad_aligned.index.astype(float) + dt
        except Exception:
            new_index = pd.to_numeric(dad_aligned.index, errors='coerce') + dt
        dad_aligned.index = new_index
        dad_aligned.index.name = rid.index.name or 'Time (min.)'

        # Store results
        sample['chrom_dad_aligned'] = dad_aligned
        sample['alignment'] = {
            'mode': 'peak',
            'window': (tmin, tmax),
            'dad_nm': float(effective_nm) if effective_nm is not None else float(dad_nm),
            'dt': dt,
        }

        # Optional plot
        fig = None
        if plot:
            try:
                import plotly.graph_objs as go
                from plotly.subplots import make_subplots
                # Dual y-axes: RID on left (primary), DAD on right (secondary)
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Continuous traces
                fig.add_trace(
                    go.Scatter(x=rid_slice.index, y=rid_slice.values, mode='lines', name=f'RID: {rid_col}'),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(x=dad_slice.index, y=dad_slice.values, mode='lines', name=f'DAD: {dad_col}'),
                    secondary_y=True,
                )

                # Peak markers on their respective axes
                fig.add_trace(
                    go.Scatter(x=[t_rid_peak], y=[rid_slice.loc[t_rid_peak]],
                               mode='markers', name='RID peak', marker=dict(color='red', size=10)),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(x=[t_dad_peak], y=[dad_slice.loc[t_dad_peak]],
                               mode='markers', name='DAD peak', marker=dict(color='blue', size=10)),
                    secondary_y=True,
                )

                fig.update_layout(
                    title=f"Peak alignment window [{tmin:.3f}, {tmax:.3f}] min | dt={dt:.4f} min",
                    xaxis_title='Time (min.)',
                )
                fig.update_yaxes(title_text='RID (a.u.)', secondary_y=False)
                fig.update_yaxes(title_text='DAD (a.u.)', secondary_y=True)
            except Exception:
                fig = None

        return (dt, fig) if plot else dt

    
    def extract_dad_peak_spectra(self, hplc_sample, merged_peaks, rt_shift=-0.043):
        """
        For each RID peak (from merged_peaks DataFrame), extract mean DAD spectrum in the RT window.
        Saves list of dicts to hplc_sample['dad_peak_spectra'].
        Each dict contains: {'peak_idx', 'rt', 'width', 'mean_spectrum', 'channels', 'rt_mask'}
        """
        # Prefer aligned DAD when available
        chrom_dad = hplc_sample.get('chrom_dad_aligned')
        if (chrom_dad is None) or chrom_dad.empty:
            chrom_dad = hplc_sample.get('chrom_dad')
        if chrom_dad is None or chrom_dad.empty:
            hplc_sample['dad_peak_spectra'] = []
            return

        # Derive wavelengths from column names like 'Signal 254 nm (mAU)'
        channels = list(chrom_dad.columns)
        wavelengths = []
        for c in channels:
            m = re.search(r"(?i)signal\s+(\d+(?:\.\d+)?)\s*nm", str(c))
            try:
                wavelengths.append(float(m.group(1)) if m else np.nan)
            except Exception:
                wavelengths.append(np.nan)

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
        if merged_df is None or isinstance(merged_df, float) or not isinstance(merged_df, pd.DataFrame) or merged_df.empty:
            return merged_df
        # Prefer Qual 'Area', else Quick/Quant area from merged as 'quant_Area' or 'quant_area'
        area_col = None
        if 'Area' in merged_df.columns:
            area_col = 'Area'
        elif 'quant_Area' in merged_df.columns:
            area_col = 'quant_Area'
        elif 'quant_area' in merged_df.columns:
            area_col = 'quant_area'
        if area_col is None:
            return merged_df
        area = pd.to_numeric(merged_df[area_col], errors='coerce').fillna(0)
        total_area = float(area.sum())
        keep = area / total_area * 100 >= float(min_area_percent) if total_area > 0 else False
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
