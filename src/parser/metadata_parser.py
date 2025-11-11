import os
import re
import pandas as pd
import numpy as np

class MetadataParser:
    def __init__(self, metadata_dir="../data/metadata", project_code="A046"):
        self.metadata_dir = metadata_dir
        self.project_code = project_code

    # ---------- IO helpers ----------
    @staticmethod
    def universal_read(path, **kwargs):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path, **kwargs)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(path, **kwargs)
        else:
            raise ValueError(f"Unknown file extension: {ext}")

    def extract_from_filename(self, filename):
        m = re.search(r'(A\d+)_T(\d+)_R(\d+)_RP(\d+)', filename)
        return m.groups() if m else ('n/a', 'n/a', 'n/a', 'n/a')

    # ---------- barcode utils ----------
    @staticmethod
    def normalize_barcode(value):
        s = str(value).strip().replace(' ', '')
        try:
            f = float(s)
            if f.is_integer():
                s = str(int(f))
        except Exception:
            pass
        if s.isdigit():
            s = s.zfill(10)
        return s

    def is_barcode(self, value):
        return bool(re.fullmatch(r"\d{10}", self.normalize_barcode(value)))

    # ---------- loaders ----------
    def load_barcodes(self):
        files = [f for f in os.listdir(self.metadata_dir)
                if (self.project_code in f and "Barcodes" in f)]
        all_barcodes = []
        for file in files:
            df = self.universal_read(os.path.join(self.metadata_dir, file))
            project, task, run, repeat = self.extract_from_filename(file)
            df['Project'], df['Task'], df['Run'], df['Repeat'] = project, task, run, repeat

            # Normalize all relevant barcode columns (including Stock Solution Barcode)
            barcode_cols = [
                'Catalyst Vial Barcode',
                'ILS Vial Barcode',
                'Storage Vial Barcode',
                'HPLC Vial Barcode',
                'Stock Solution Vial Barcode',
                'Stock Solution Barcode',
            ]
            for col in barcode_cols:
                if col in df.columns:
                    df[col] = df[col].apply(self.normalize_barcode)
                    df[col] = df[col].replace(['', '-', 'nan', 'None', None, np.nan], 'n/a')

            if 'Substrate' in df.columns:
                df['Substrate'] = df['Substrate'].astype(str).str.strip()

            all_barcodes.append(df)

        if not all_barcodes:
            raise Exception("No barcode files found in directory!")

        return pd.concat(all_barcodes, ignore_index=True)

    def load_synthesis(self):
        csvp = os.path.join(self.metadata_dir, f"{self.project_code}_T01_R01_output.csv")
        xlsxp = os.path.join(self.metadata_dir, f"{self.project_code}_T01_R01_output.xlsx")
        if os.path.exists(csvp):
            synth = self.universal_read(csvp, header=1)
        elif os.path.exists(xlsxp):
            synth = self.universal_read(xlsxp, header=1)
        else:
            raise Exception("No synthesis file found!")

        colmap = {c.lower().strip(): c for c in synth.columns}
        vg1 = colmap.get('vial_g1_barcode', 'Vial_G1_Barcode')
        cid = colmap.get('custom_id', 'custom_id')
        batch = colmap.get('batch', 'batch')

        synth[vg1] = synth[vg1].apply(self.normalize_barcode)
        synth[cid] = synth[cid].astype(str).str.strip()
        return synth[[vg1, cid, batch]].rename(
            columns={vg1: 'Vial_G1_Barcode', cid: 'custom_id', batch: 'batch'}
        )

    # ---------- ILSLDNG helpers ----------
    def find_ilsl_file(self, run, repeat):
        pattern = f"{self.project_code}_T*_R{int(run):02d}_RP{int(repeat):01d}_ILSLDNG"
        candidates = [f for f in os.listdir(self.metadata_dir) if pattern in f]
        if not candidates:
            candidates = [f for f in os.listdir(self.metadata_dir)
                          if f"{self.project_code}_T" in f and
                          f"_R{int(run):02d}_" in f and
                          f"_RP{int(repeat):01d}_" in f and
                          "ILSLDNG" in f]
        return os.path.join(self.metadata_dir, candidates[0]) if candidates else None

    def _load_ilsl_filtered(self, run, repeat):
        """
        Keep ONLY rows where Source is a 10-digit barcode.
        Map: Source -> Catalyst Vial Barcode; Destination -> ILS Vial Barcode.
        Return:
        A) pair table  : ['ILS Vial Barcode','Catalyst Vial Barcode','Actual Weight (mg)','Target Weight (mg)']
        B) catalyst tbl: ['Catalyst Vial Barcode','Actual Weight (mg)','Target Weight (mg)']  (median per catalyst)
        """
        path = self.find_ilsl_file(run, repeat)
        if not path:
            emptyA = pd.DataFrame(columns=[
                'ILS Vial Barcode', 'Catalyst Vial Barcode',
                'Actual Weight (mg)', 'Target Weight (mg)'
            ])
            emptyB = pd.DataFrame(columns=[
                'Catalyst Vial Barcode', 'Actual Weight (mg)', 'Target Weight (mg)'
            ])
            return emptyA, emptyB

        ilsl = self.universal_read(path)
        if 'Source' not in ilsl.columns or 'Destination Vial Barcode' not in ilsl.columns:
            emptyA = pd.DataFrame(columns=[
                'ILS Vial Barcode', 'Catalyst Vial Barcode',
                'Actual Weight (mg)', 'Target Weight (mg)'
            ])
            emptyB = pd.DataFrame(columns=[
                'Catalyst Vial Barcode', 'Actual Weight (mg)', 'Target Weight (mg)'
            ])
            return emptyA, emptyB

        # Only barcode sources
        ilsl['Source_norm'] = ilsl['Source'].apply(self.normalize_barcode)
        ilsl = ilsl[ilsl['Source_norm'].astype(str).str.fullmatch(r'\d{10}')].copy()
        if ilsl.empty:
            emptyA = pd.DataFrame(columns=[
                'ILS Vial Barcode', 'Catalyst Vial Barcode',
                'Actual Weight (mg)', 'Target Weight (mg)'
            ])
            emptyB = pd.DataFrame(columns=[
                'Catalyst Vial Barcode', 'Actual Weight (mg)', 'Target Weight (mg)'
            ])
            return emptyA, emptyB

        ilsl['Dest_norm'] = ilsl['Destination Vial Barcode'].apply(self.normalize_barcode)

        # Rename to standard columns
        weights_pair = ilsl.rename(columns={
            'Source_norm': 'Catalyst Vial Barcode',
            'Dest_norm':   'ILS Vial Barcode',
            'Actual Weight': 'Actual Weight (mg)',
            'Target Weight': 'Target Weight (mg)',
        })

        # Ensure both weight columns exist even if missing in file
        if 'Actual Weight (mg)' not in weights_pair.columns:
            weights_pair['Actual Weight (mg)'] = pd.NA
        if 'Target Weight (mg)' not in weights_pair.columns:
            weights_pair['Target Weight (mg)'] = pd.NA

        weights_pair = weights_pair[
            ['ILS Vial Barcode', 'Catalyst Vial Barcode', 'Actual Weight (mg)', 'Target Weight (mg)']
        ].drop_duplicates()

        # Catalyst-only (median per catalyst for robustness)
        weights_cat = (
            weights_pair
            .groupby('Catalyst Vial Barcode', as_index=False)
            .agg({'Actual Weight (mg)': 'median', 'Target Weight (mg)': 'median'})
        )

        return weights_pair, weights_cat

    
    # helper to build the stock→substrate mapping
    def _build_stock_substrate_map(self) -> pd.DataFrame:
        """
        Returns a 2-col DataFrame: ['Stock Solution Barcode', 'Substrate']
        Uses 'Stock Solution Barcode' if present, else 'Stock Solution Vial Barcode'.
        Only keeps rows where the stock code is a 10-digit barcode.
        """
        bdf = self.load_barcodes()
        stock_col = None
        if 'Stock Solution Barcode' in bdf.columns:
            stock_col = 'Stock Solution Barcode'
        elif 'Stock Solution Vial Barcode' in bdf.columns:
            stock_col = 'Stock Solution Vial Barcode'
        if stock_col is None or 'Substrate' not in bdf.columns:
            return pd.DataFrame(columns=['Stock Solution Barcode', 'Substrate'])

        out = bdf[[stock_col, 'Substrate']].dropna().copy()
        out[stock_col] = out[stock_col].astype(str).str.strip().apply(self.normalize_barcode)
        out = out[out[stock_col].str.fullmatch(r'\d{10}')]

        # normalize header to a single name for downstream joins
        out = out.rename(columns={stock_col: 'Stock Solution Barcode'})
        out['Substrate'] = out['Substrate'].astype(str).str.strip()
        out = out.drop_duplicates(subset=['Stock Solution Barcode'])
        return out

    def _load_all_ilsl_candidates(self, barcode_df):
        if not {'Run', 'Repeat'}.issubset(barcode_df.columns):
            return (pd.DataFrame(columns=['ILS Vial Barcode', 'Catalyst Vial Barcode',
                                          'Actual Weight (mg)', 'Target Weight (mg)']),
                    pd.DataFrame(columns=['Catalyst Vial Barcode',
                                          'Actual Weight (mg)', 'Target Weight (mg)']))
        pairs, cats = [], []
        for (r, rp), _ in barcode_df.groupby(['Run', 'Repeat']):
            wA, wB = self._load_ilsl_filtered(r, rp)
            if not wA.empty: pairs.append(wA)
            if not wB.empty: cats.append(wB)
        wA = pd.concat(pairs, ignore_index=True).drop_duplicates(['ILS Vial Barcode', 'Catalyst Vial Barcode']) if pairs else \
             pd.DataFrame(columns=['ILS Vial Barcode', 'Catalyst Vial Barcode', 'Actual Weight (mg)', 'Target Weight (mg)'])
        wB = pd.concat(cats,  ignore_index=True).drop_duplicates(['Catalyst Vial Barcode']) if cats else \
             pd.DataFrame(columns=['Catalyst Vial Barcode', 'Actual Weight (mg)', 'Target Weight (mg)'])
        return wA, wB

    # ---------- build master metadata ----------
    def build_master_metadata(self):
        bar = self.load_barcodes()
        syn = self.load_synthesis()

        # custom_id via Catalyst ↔ Vial_G1_Barcode
        meta = pd.merge(
            bar, syn,
            left_on='Catalyst Vial Barcode',
            right_on='Vial_G1_Barcode',
            how='left'
        )
        meta['custom_id'] = meta['custom_id'].fillna('n/a')
        meta['batch'] = meta['batch'].fillna('n/a')

        # Weights: try pair first, then catalyst-only; choose the one with more hits
        w_pair, w_cat = self._load_all_ilsl_candidates(meta)

        m_pair = meta.merge(w_pair, how='left', on=['ILS Vial Barcode','Catalyst Vial Barcode'])
        hits_pair = m_pair['Actual Weight (mg)'].notna().sum()

        m_cat  = meta.merge(w_cat,  how='left', on='Catalyst Vial Barcode')
        hits_cat  = m_cat['Actual Weight (mg)'].notna().sum()

        merged = m_pair if hits_pair >= hits_cat else m_cat

        # Ensure numeric
        if 'Actual Weight (mg)' not in merged.columns: merged['Actual Weight (mg)'] = pd.NA
        if 'Target Weight (mg)' not in merged.columns: merged['Target Weight (mg)'] = pd.NA
        merged['Actual Weight (mg)'] = pd.to_numeric(merged['Actual Weight (mg)'], errors='coerce')
        merged['Target Weight (mg)'] = pd.to_numeric(merged['Target Weight (mg)'], errors='coerce')

        # Final selection/order
        cols = [
        'Project','Task','Run','Repeat',
        'Catalyst Vial Barcode','custom_id','batch',
        'ILS Vial Barcode','Storage Vial Barcode','HPLC Vial Barcode',
        'Substrate','Target Weight (mg)','Actual Weight (mg)' 
        ]
        for c in cols:
            if c not in merged.columns: merged[c] = 'n/a'
        return merged[cols]


    # Excel export
    def parse(self, output_file='barcode_results_full.xlsx'):
        df = self.build_master_metadata()
        df.to_excel(os.path.join(self.metadata_dir, output_file), index=False)
        print(f"Done! Output: {os.path.join(self.metadata_dir, output_file)}, Rows: {len(df)}")

    # ---------- final_table enrichment ----------
    @staticmethod
    def _norm_series(s):
        return s.astype(str).str.strip().apply(MetadataParser.normalize_barcode)

    def _ensure_keys_from_barcodes(self, final_table):
        out = final_table.copy()
        if 'HPLC Vial Barcode' in out.columns and (
            'Catalyst Vial Barcode' not in out.columns or 'ILS Vial Barcode' not in out.columns
        ):
            b = self.load_barcodes()[['HPLC Vial Barcode','Catalyst Vial Barcode','ILS Vial Barcode']].drop_duplicates()
            out['HPLC Vial Barcode'] = self._norm_series(out['HPLC Vial Barcode'])
            b['HPLC Vial Barcode'] = self._norm_series(b['HPLC Vial Barcode'])
            out = out.merge(b, how='left', on='HPLC Vial Barcode', suffixes=('', '_bc'))
        return out

    def enrich_final_table(self, final_table: pd.DataFrame, metadata_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Enriches final_table with:
        - custom_id & batch via Catalyst ↔ Vial_G1_Barcode
        - weights from ILSLDNG (auto-choose pair vs catalyst-only)
        - Substrate/custom_id for stock-solution samples ("<Substrate>_initial")
        - "Stock Solution Vial Barcode" column (AFTER Storage Vial Barcode)
        - For TRUE stock rows (where the sample barcode itself is a stock barcode):
            • Project/Task/Run/Repeat are filled from Barcodes (if available)
            • batch is set empty ("")
            • HPLC Vial Barcode is set to the stock barcode if missing

        Final column order (handled at the end):
            Project, Task, Run, Repeat, batch, custom_id,
            Catalyst Vial Barcode, ILS Vial Barcode, HPLC Vial Barcode,
            Storage Vial Barcode, Stock Solution Vial Barcode,
            Substrate, Target Weight (mg), Actual Weight (mg)
        """
        import pandas as pd
        import numpy as np

        out = final_table.copy()
        out.columns = (
            out.columns.astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # expose index 'barcode' if needed
        if 'barcode' not in out.columns:
            idx_name = str(out.index.name).lower() if out.index.name is not None else ''
            if idx_name == 'barcode':
                out['barcode'] = out.index

        # backfill Catalyst/ILS from HPLC barcode if only HPLC available
        if 'HPLC Vial Barcode' in out.columns and (
            'Catalyst Vial Barcode' not in out.columns or 'ILS Vial Barcode' not in out.columns
        ):
            b = self.load_barcodes()[['HPLC Vial Barcode','Catalyst Vial Barcode','ILS Vial Barcode']].drop_duplicates()
            b.columns = b.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            norm = lambda s: s.astype(str).str.strip().apply(self.normalize_barcode)
            out['HPLC Vial Barcode'] = norm(out['HPLC Vial Barcode'])
            b['HPLC Vial Barcode']   = norm(b['HPLC Vial Barcode'])
            out = out.merge(b, how='left', on='HPLC Vial Barcode', suffixes=('', '_bc'))

        # build master metadata if not given
        if metadata_df is None:
            metadata_df = self.build_master_metadata()
        metadata_df = metadata_df.copy()
        metadata_df.columns = metadata_df.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

        # merge custom_id + batch via Catalyst barcode
        if 'Catalyst Vial Barcode' in out.columns:
            norm = lambda s: s.astype(str).str.strip().apply(self.normalize_barcode)
            out['Catalyst Vial Barcode'] = norm(out['Catalyst Vial Barcode'])
            meta_c = metadata_df[['Catalyst Vial Barcode','custom_id','batch']].drop_duplicates('Catalyst Vial Barcode').copy()
            meta_c['Catalyst Vial Barcode'] = norm(meta_c['Catalyst Vial Barcode'])
            out = out.merge(meta_c, how='left', on='Catalyst Vial Barcode')

        # --- ensure 'batch' is object/string dtype early to avoid dtype-mismatch warnings later ---
        if 'batch' in out.columns and not pd.api.types.is_object_dtype(out['batch'].dtype):
            out['batch'] = out['batch'].astype('object')

        # choose weights mapping (pair vs catalyst-only)
        if {'ILS Vial Barcode','Catalyst Vial Barcode'}.issubset(out.columns):
            norm = lambda s: s.astype(str).str.strip().apply(self.normalize_barcode)
            out['ILS Vial Barcode'] = norm(out['ILS Vial Barcode'])
            out['Catalyst Vial Barcode'] = norm(out['Catalyst Vial Barcode'])

            meta_pair = metadata_df[['ILS Vial Barcode','Catalyst Vial Barcode','Actual Weight (mg)','Target Weight (mg)']].drop_duplicates()
            meta_pair['ILS Vial Barcode'] = norm(meta_pair['ILS Vial Barcode'])
            meta_pair['Catalyst Vial Barcode'] = norm(meta_pair['Catalyst Vial Barcode'])

            m_pair = out.merge(meta_pair, how='left', on=['ILS Vial Barcode','Catalyst Vial Barcode'])
            hits_pair = m_pair['Actual Weight (mg)'].notna().sum()

            meta_cat = metadata_df[['Catalyst Vial Barcode','Actual Weight (mg)','Target Weight (mg)']].drop_duplicates('Catalyst Vial Barcode').copy()
            meta_cat['Catalyst Vial Barcode'] = norm(meta_cat['Catalyst Vial Barcode'])
            m_cat  = out.merge(meta_cat,  how='left', on='Catalyst Vial Barcode')
            hits_cat  = m_cat['Actual Weight (mg)'].notna().sum()

            out = m_pair if hits_pair >= hits_cat else m_cat

        # stock-solution handling
        bcodes = self.load_barcodes().copy()
        bcodes.columns = bcodes.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

        stock_col = 'Stock Solution Barcode' if 'Stock Solution Barcode' in bcodes.columns \
                    else ('Stock Solution Vial Barcode' if 'Stock Solution Vial Barcode' in bcodes.columns else None)

        if 'Stock Solution Vial Barcode' not in out.columns:
            out['Stock Solution Vial Barcode'] = pd.NA

        if stock_col and 'Substrate' in bcodes.columns:
            norm = lambda s: s.astype(str).str.strip().apply(self.normalize_barcode)

            stock_map = bcodes[[stock_col, 'Substrate']].dropna(subset=[stock_col]).copy()
            stock_map[stock_col] = norm(stock_map[stock_col])
            stock_map['Substrate'] = stock_map['Substrate'].astype(str).str.strip()
            stock_map = stock_map[stock_map[stock_col].str.fullmatch(r'\d{10}')].drop_duplicates(subset=[stock_col])
            stock_map = stock_map.rename(columns={stock_col: '__stock_barcode__', 'Substrate': '__stock_substrate__'})

            meta_cols = [c for c in ['Project','Task','Run','Repeat'] if c in bcodes.columns]
            stock_meta = None
            if meta_cols:
                stock_meta = bcodes[[stock_col] + meta_cols].dropna(subset=[stock_col]).copy()
                stock_meta[stock_col] = norm(stock_meta[stock_col])
                stock_meta = stock_meta.drop_duplicates(subset=[stock_col]).rename(columns={stock_col: '__stock_barcode__'})

            candidate_keys = [k for k in ['HPLC Vial Barcode', 'GCMS Vial Barcode', 'barcode'] if k in out.columns]

            # (a) TRUE stock samples: key equals stock barcode
            for key in candidate_keys:
                tmp = out[[key]].copy()
                tmp[key] = norm(tmp[key])
                tmp = tmp.merge(stock_map, how='left', left_on=key, right_on='__stock_barcode__')

                hit_mask = tmp['__stock_substrate__'].notna()
                if not hit_mask.any():
                    continue
                hit_idx = hit_mask[hit_mask].index

                for col in ['Substrate', 'custom_id', 'Stock Solution Vial Barcode']:
                    if col not in out.columns:
                        out[col] = pd.NA

                out.loc[hit_idx, 'Substrate'] = tmp.loc[hit_idx, '__stock_substrate__'].astype(str).str.strip()
                out.loc[hit_idx, 'custom_id'] = tmp.loc[hit_idx, '__stock_substrate__'].astype(str).str.strip() + '_initial'
                out.loc[hit_idx, 'Stock Solution Vial Barcode'] = tmp.loc[hit_idx, '__stock_barcode__']

                # HPLC Vial Barcode: fill with stock barcode if missing
                if 'HPLC Vial Barcode' in out.columns:
                    cur = out.loc[hit_idx, 'HPLC Vial Barcode'].astype(str).str.strip()
                    need_fill = cur.isna() | (cur.eq('')) | (cur.str.lower().isin(['nan', 'none']))
                    need_idx = need_fill[need_fill].index
                    out.loc[need_idx, 'HPLC Vial Barcode'] = tmp.loc[need_idx, '__stock_barcode__']

                # Project/Task/Run/Repeat for stock rows
                if stock_meta is not None:
                    add = tmp.loc[hit_idx, ['__stock_barcode__']].merge(stock_meta, how='left', on='__stock_barcode__')
                    add.index = hit_idx
                    for mcol in meta_cols:
                        if mcol not in out.columns:
                            out[mcol] = pd.NA
                        out.loc[hit_idx, mcol] = add[mcol].values

                # --- ensure object dtype before assigning empty strings to 'batch' ---
                if 'batch' in out.columns and not pd.api.types.is_object_dtype(out['batch'].dtype):
                    out['batch'] = out['batch'].astype('object')
                if 'batch' in out.columns:
                    out.loc[hit_idx, 'batch'] = ""

            # (b) Non-stock rows: map to stock via HPLC/GCMS
            if 'HPLC Vial Barcode' in out.columns and 'HPLC Vial Barcode' in bcodes.columns:
                map_hplc = bcodes[['HPLC Vial Barcode', stock_col]].dropna().copy()
                map_hplc['HPLC Vial Barcode'] = norm(map_hplc['HPLC Vial Barcode'])
                map_hplc[stock_col] = norm(map_hplc[stock_col])
                map_hplc = map_hplc.rename(columns={stock_col: '__stock_from_hplc__'})
                out['HPLC Vial Barcode'] = norm(out['HPLC Vial Barcode'])
                out = out.merge(map_hplc, how='left', on='HPLC Vial Barcode')
                out['Stock Solution Vial Barcode'] = out['Stock Solution Vial Barcode'].fillna(out['__stock_from_hplc__'])
                out.drop(columns=['__stock_from_hplc__'], inplace=True)

            if 'GCMS Vial Barcode' in out.columns and 'GCMS Vial Barcode' in bcodes.columns:
                map_gcms = bcodes[['GCMS Vial Barcode', stock_col]].dropna().copy()
                map_gcms['GCMS Vial Barcode'] = norm(map_gcms['GCMS Vial Barcode'])
                map_gcms[stock_col] = norm(map_gcms[stock_col])
                map_gcms = map_gcms.rename(columns={stock_col: '__stock_from_gcms__'})
                out['GCMS Vial Barcode'] = norm(out['GCMS Vial Barcode'])
                out = out.merge(map_gcms, how='left', on='GCMS Vial Barcode')
                out['Stock Solution Vial Barcode'] = out['Stock Solution Vial Barcode'].fillna(out['__stock_from_gcms__'])
                out.drop(columns=['__stock_from_gcms__'], inplace=True)

            out['Stock Solution Vial Barcode'] = out['Stock Solution Vial Barcode'].astype(str).str.strip()
            out.loc[out['Stock Solution Vial Barcode'].isin(['', 'nan', 'None']), 'Stock Solution Vial Barcode'] = pd.NA

        # numeric coercion for weights
        if 'Actual Weight (mg)' in out.columns:
            out['Actual Weight (mg)'] = pd.to_numeric(out['Actual Weight (mg)'], errors='coerce')
        if 'Target Weight (mg)' in out.columns:
            out['Target Weight (mg)'] = pd.to_numeric(out['Target Weight (mg)'], errors='coerce')

        # drop helper column
        if 'barcode' in out.columns:
            out = out.drop(columns=['barcode'])

        # filter rows without custom_id
        if 'custom_id' in out.columns:
            keep_mask = (
                out['custom_id'].notna()
                & out['custom_id'].astype(str).str.strip().ne('')
                & out['custom_id'].astype(str).str.lower().ne('n/a')
            )
            out = out.loc[keep_mask].reset_index(drop=True)

        # final column order (puts Stock Solution Vial Barcode after Storage Vial Barcode)
        out.columns = out.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        meta_order = [
            'Project','Task','Run','Repeat','batch','custom_id',
            'Catalyst Vial Barcode','ILS Vial Barcode','HPLC Vial Barcode',
            'Storage Vial Barcode','Stock Solution Vial Barcode',
            'Substrate','Target Weight (mg)','Actual Weight (mg)'
        ]
        present_meta = [c for c in meta_order if c in out.columns]
        remaining = [c for c in out.columns if c not in present_meta]
        area_cols     = [c for c in remaining if str(c).startswith('Area|')]
        area_pct_cols = [c for c in remaining if str(c).startswith('Area%|')]
        quant_cols    = [c for c in remaining if str(c).startswith('quant_')]
        used = set(area_cols + area_pct_cols + quant_cols)
        others = [c for c in remaining if c not in used]

        substrate_terms = []
        if 'Substrate' in out.columns:
            substrate_terms = (
                out['Substrate'].dropna().astype(str).str.strip().str.lower()
                .replace({'nan': ''}).unique().tolist()
            )
            substrate_terms = [s for s in substrate_terms if s]

        def has_substrate(col: str) -> bool:
            s = str(col).lower()
            return any(term in s for term in substrate_terms)

        istd_terms = ('istd', 'internal standard', 'internal standart', 'internal std')
        def is_istd(col: str) -> bool:
            s = str(col).lower()
            return any(t in s for t in istd_terms)

        orig_pos = {c: i for i, c in enumerate(out.columns)}
        def sort_group(cols):
            return sorted(cols, key=lambda c: (1 if is_istd(c) else 0,
                                            0 if has_substrate(c) else 1,
                                            orig_pos.get(c, 10_000)))

        area_cols_sorted     = sort_group(area_cols)
        area_pct_cols_sorted = sort_group(area_pct_cols)
        quant_cols_sorted    = sort_group(quant_cols)

        new_cols = present_meta + area_cols_sorted + area_pct_cols_sorted + quant_cols_sorted + others
        new_cols = [c for c in new_cols if c in out.columns]
        out = out.reindex(columns=new_cols)

        return out


        def reorder_final_table_columns(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Reorder columns as requested:
            1) Fixed metadata columns first (in this exact order):
            Project, Task, Run, Repeat, batch, custom_id,
            Catalyst Vial Barcode, ILS Vial Barcode, HPLC Vial Barcode,
            Storage Vial Barcode, Stock Solution Vial Barcode, Substrate,
            Target Weight (mg), Actual Weight (mg)
            2) Then Area|* columns
            3) Then Area%|* columns
            4) Then quant_* columns

            Within each group:
            - Columns containing any Substrate term (from the rows) come first
            - ISTD columns go last (match 'istd'/'internal standard/standart/std/Triglyme')
            """
            # Normalize headers to avoid whitespace/name mismatches
            out = df.copy()
            out.columns = (
                out.columns.astype(str)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

            meta_order = [
                'Project','Task','Run','Repeat','batch','custom_id',
                'Catalyst Vial Barcode','ILS Vial Barcode','HPLC Vial Barcode',
                'Storage Vial Barcode','Stock Solution Vial Barcode',
                'Substrate','Target Weight (mg)','Actual Weight (mg)'
            ]
            present_meta = [c for c in meta_order if c in out.columns]

            remaining = [c for c in out.columns if c not in present_meta]
            area_cols      = [c for c in remaining if str(c).startswith('Area|')]
            area_pct_cols  = [c for c in remaining if str(c).startswith('Area%|')]
            quant_cols     = [c for c in remaining if str(c).startswith('quant_')]
            used = set(area_cols + area_pct_cols + quant_cols)
            others         = [c for c in remaining if c not in used]  # keep any stragglers at the end

            # Substrate & ISTD detectors
            substrate_terms = []
            if 'Substrate' in out.columns:
                substrate_terms = (
                    out['Substrate']
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace({'nan': ''})
                    .unique()
                    .tolist()
                )
                substrate_terms = [s for s in substrate_terms if s]

            def has_substrate(col: str) -> bool:
                s = str(col).lower()
                return any(term in s for term in substrate_terms)

            istd_terms = ('istd', 'internal standard', 'internal standart', 'internal std', 'triglyme')
            def is_istd(col: str) -> bool:
                s = str(col).lower()
                return any(t in s for t in istd_terms)

            orig_pos = {c: i for i, c in enumerate(out.columns)}
            def sort_group(cols):
                # (ISTD last, substrate first, original position stable)
                return sorted(
                    cols,
                    key=lambda c: (
                        1 if is_istd(c) else 0,
                        0 if has_substrate(c) else 1,
                        orig_pos.get(c, 10_000),
                    )
                )

            area_cols_sorted     = sort_group(area_cols)
            area_pct_cols_sorted = sort_group(area_pct_cols)
            quant_cols_sorted    = sort_group(quant_cols)

            new_cols = present_meta + area_cols_sorted + area_pct_cols_sorted + quant_cols_sorted + others
            new_cols = [c for c in new_cols if c in out.columns]  # defensive
            return out.reindex(columns=new_cols)
