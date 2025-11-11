# -*- coding: utf-8 -*-
# src/gcms_parser/compute_results.py

from __future__ import annotations
import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

ISTD_TERMS = ('istd', 'internal standard', 'internal standart', 'internal std', 'triglyme')


class HPLCResultsComputer:
    """
    Compute conversions (substrate-based and product-based), selectivities, and yields.

    mass_normalize=True (default) → results are %/mg (divide by 'Actual Weight (mg)' then ×100)
    mass_normalize=False          → results are %      (no mass division; still ×100)

    Definitions per group (Area / Area% / quant_*):
      - Normalize signals: Area & Area% / ISTD; quant_* as-is.
      - Baseline S0: median of normalized substrate peak across '<Substrate>_initial' rows.
      - Substrate-based conversion:        conv = 1 − S/S0  (optional clip to [0,1]).
      - Product-based conversion (new):     conv_from_prod = Σ_p ( p̂ / S0 ).
      - Selectivity of product p:           sel_p  = p̂ / Σ_p p̂
      - Yield of product p:                 yld_p  = p̂ / S0
    """

    def __init__(self, mass_normalize: bool = True, clip_conversion_0_1: bool = True):
        self.mass_normalize = mass_normalize
        self.clip_conversion_0_1 = clip_conversion_0_1

    # ----------------------------- public API -----------------------------

    def compute(self, final_table: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compute metrics and return a new DataFrame with columns inserted BEFORE each signal block.
        If save_path is provided, writes "<save_path_stem>_RESULTS.csv".
        """
        df = final_table.copy()
        df.columns = df.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

        if 'Substrate' not in df.columns:
            raise ValueError("Final table is missing 'Substrate' column.")
        if 'custom_id' not in df.columns:
            df['custom_id'] = ""

        groups = self._detect_groups(df)              # list[(group_key, cols)]
        baselines = self._compute_baselines(df, groups)

        mass = pd.to_numeric(df.get('Actual Weight (mg)', np.nan), errors='coerce')
        unit_suffix = " [%/mg]" if self.mass_normalize else " [%]"

        for group_key, group_cols in groups:
            if not group_cols:
                continue

            istd_col = self._first_istd_column(group_cols) if group_key in ('Area', 'Area%') else None
            norm_maps = [self._row_norm_values(df.loc[i], group_cols, group_key, istd_col) for i in df.index]

            conv_vals:            List[float] = [np.nan] * len(df)
            conv_from_prod_vals:  List[float] = [np.nan] * len(df)

            product_cols_all = [c for c in group_cols if not self._is_istd(c)]
            sel_data: Dict[str, List[float]] = {c: [np.nan] * len(df) for c in product_cols_all}
            yld_data: Dict[str, List[float]] = {c: [np.nan] * len(df) for c in product_cols_all}

            for row_idx in range(len(df)):
                subs  = str(df.iloc[row_idx]['Substrate']).strip()
                subs_l = subs.lower()
                sub_col = self._find_substrate_column(group_cols, subs)

                products = [c for c in product_cols_all if c != sub_col]
                nmap = norm_maps[row_idx]
                S  = nmap.get(sub_col, np.nan) if sub_col else np.nan
                S0 = baselines.get((subs_l, group_key), np.nan)

                # ---- substrate-based conversion (fraction) ----
                conv_f = np.nan
                if (not pd.isna(S)) and (not pd.isna(S0)) and S0 != 0:
                    conv_f = 1.0 - (S / S0)
                if str(df.iloc[row_idx].get('custom_id', '')).strip().lower() == f"{subs_l}_initial":
                    conv_f = 0.0
                if not pd.isna(conv_f) and self.clip_conversion_0_1:
                    conv_f = float(np.clip(conv_f, 0.0, 1.0))

                # ---- products: selectivity & yield (fractions) ----
                pvals = [nmap.get(c, np.nan) for c in products]
                pvals = [v for v in pvals if (v is not None and not pd.isna(v))]
                sum_prod = float(np.nansum(pvals)) if pvals else np.nan

                # accumulate product-based conversion as Σ (p̂ / S0)
                conv_from_prod_f = 0.0 if (not pd.isna(S0) and S0 != 0 and products) else np.nan

                for c in products:
                    pv = nmap.get(c, np.nan)

                    sel_f = pv / sum_prod if (not pd.isna(pv) and not pd.isna(sum_prod) and sum_prod != 0) else np.nan
                    yld_f = pv / S0      if (not pd.isna(pv) and not pd.isna(S0)      and S0 != 0)      else np.nan

                    if not pd.isna(yld_f) and not pd.isna(conv_from_prod_f):
                        conv_from_prod_f += yld_f

                    # mass-normalize per-product metrics if requested
                    if self.mass_normalize:
                        m = mass.iloc[row_idx]
                        if not pd.isna(m) and m != 0:
                            sel_f = sel_f / m if not pd.isna(sel_f) else np.nan
                            yld_f = yld_f / m if not pd.isna(yld_f) else np.nan

                    # scale to percent
                    sel_data[c][row_idx] = sel_f * 100.0 if not pd.isna(sel_f) else np.nan
                    yld_data[c][row_idx] = yld_f * 100.0 if not pd.isna(yld_f) else np.nan

                # mass-normalize both conversion metrics
                conv_out = conv_f
                conv_from_prod_out = conv_from_prod_f
                if self.mass_normalize:
                    m = mass.iloc[row_idx]
                    if not pd.isna(m) and m != 0:
                        conv_out = conv_out / m if not pd.isna(conv_out) else np.nan
                        conv_from_prod_out = conv_from_prod_out / m if not pd.isna(conv_from_prod_out) else np.nan

                # scale to percent
                conv_vals[row_idx]           = conv_out * 100.0 if not pd.isna(conv_out) else np.nan
                conv_from_prod_vals[row_idx] = conv_from_prod_out * 100.0 if not pd.isna(conv_from_prod_out) else np.nan

            # ---- assemble new block BEFORE this group's raw columns ----
            new_block = pd.DataFrame(index=df.index)
            new_block[f"Conversion|{group_key}{unit_suffix}"] = conv_vals
            new_block[f"ConversionFromProducts|{group_key}{unit_suffix}"] = conv_from_prod_vals

            for c in product_cols_all:
                label = str(c).split('|', 1)[1] if '|' in str(c) else str(c)
                new_block[f"Selectivity|{group_key}|{label}{unit_suffix}"] = sel_data[c]
                new_block[f"Yield|{group_key}|{label}{unit_suffix}"]       = yld_data[c]

            first_col = group_cols[0] if group_cols else None
            df = self._insert_before_block(df, first_col, new_block)

        if save_path:
            stem, ext = os.path.splitext(save_path)
            out_path = f"{stem}_RESULTS.csv"
            df.to_csv(out_path, index=False)
            print(f"Results saved to: {out_path}")

        return df

    def compute_from_csv(self, csv_path: str, save: bool = True) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        return self.compute(df, save_path=csv_path if save else None)

    # ----------------------------- helpers -----------------------------

    @staticmethod
    def _is_istd(colname: str) -> bool:
        s = str(colname).lower()
        return any(term in s for term in ISTD_TERMS)

    @staticmethod
    def _split_prefix(col: str) -> Tuple[str, str]:
        s = str(col)
        if '|' in s:
            a, b = s.split('|', 1)
            return a, b
        return s, s

    def _detect_groups(self, df: pd.DataFrame) -> List[Tuple[str, List[str]]]:
        order = []
        seen = set()
        for c in df.columns:
            prefix, _ = self._split_prefix(c)
            if prefix in ('Area', 'Area%') or prefix.startswith('quant_'):
                if prefix not in seen:
                    order.append(prefix)
                    seen.add(prefix)
        groups: List[Tuple[str, List[str]]] = []
        for g in order:
            cols = [c for c in df.columns if str(c).startswith(f"{g}|")]
            groups.append((g, cols))
        return groups

    @staticmethod
    def _first_istd_column(cols: List[str]) -> Optional[str]:
        for c in cols:
            s = str(c).lower()
            if any(term in s for term in ISTD_TERMS):
                return c
        return None

    @staticmethod
    def _find_substrate_column(cols: List[str], substrate: str) -> Optional[str]:
        if not substrate or pd.isna(substrate):
            return None
        s = str(substrate).strip().lower()
        if not s:
            return None
        for c in cols:
            label = str(c).split('|', 1)[1] if '|' in str(c) else str(c)
            if s in label.lower():
                return c
        return None

    @staticmethod
    def _row_norm_values(row: pd.Series, cols: List[str], group_key: str, istd_col: Optional[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        istd_val = np.nan
        if group_key in ('Area', 'Area%') and istd_col is not None:
            try:
                istd_val = float(row.get(istd_col, np.nan))
            except Exception:
                istd_val = np.nan
            if pd.isna(istd_val) or istd_val == 0:
                istd_val = np.nan

        for c in cols:
            v = row.get(c, np.nan)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            if group_key in ('Area', 'Area%') and not pd.isna(istd_val):
                out[c] = v / istd_val if not pd.isna(v) else np.nan
            else:
                out[c] = v
        return out

    def _compute_baselines(self, df: pd.DataFrame, groups: List[Tuple[str, List[str]]]) -> Dict[Tuple[str, str], float]:
        baselines: Dict[Tuple[str, str], float] = {}
        if 'Substrate' not in df.columns:
            return baselines

        mask_init = (
            df.get('custom_id', '').astype(str).str.strip().str.lower()
            == (df['Substrate'].astype(str).str.strip().str.lower() + '_initial')
        )
        df_init = df[mask_init]

        for subs in df_init['Substrate'].astype(str).unique():
            subs_l = subs.strip().lower()
            sub_rows = df_init[df_init['Substrate'].astype(str).str.strip().str.lower() == subs_l]

            for group_key, cols in groups:
                if not cols:
                    continue
                istd_col = self._first_istd_column(cols) if group_key in ('Area', 'Area%') else None
                sub_col = self._find_substrate_column(cols, subs)
                if sub_col is None:
                    baselines[(subs_l, group_key)] = np.nan
                    continue

                vals = []
                for _, r in sub_rows.iterrows():
                    nmap = self._row_norm_values(r, cols, group_key, istd_col)
                    v = nmap.get(sub_col, np.nan)
                    if v is not None and not pd.isna(v):
                        vals.append(v)
                baselines[(subs_l, group_key)] = float(np.median(vals)) if vals else np.nan

        return baselines

    @staticmethod
    def _insert_before_block(df: pd.DataFrame,
                             first_col_of_block: Optional[str],
                             new_cols_df: pd.DataFrame) -> pd.DataFrame:
        if new_cols_df is None or new_cols_df.empty:
            return df
        if first_col_of_block is None or first_col_of_block not in df.columns:
            for c in new_cols_df.columns:
                df[c] = new_cols_df[c]
            return df

        cols = list(df.columns)
        insert_idx = cols.index(first_col_of_block)
        left = cols[:insert_idx]
        right = cols[insert_idx:]

        drop_cols = [c for c in new_cols_df.columns if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        df = pd.concat([df[left], new_cols_df, df[right]], axis=1)
        return df


# ----------------------------- CLI support -----------------------------

def _main():
    if len(sys.argv) < 2:
        print("Usage: python -m gcms_parser.compute_results <final_table.csv>")
        sys.exit(1)
    path = sys.argv[1]
    comp = HPLCResultsComputer()
    comp.compute_from_csv(path, save=True)


if __name__ == "__main__":
    _main()
