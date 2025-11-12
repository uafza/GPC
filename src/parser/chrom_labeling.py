# src/parser/chrom_labeling.py
from __future__ import annotations
from typing import Dict, Any, Optional
import re
import pandas as pd


def _extract_unit_from_analog_att(text: Optional[str]) -> Optional[str]:
    """
    Parse unit token from 'Analog Attenuation' like '1000 mAU' or '500000 nRIU'.
    Returns 'mAU', 'nRIU', etc. If nothing found, returns None.
    """
    if not text:
        return None
    m = re.search(r"([a-zA-Zµ]+)$", str(text).strip())
    return m.group(1) if m else None


def _rid_unit_from_method(method_struct: Dict[str, Any]) -> str:
    """
    Return RID unit exactly as reported (e.g., 'nRIU'), without normalization.
    Fallback to 'nRIU' if not found.
    """
    sec61 = method_struct.get("6 RID Method", {}).get("6.1 Analog Output", {})
    unit = _extract_unit_from_analog_att(sec61.get("Analog Attenuation"))
    return unit or "nRIU"


def _dad_unit_from_method(method_struct: Dict[str, Any]) -> str:
    sec31 = method_struct.get("3 DAD Method", {}).get("3.1 Analog Output", {})
    unit = _extract_unit_from_analog_att(sec31.get("Analog Attenuation"))
    return unit or "mAU"


def _dad_channel_wavelength_map(method_struct: Dict[str, Any]) -> Dict[str, float]:
    """
    Returns dict like {'A': 254.0, 'B': 210.0, ...} from 3.2 Signals → 'Signal table'.
    """
    out: Dict[str, float] = {}
    sec3 = method_struct.get("3 DAD Method", {})
    sig_node = sec3.get("3.2 Signals", {})
    df = None
    if isinstance(sig_node, dict) and isinstance(sig_node.get("Signal table"), pd.DataFrame):
        df = sig_node["Signal table"]
    elif isinstance(sig_node, pd.DataFrame):
        df = sig_node
    if not isinstance(df, pd.DataFrame) or df.empty:
        return out

    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)

    sig_col = next((c for c in df.columns if c.lower() == "signal"), None)
    wl_col  = next((c for c in df.columns if "wavelength" in c.lower() and "ref" not in c.lower()), None)
    if not sig_col or not wl_col:
        return out

    for _, r in df.iterrows():
        sig_txt = str(r.get(sig_col, ""))
        m = re.search(r"Signal\s+([A-H])", sig_txt, flags=re.I)
        if not m:
            continue
        ch = m.group(1).upper()
        try:
            wl = float(str(r.get(wl_col)).split()[0].replace(",", "."))
            out[ch] = wl
        except Exception:
            continue
    return out


def relabel_chromatograms(sample: Dict[str, Any]) -> None:
    """
    In-place relabeling of sample['chrom_dad'] and sample['chrom_rid'] using method_struct info.
    - DAD columns -> 'Signal <wl> nm (<unit>)'
    - RID column  -> 'Signal (<unit>)'
    - Index name  -> 'Time (min.)'
    """
    method_struct = sample.get("method_struct", {}) or sample.get("method", {}) or {}
    dad_unit = _dad_unit_from_method(method_struct)
    rid_unit = _rid_unit_from_method(method_struct)
    ch2wl = _dad_channel_wavelength_map(method_struct)

    # --- DAD ---
    dad = sample.get("chrom_dad")
    if isinstance(dad, pd.DataFrame) and not dad.empty:
        # Ensure index name
        dad.index.name = "Time (min.)"

        # Rename columns: detect channel letter from names like 'DAD_1A', 'DAD_1B_254', etc.
        new_cols = {}
        for c in dad.columns:
            c_str = str(c)
            m = re.search(r"DAD[_\-]?\s*1?([A-H])", c_str, flags=re.I)  # capture 'A' from 'DAD_1A'
            ch = m.group(1).upper() if m else None
            wl = ch2wl.get(ch) if ch else None
            if wl is not None:
                wl_int = int(round(wl))
                new_cols[c] = f"Signal {wl_int} nm ({dad_unit})"
            else:
                # fallback: keep old name but append unit if missing
                if f"({dad_unit})" in c_str:
                    new_cols[c] = c_str
                else:
                    new_cols[c] = f"{c_str} ({dad_unit})"
        dad.rename(columns=new_cols, inplace=True)
        sample["chrom_dad"] = dad

    # --- RID ---
    rid = sample.get("chrom_rid")
    if isinstance(rid, pd.DataFrame) and not rid.empty:
        # If raw CSV, make first column the index and rename second column
        if rid.shape[1] >= 2:
            # Heuristic: first column is time
            if rid.index.name is None or rid.index.name == "":
                rid = rid.set_index(rid.columns[0])
        rid.index.name = "Time (min.)"

        # If there is exactly one signal column, rename to 'Signal (unit)'
        if rid.shape[1] == 1:
            only_col = rid.columns[0]
            if f"({rid_unit})" in only_col:
                new_name = only_col
            else:
                # strip common tokens like 'RID1A' etc.
                new_name = f"Signal ({rid_unit})"
            rid.rename(columns={only_col: new_name}, inplace=True)
        else:
            # For multiple columns, append unit to each
            rid.rename(columns={c: f"{c} ({rid_unit})" if f"({rid_unit})" not in str(c) else c
                                for c in rid.columns}, inplace=True)

        sample["chrom_rid"] = rid
