# src/parser/method_normalizer.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
import re
import pandas as pd


def _get(sec: Dict[str, Any] | None, key: str, default=None):
    if isinstance(sec, dict):
        return sec.get(key, default)
    return default


def _parse_float(s: Any) -> Optional[float]:
    if s is None:
        return None
    try:
        # keep only first token, strip units (e.g. "1.000 mL/min" -> "1.000")
        token = str(s).strip().split()[0].replace(",", ".")
        return float(token)
    except Exception:
        return None


def _parse_percent(s: Any) -> Optional[float]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    try:
        txt = str(s).strip().replace(",", ".")
        # accept "100.00 %", "0.00%", "100"
        m = re.search(r"([-+]?\d+(?:\.\d+)?)", txt)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _dad_signals_from_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(df, pd.DataFrame) or df.empty:
        return out
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)
    # try to find columns
    sig_col = next((c for c in df.columns if c.lower() == "signal"), None)
    wl_col  = next((c for c in df.columns if "wavelength" in c.lower() and "ref" not in c.lower()), None)
    bw_col  = next((c for c in df.columns if "bandwidth" in c.lower() and "ref" not in c.lower()), None)
    use_ref = next((c for c in df.columns if "use ref" in c.lower()), None)
    ref_wl  = next((c for c in df.columns if "ref wavel" in c.lower()), None)
    ref_bw  = next((c for c in df.columns if "ref bandw" in c.lower()), None)

    for _, r in df.iterrows():
        sig_txt = str(r.get(sig_col, ""))
        m = re.search(r"Signal\s+([A-H])", sig_txt, flags=re.I)
        chan = m.group(1).upper() if m else None
        wl   = _parse_float(r.get(wl_col))
        bw   = _parse_float(r.get(bw_col))
        uref = str(r.get(use_ref, "")).strip().lower() == "yes" if use_ref else None
        rw   = _parse_float(r.get(ref_wl)) if ref_wl else None
        rb   = _parse_float(r.get(ref_bw)) if ref_bw else None
        out.append({
            "channel": chan,
            "wavelength_nm": wl,
            "bandwidth_nm":  bw,
            "use_ref":       uref,
            "ref_wavelength_nm": rw,
            "ref_bandwidth_nm":  rb,
        })
    return out


def normalize_method_info(method_struct: Dict[str, Any]) -> Dict[str, Any]:
    """
    Consume the structured dict from method_parser_structured and return a compact view.
    """
    out: Dict[str, Any] = {
        "meta": {
            "method_path": None,
            "created": None,
            "modified": None,
            "creator": None,
            "modifier": None,
            "status": None,
            "description": None,
        },
        "oven": {"temperature_c": None},
        "pump": {"flow_ml_min": None, "primary_channel": None, "solvents": {}},
        "detectors": {"DAD": {"lamp_required": None, "peakwidth_info": None},
                      "RID": {"peakwidth_min": None, "acquire_A": None, "polarity": None, "temp_c": None}},
        "signals": {"DAD": [], "RID": []},
    }

    # -------- 1) META from "1 Method Information"
    sec1 = method_struct.get("1 Method Information", {})
    out["meta"]["method_path"] = _get(sec1, "Last Saved As", None) or _get(sec1, "Acquisition Method", None)
    out["meta"]["created"]     = _get(sec1, "Created")
    out["meta"]["modified"]    = _get(sec1, "Modified")
    out["meta"]["creator"]     = _get(sec1, "Creator")
    out["meta"]["modifier"]    = _get(sec1, "Modifier")
    out["meta"]["status"]      = _get(sec1, "Method Status")
    out["meta"]["description"] = _get(sec1, "Description")

    # -------- 2) OVEN temperature
    # Left/Right control (2.1 / 2.2) or RID optical unit temp (6.2)
    sec21 = method_struct.get("2 Column Comp. Method", {}).get("2.1 Left Temperature Control", {})
    sec22 = method_struct.get("2 Column Comp. Method", {}).get("2.2 Right Temperature Control", {})
    sec62 = method_struct.get("6 RID Method", {}).get("6.2 Optical Unit Temperature", {})
    t = None
    for src in (sec21, sec22, sec62):
        t = t or _parse_float(_get(src, "Left Temperature")) \
              or _parse_float(_get(src, "Temperature"))
    out["oven"]["temperature_c"] = t

    # -------- 3) PUMP: flow, primary channel, solvent composition
    sec4 = method_struct.get("4 Quat. Pump Method", {})
    out["pump"]["flow_ml_min"]     = _parse_float(_get(sec4, "Flow"))
    out["pump"]["primary_channel"] = (_get(sec4, "Primary Channel") or "").strip() or None

    # Solvent Composition table: handle both correct and fallback keys
    solv_df = None
    if isinstance(sec4, dict):
        if isinstance(sec4.get("Solvent Composition"), pd.DataFrame):
            solv_df = sec4["Solvent Composition"]
        else:
            # fallback to any DF in this section (e.g., older fallback key)
            for k, v in sec4.items():
                if isinstance(v, pd.DataFrame):
                    solv_df = v
                    break
    if isinstance(solv_df, pd.DataFrame) and not solv_df.empty:
        cols = {c: str(c).strip() for c in solv_df.columns}
        df = solv_df.rename(columns=cols)
        # expected columns: Channel, Name 1, Used, Percent (leniently parsed)
        for _, r in df.iterrows():
            ch = str(r.get("Channel", "")).strip()
            if ch not in list("ABCD"):
                continue
            name = None
            # prefer 'Name 1' if present; otherwise pick first non-% token
            if "Name 1" in df.columns:
                name = str(r.get("Name 1", "")).strip() or None
            if not name:
                for c in df.columns:
                    val = str(r.get(c, "")).strip()
                    if val and "%" not in val and c.lower() not in ("channel", "used", "percent"):
                        name = val
                        break
            pct = _parse_percent(r.get("Percent"))
            out["pump"]["solvents"][ch] = {"name": name or None, "percent": pct}

    # -------- 4) DAD detector & signals
    sec3 = method_struct.get("3 DAD Method", {})
    out["detectors"]["DAD"]["lamp_required"]  = _get(sec3, "UV Lamp Required")
    out["detectors"]["DAD"]["peakwidth_info"] = _get(sec3, "Peakwidth")

    # Signals table
    sig_node = sec3.get("3.2 Signals", {})
    sig_df = None
    if isinstance(sig_node, dict) and isinstance(sig_node.get("Signal table"), pd.DataFrame):
        sig_df = sig_node["Signal table"]
    elif isinstance(sig_node, pd.DataFrame):  # ultra-fallback
        sig_df = sig_node
    out["signals"]["DAD"] = _dad_signals_from_table(sig_df) if isinstance(sig_df, pd.DataFrame) else []

    # -------- 5) RID detector (6 RID Method)
    sec6 = method_struct.get("6 RID Method", {})
    # peakwidth like "> 0.2 min (...)" -> 0.2
    pw_txt = _get(sec6, "Peakwidth")
    if pw_txt:
        m = re.search(r"([\d.,]+)\s*min", str(pw_txt))
        if m:
            out["detectors"]["RID"]["peakwidth_min"] = float(m.group(1).replace(",", "."))
    out["detectors"]["RID"]["acquire_A"] = _get(sec6, "Acquire Signal A")
    out["detectors"]["RID"]["polarity"]  = _get(sec6, "Signal Polarity")
    # If not found earlier, use 6.2 temp explicitly
    out["detectors"]["RID"]["temp_c"] = out["detectors"]["RID"]["temp_c"] or _parse_float(_get(sec62, "Temperature"))

    # add a simple signals.RID view so you see it alongside DAD
    if str(_get(sec6, "Acquire Signal A", "")).strip().lower() in ("yes", "true", "1"):
        out["signals"]["RID"] = [{"channel": "A"}]
    else:
        out["signals"]["RID"] = []

    return out
