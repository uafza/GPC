# calibration_parser.py
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

LEGACY_EXPECTED: Dict[str, Dict[str, List[float]]] = {
    # PS-H (High)
    "PS-H_Blue":  {"mp": [935000, 66600, 4950, 162],         "mass_mg": [0.4, 0.8, 1.2, 1.6]},
    "PS-H_Red":   {"mp": [6085000, 474500, 20140, 1180],     "mass_mg": [0.4, 0.8, 1.2, 1.6]},
    "PS-H_White": {"mp": [2811000, 182200, 11140, 580],      "mass_mg": [0.4, 0.8, 1.2, 1.6]},
    # PS-M (Medium)
    "PS-M_Blue":  {"mp": [87200, 11720, 1180, 162],          "mass_mg": [0.4, 0.8, 1.2, 1.6]},
    "PS-M_Red":   {"mp": [365000, 50700, 6920, 935],         "mass_mg": [0.4, 0.8, 1.2, 1.6]},
    "PS-M_White": {"mp": [182200, 26390, 3260, 370],         "mass_mg": [0.4, 0.8, 1.2, 1.6]},
}


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "")
    if not s or s == "-" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_expected_from_coa(json_path: Path, prefix: str) -> Dict[str, Dict[str, Any]]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    expected: Dict[str, Dict[str, Any]] = {}
    for row in data.get("table", []):
        color = str(row.get("Vial Code", "")).strip()
        if not color or color == "-":
            continue
        color = color.title()
        entry = {
            "vial_code": color,
            "iv_dl_per_g": _coerce_float(row.get("IV")),
            "mw_light_scattering": _coerce_float(row.get("MW")),
            "mn": _coerce_float(row.get("Mn")),
            "mw": _coerce_float(row.get("Mw")),
            "mw_mn": _coerce_float(row.get("Mw/Mn")),
            "mp": _coerce_float(row.get("Mp")),
            "mass_mg": _coerce_float(row.get("Mass")),
        }
        key = f"PS-{prefix}_{color}"
        spec = expected.setdefault(key, {"entries": [], "mp": [], "mass_mg": []})
        spec["entries"].append(entry)
        if entry["mp"] is not None:
            spec["mp"].append(entry["mp"])
        if entry["mass_mg"] is not None:
            spec["mass_mg"].append(entry["mass_mg"])
    return expected


def _build_default_expected() -> Dict[str, Dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    coa_dir = project_root / "data" / "CoA"
    expected: Dict[str, Dict[str, Any]] = {}
    for prefix, filename in (("H", "COA_Polystyrene_High.json"), ("M", "COA_Polystyrene_Medium.json")):
        expected.update(_load_expected_from_coa(coa_dir / filename, prefix))
    return expected if expected else LEGACY_EXPECTED.copy()


_DEFAULT_EXPECTED = _build_default_expected()

class GPCCalibrationParser:
    """
    Build a GPC calibration table from a single Agilent QuickReport DataFrame.
    The DataFrame must have columns: Name (str), rt_min (float), area (float).

    Expected EasiVials (you can extend/override via 'expected=...' in __init__):
      - PS-H_*, PS-M_* + V0 (Toluene)
    """

    DEFAULT_EXPECTED: Dict[str, Dict[str, Any]] = _DEFAULT_EXPECTED

    def __init__(self, expected: Optional[Dict[str, Dict[str, Any]]] = None, signal_label: str = "RID"):
        self.EXPECTED = expected.copy() if expected else self.DEFAULT_EXPECTED.copy()
        self.signal_label = signal_label

    @staticmethod
    def _to_number(x):
        s = str(x).strip().replace(",", "")
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return np.nan

    def _parse_v0(self, df_norm: pd.DataFrame) -> Optional[dict]:
        hit = df_norm[df_norm["Name"].astype(str).str.strip().str.startswith("V0 (Toluene")]
        if hit.empty:
            return None
        r = hit.iloc[0]
        return {
            "Exp. RT (min)": float(r["rt_min"]),
            "MW": "",
            "Mass": "",
            "Peak Area": r["area"],
            "Signal": self.signal_label,
            "Vial": "V0",
        }

    def _parse_easivial(self, vial_key: str, df_norm: pd.DataFrame) -> List[dict]:
        spec = self.EXPECTED[vial_key]
        rows = []
        # Name column can be exact Mp (as string) OR numeric in your QuickReport
        # We match first on numeric equality, else exact string equality.
        df_num = df_norm.copy()
        df_num["Name_num"] = df_num["Name"].apply(self._to_number)
        entry_specs = spec.get("entries")
        if entry_specs:
            expected_rows = entry_specs
        else:
            expected_rows = [{"mp": mp, "mass_mg": mass} for mp, mass in zip(spec.get("mp", []), spec.get("mass_mg", []))]

        for entry in expected_rows:
            mp = entry.get("mp")
            mass = entry.get("mass_mg")
            if mp is None or mass is None:
                continue
            hit = df_num[np.isclose(df_num["Name_num"], mp, atol=1e-12)]
            if hit.empty:
                hit = df_num[df_num["Name"].astype(str).str.strip() == str(mp)]
            if hit.empty:
                continue
            r = hit.sort_values(by="area", ascending=False).iloc[0]
            row_payload = {
                "Exp. RT (min)": float(r["rt_min"]),
                "MW": mp,
                "Mass": mass,
                "Peak Area": r["area"],
                "Signal": self.signal_label,
                "Vial": vial_key,
            }
            iv_val = entry.get("iv_dl_per_g")
            if iv_val is not None:
                row_payload["iv_dl_per_g"] = iv_val
            rows.append(row_payload)
        return rows

    def from_quick_report_df(self, df_quick: pd.DataFrame, vial_key: Optional[str] = None) -> pd.DataFrame:
        """
        Build the calibration table from one normalized QuickReport DF.
        If 'vial_key' is provided (e.g., 'PS-M_Blue'), only parse that vial + V0.
        Otherwise, try all known vials and keep rows that are found.
        """
        need_cols = {"Name", "rt_min", "area"}
        if not need_cols.issubset(set(df_quick.columns)):
            raise ValueError(f"QuickReport DF must contain columns {need_cols}, got {df_quick.columns.tolist()}")

        rows: List[dict] = []
        # V0 (Toluene) if present
        v0 = self._parse_v0(df_quick)
        if v0:
            rows.append(v0)

        vial_keys = [vial_key] if vial_key else list(self.EXPECTED.keys())
        for k in vial_keys:
            if k in self.EXPECTED:
                rows.extend(self._parse_easivial(k, df_quick))

        base_cols = ["Exp. RT (min)", "MW", "Mass", "Peak Area", "Signal", "Vial"]
        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=base_cols)

        for col in base_cols:
            if col not in out.columns:
                out[col] = np.nan
        # order base columns first, keep any extras (e.g., iv_dl_per_g) afterward
        extra_cols = [c for c in out.columns if c not in base_cols]
        ordered_cols = base_cols + extra_cols
        out = out[ordered_cols]

        out["sort_key"] = out.apply(lambda x: 1e6 if str(x["Vial"]) == "V0" else float(x["Exp. RT (min)"]), axis=1)
        out.sort_values("sort_key", inplace=True, ignore_index=True)
        out.drop(columns="sort_key", inplace=True)
        return out

    def write_output(self, df: pd.DataFrame, output_csv: str) -> None:
        header1 = ["Exp. RT (min)", "MW", "Mass", "Peak Area", "Signal", "Vial"]
        header2 = ["min", "Da", "mg", "nRU mg/mL", " nRU", "-", "-"]
        df_to_write = df.copy()
        for col in header1:
            if col not in df_to_write.columns:
                df_to_write[col] = ""
        df_to_write = df_to_write[header1]
        with open(output_csv, "w", newline="") as f:
            f.write(",".join(header1) + "\n")
            f.write(",".join(header2) + "\n")
            df_to_write.to_csv(f, index=False, header=False)
