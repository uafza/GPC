# calibration_parser.py  
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class GPCCalibrationParser:
    """
    Build a GPC calibration table from a single Agilent QuickReport DataFrame.
    The DataFrame must have columns: Name (str), rt_min (float), area (float).

    Expected EasiVials (you can extend/override via 'expected=...' in __init__):
      - PS-H_*, PS-M_* + V0 (Toluene)
    """

    DEFAULT_EXPECTED: Dict[str, Dict[str, List[float]]] = {
        # PS-H (High)
        "PS-H_Blue":  {"mp": [935000, 66600, 4950, 162],         "mass_mg": [0.4, 0.8, 1.2, 1.6]},
        "PS-H_Red":   {"mp": [6085000, 474500, 20140, 1180],     "mass_mg": [0.4, 0.8, 1.2, 1.6]},
        "PS-H_White": {"mp": [2811000, 182200, 11140, 580],      "mass_mg": [0.4, 0.8, 1.2, 1.6]},
        # PS-M (Medium)
        "PS-M_Blue":  {"mp": [87200, 11720, 1180, 162],          "mass_mg": [0.4, 0.8, 1.2, 1.6]},
        "PS-M_Red":   {"mp": [365000, 50700, 6920, 935],         "mass_mg": [0.4, 0.8, 1.2, 1.6]},
        "PS-M_White": {"mp": [182200, 26390, 3260, 370],         "mass_mg": [0.4, 0.8, 1.2, 1.6]},
    }

    def __init__(self, expected: Optional[Dict[str, Dict[str, List[float]]]] = None, signal_label: str = "RID"):
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
        for mp, mass in zip(spec["mp"], spec["mass_mg"]):
            hit = df_num[np.isclose(df_num["Name_num"], mp, atol=1e-12)]
            if hit.empty:
                hit = df_num[df_num["Name"].astype(str).str.strip() == str(mp)]
            if hit.empty:
                continue
            r = hit.sort_values(by="area", ascending=False).iloc[0]
            rows.append({
                "Exp. RT (min)": float(r["rt_min"]),
                "MW": mp,
                "Mass": mass,
                "Peak Area": r["area"],
                "Signal": self.signal_label,
                "Vial": vial_key,
            })
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

        out = pd.DataFrame(rows, columns=["Exp. RT (min)", "MW", "Mass", "Peak Area", "Signal", "Vial"])
        if not out.empty:
            out["sort_key"] = out.apply(lambda x: 1e6 if str(x["Vial"]) == "V0" else float(x["Exp. RT (min)"]), axis=1)
            out.sort_values("sort_key", inplace=True, ignore_index=True)
            out.drop(columns="sort_key", inplace=True)
        return out

    def write_output(self, df: pd.DataFrame, output_csv: str) -> None:
        header1 = ["Exp. RT (min)", "MW", "Mass", "Peak Area", "Signal", "Vial"]
        header2 = ["min", "Da", "mg", "nRU mg/mL", " nRU", "-", "-"]
        with open(output_csv, "w", newline="") as f:
            f.write(",".join(header1) + "\n")
            f.write(",".join(header2) + "\n")
            df.to_csv(f, index=False, header=False)
