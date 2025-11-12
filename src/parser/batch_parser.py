import os
import re
import pandas as pd
import io


from parser.method_normalizer import normalize_method_info
from parser.method_parser_structured import parse_method_report_file
from parser.chrom_labeling import relabel_chromatograms


class HPLCBatchParser:
    def __init__(self, base_folder, metadata_parser=None):
        self.base_folder = base_folder
        self.method_folder = os.path.join(base_folder, 'AqMethodReport')
        self.chrom_folder  = os.path.join(base_folder, 'Chromatograms')
        self.qual_folder   = os.path.join(base_folder, 'QualResults')   # optional
        self.quick_folder  = os.path.join(base_folder, 'QuickReport')    # NEW (replaces Quant_Results)
        self.samples = []
        self.metadata_parser = metadata_parser  # pass your MetadataParser instance

    def parse_batch(self):
        barcode_versions = dict()
        # Collect from QuickReport folder (replacing old Quant_Results source)
        if not os.path.exists(self.quick_folder):
            print(f"WARNING: QuickReport folder not found: {self.quick_folder}")
            return []

        for f in os.listdir(self.quick_folder):
            if f.lower().endswith('.csv'):
                bc, repeat = self._extract_barcode_and_repeat(f)
                if not bc:
                    continue
                barcode_versions.setdefault(bc, []).append((f, repeat))

        self.samples = []
        for bc, files in sorted(barcode_versions.items()):
            files = sorted(files, key=lambda x: x[1])
            for (quick_file, repeat) in files:
                quick_path = os.path.join(self.quick_folder, quick_file)
                # Optional: sample type from QuickReport; default to "sample"
                sample_type = self._get_sample_type_from_quick(quick_path) or "sample"
                if sample_type.lower() != "sample":
                    continue

                sample = self.parse_sample(bc, repeat=repeat, quick_filename=quick_file)
                if sample:
                    # ---- Metadata lookup ----
                    metadata = {}
                    if self.metadata_parser is not None:
                        meta_df = self.metadata_parser.load_barcodes()
                        barcode_norm = self.metadata_parser.normalize_barcode(bc)
                        candidates = []
                        for col in ['HPLC Vial Barcode', 'Catalyst Vial Barcode']:
                            if col in meta_df.columns:
                                found = meta_df[meta_df[col] == barcode_norm]
                                if not found.empty:
                                    candidates.append(found)
                        if candidates:
                            metadata = candidates[0].iloc[0].to_dict()
                        else:
                            metadata = {}
                    sample['sample_type'] = "hplc_sample"
                    sample['metadata'] = metadata
                    self.samples.append(sample)
        return self.samples

    @staticmethod
    def _extract_barcode_and_repeat(filename):
        """
        Extract sample identifier and repeat number from filename.
        Works for both barcode-style (4000000020_QuantReport_0001.csv)
        and descriptive GPC names (PS-M Blue_1.csv).
        """
        # Pattern 1: barcode-based (old style)
        m = re.match(r".*_(\d{10})(?:_QuantReport(?:_(\d{4}))?)?\.csv", filename)
        if m:
            barcode = m.group(1)
            repeat = int(m.group(2)) + 1 if m.group(2) else 1
            return barcode, repeat

        # Pattern 2: descriptive sample name + repeat (new GPC)
        m = re.match(r"(.+?)_(\d+)\.csv$", filename)
        if m:
            name = m.group(1).strip()
            repeat = int(m.group(2))
            return name, repeat

        # Fallback: just filename without extension
        name = os.path.splitext(filename)[0]
        return name, 1
    @staticmethod
    def _get_sample_type_from_quick(quick_file):
        """
        Try to detect 'Type: Sample' (or similar) in QuickReport CSV (if present).
        Many QuickReport exports won’t include it; we just return "" in that case.
        """
        try:
            # Best effort: scan first ~100 lines for a field like "Type:,Sample"
            with open(quick_file, encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i > 100:
                        break
                    if "Type:" in line:
                        parts = [p.strip() for p in line.split(",")]
                        for j, p in enumerate(parts):
                            if p.lower().startswith("type:"):
                                val = parts[j+1] if (j+1) < len(parts) else ""
                                return val.strip()
            return ""
        except Exception as e:
            print(f"Failed to parse sample type from {quick_file}: {e}")
            return ""

    @staticmethod
    def _extract_injection_timestamp_from_quick(quick_file):
        """
        Try to get an injection timestamp from QuickReport. If not present,
        attempt to parse a leading 'YYYYMMDD HHMMSS' from the filename.
        Returns string 'yyyymmdd HHMMSS' or None.
        """
        try:
            with open(quick_file, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if "Injection date:" in line:
                        # e.g. Injection date:,2025-06-22 12:45:50+02:00
                        parts = line.split(",")
                        for i, p in enumerate(parts):
                            if "Injection date:" in p:
                                inj_date = parts[i+1].strip()
                                dt_match = re.match(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})', inj_date)
                                if dt_match:
                                    return f"{dt_match.group(1)}{dt_match.group(2)}{dt_match.group(3)} {dt_match.group(4)}{dt_match.group(5)}{dt_match.group(6)}"
            # Fallback to filename: ^YYYYMMDD HHMMSS...
            fname = os.path.basename(quick_file)
            m = re.match(r'^(\d{8} \d{6})', fname)
            if m:
                return m.group(1)
            return None
        except Exception as e:
            print(f"Failed to get injection timestamp from {quick_file}: {e}")
            return None

    def parse_sample(self, barcode, repeat=1, quick_filename=None):
        import pandas as pd
        import re

        sample = {'barcode': barcode, 'repeat': repeat}

        # --- locate files ---
        quick_file  = self._find_file(self.quick_folder,  barcode, repeat=repeat)
        method_file = self._find_file(self.method_folder, barcode, repeat=repeat)
        qual_file   = self._find_file(self.qual_folder,   barcode, repeat=repeat)

        # --- method parsing (structured + normalized) ---
        try:
            from .method_parser_structured import parse_method_report_file
            method_struct = parse_method_report_file(method_file) if method_file else {}
        except Exception:
            method_struct = {}
        sample['method_struct'] = method_struct
        sample['method_version'] = (
            method_struct.get("8 Schema version", {}).get("Schema version")
            if isinstance(method_struct.get("8 Schema version"), dict) else None
        )
        # Back-compat alias so old code using sample['method'] continues to work
        sample['method'] = method_struct

        try:
            from .method_normalizer import normalize_method_info
            sample['method_norm'] = normalize_method_info(method_struct)
        except Exception:
            sample['method_norm'] = {}

        # --- QuickReport table (replaces quant_results) ---
        sample['quickreport'] = self._read_quick_report_csv(quick_file) if quick_file else pd.DataFrame()

        # --- Qual is OPTIONAL ---
        sample['qual_results'] = (
            self._extract_table_from_shimadzu_csv(qual_file, table_start_marker="Peak Results")
            if qual_file else pd.DataFrame()
        )

        # --- helper: DAD wavelength lookup from structured method ---
        def _dad_wavelength_from_method(method_struct, dad_channel: str):
            """dad_channel like '1A' -> look up wavelength for Signal A from '3.2 Signals' table."""
            if not dad_channel:
                return None
            ch_letter = dad_channel[-1]  # 'A' from '1A'
            # Prefer explicit "Signal table" under 3 DAD Method
            sec = method_struct.get("3 DAD Method", {})
            cand = None
            if isinstance(sec, dict):
                # Common key label from reports
                if isinstance(sec.get("Signal table"), pd.DataFrame):
                    cand = sec["Signal table"]
                else:
                    # fallback: find any DF with a 'Signal' column
                    for v in sec.values():
                        if isinstance(v, pd.DataFrame) and any("signal" in str(c).lower() for c in v.columns):
                            cand = v
                            break
            if isinstance(cand, pd.DataFrame) and not cand.empty:
                df = cand.copy()
                # normalize columns
                cols = {c: str(c).strip() for c in df.columns}
                df.rename(columns=cols, inplace=True)
                # find row for this channel (e.g., 'Signal A')
                if "Signal" in df.columns:
                    mask = df["Signal"].astype(str).str.contains(rf"\bSignal\s+{re.escape(ch_letter)}\b", case=False, regex=True)
                    hit = df[mask]
                    if not hit.empty:
                        # Wavelength column may be 'Wavelength' or 'Signal Wavelength'
                        wl_col = next((c for c in df.columns if "wavelength" in c.lower() and "ref" not in c.lower()), None)
                        if wl_col:
                            # extract numeric part (e.g., '254.0 nm' -> 254.0)
                            try:
                                return float(str(hit.iloc[0][wl_col]).split()[0])
                            except Exception:
                                return None
            return None

        # --- find chromatograms (RID + DAD) ---
        inj_ts = self._extract_injection_timestamp_from_quick(quick_file) if quick_file else None

        # RID
        rid_file = self._find_file(self.chrom_folder, barcode, repeat=repeat, inj_ts=inj_ts, contains='RID1A')
        if rid_file is None:
            print(f"WARNING: Could not find chromatogram for barcode {barcode} repeat {repeat} (expected inj time {inj_ts})")
        sample['chrom_rid'] = self._safe_read_csv(rid_file)

        # DAD channels
        dad_files = self._find_all_dad_files(self.chrom_folder, barcode, repeat=repeat, inj_ts=inj_ts)
        if not dad_files:
            print(f"WARNING: Could not find DAD chromatograms for barcode {barcode} repeat {repeat} (expected inj time {inj_ts})")

        if dad_files:
            dad_dfs = []
            for dad_file, dad_channel in dad_files:
                df = self._safe_read_csv(dad_file)
                wl = _dad_wavelength_from_method(sample['method_struct'], dad_channel)
                label = f"DAD_{dad_channel}_{int(wl)}" if wl is not None else f"DAD_{dad_channel}"
                if not df.empty and len(df.columns) >= 2:
                    df = df.rename(columns={df.columns[1]: label})
                    dad_dfs.append(df.set_index(df.columns[0]))
            if dad_dfs:
                dad_merged = pd.concat(dad_dfs, axis=1)
                dad_merged = dad_merged.loc[:, ~dad_merged.columns.duplicated()]
                sample['chrom_dad'] = dad_merged
            else:
                sample['chrom_dad'] = pd.DataFrame()
        else:
            sample['chrom_dad'] = pd.DataFrame()

        try:
            relabel_chromatograms(sample)
        except Exception as e:
            print(f"WARNING: relabel_chromatograms failed for {sample.get('barcode')}: {e}")

        return sample

    def _find_file(self, folder, barcode, repeat=1, inj_ts=None, contains=None, filename=None):
        """
        Finds the correct chromatogram file (RID or a specific channel) for a given barcode and repeat.
        - Works recursively through subfolders.
        - Strips trailing '_<repeat>' from barcode when matching folder/files.
        - If inj_ts provided, choose closest by timestamp prefix 'YYYYMMDD HHMMSS'.
        - Else, sort by run index in '<sample>_<NNN>.dx_' and pick Nth (repeat).
        """
        if filename:
            return filename if os.path.exists(filename) else None
        if not os.path.exists(folder):
            return None

        # Normalize: drop trailing _<digits> from barcode for matching
        base_name = re.sub(r'_\d+$', '', str(barcode)).strip()
        base_key = base_name.lower().replace(" ", "")

        def run_index(fname: str) -> int:
            # e.g. 20251007 105910_THF Blank_001.dx_RID1A.CSV -> 1
            m = re.search(r'_(\d{3})\.dx_', fname)
            return int(m.group(1)) if m else 999999

        # Collect candidates recursively
        candidates = []
        for root, _, files in os.walk(folder):
            for f in files:
                if not f.lower().endswith(".csv"):
                    continue
                if contains and contains.lower() not in f.lower():
                    continue
                key = f.lower().replace(" ", "")
                if base_key in key:
                    candidates.append(os.path.join(root, f))

        if not candidates:
            return None

        # If we have an injection timestamp, use closest match
        if inj_ts:
            inj_dt = pd.to_datetime(inj_ts, format='%Y%m%d %H%M%S', errors="coerce")
            best_file, min_diff = None, None
            for path in candidates:
                f = os.path.basename(path)
                m = re.match(r"^(\d{8} \d{6})", f)
                if not m:
                    continue
                f_dt = pd.to_datetime(m.group(1), format='%Y%m%d %H%M%S', errors="coerce")
                if pd.isna(inj_dt) or pd.isna(f_dt):
                    continue
                diff = abs((f_dt - inj_dt).total_seconds())
                if min_diff is None or diff < min_diff:
                    best_file, min_diff = path, diff
            if best_file:
                return best_file

        # Otherwise: sort by run index and pick Nth (repeat)
        candidates.sort(key=lambda p: (run_index(os.path.basename(p)), os.path.basename(p)))
        idx = max(0, int(repeat) - 1)
        if idx < len(candidates):
            return candidates[idx]
        return candidates[-1]  # fallback to last if repeat too large


    def _find_chrom_file(self, folder, barcode, contains=None, inj_ts=None):
        # unchanged utility (kept for compatibility; not used directly above)
        if not os.path.exists(folder):
            return None
        files = []
        ts_map = {}
        for f in os.listdir(folder):
            if contains and contains.lower() not in f.lower():
                continue
            if f'_{barcode}_' in f'_{f}_' and f.lower().endswith('.csv'):
                files.append(f)
                m = re.match(r'^(\d{8} \d{6})', f)
                if m:
                    ts_map[f] = m.group(1)
        if inj_ts and ts_map:
            inj_dt = pd.to_datetime(inj_ts, format='%Y%m%d %H%M%S')
            best_file = None
            min_diff = None
            for fname, fts in ts_map.items():
                try:
                    f_dt = pd.to_datetime(fts, format='%Y%m%d %H%M%S')
                    diff = abs((f_dt - inj_dt).total_seconds())
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        best_file = fname
                except Exception:
                    continue
            if best_file:
                return os.path.join(folder, best_file)
        if files:
            print(f"WARNING: Timestamp match failed for barcode {barcode}; using first matching chromatogram file.")
            return os.path.join(folder, files[0])
        return None

    def _safe_read_csv(self, filepath):
        if filepath and os.path.exists(filepath):
            try:
                return pd.read_csv(filepath, on_bad_lines='skip', engine='python')
            except Exception as e:
                print(f"Failed to read CSV: {filepath} ({e})")
        return pd.DataFrame()

    def _extract_barcode(self, filename):
        matches = re.findall(r'_(\d{10})_', f'_{filename}_')
        return matches[-1] if matches else None

    def _find_all_dad_files(self, folder, barcode, repeat=1, inj_ts=None):
        """
        Return one DAD file per channel (A..H) for the given barcode & repeat.
        - Recursive search.
        - Strip trailing '_<repeat>' from barcode for matching.
        - If inj_ts, pick closest by timestamp; else sort by run index '<NNN>.dx_' and take Nth.
        """
        base_name = re.sub(r'_\d+$', '', str(barcode)).strip()
        base_key = base_name.lower().replace(" ", "")

        def run_index(fname: str) -> int:
            m = re.search(r'_(\d{3})\.dx_', fname)
            return int(m.group(1)) if m else 999999

        files_by_ch = {}
        for ch in "ABCDEFGH":
            channel_key = f"DAD1{ch}"
            pool = []

            # collect candidates
            for root, _, files in os.walk(folder):
                for f in files:
                    if not f.lower().endswith(".csv"):
                        continue
                    if channel_key not in f:
                        continue
                    key = f.lower().replace(" ", "")
                    if base_key in key:
                        pool.append(os.path.join(root, f))

            if not pool:
                continue

            # choose by timestamp if given
            if inj_ts:
                inj_dt = pd.to_datetime(inj_ts, format='%Y%m%d %H%M%S', errors="coerce")
                best_file, min_diff = None, None
                for path in pool:
                    f = os.path.basename(path)
                    m = re.match(r"^(\d{8} \d{6})", f)
                    if not m:
                        continue
                    f_dt = pd.to_datetime(m.group(1), format='%Y%m%d %H%M%S', errors="coerce")
                    if pd.isna(inj_dt) or pd.isna(f_dt):
                        continue
                    diff = abs((f_dt - inj_dt).total_seconds())
                    if min_diff is None or diff < min_diff:
                        best_file, min_diff = path, diff
                if best_file:
                    files_by_ch[ch] = (best_file, f"1{ch}")
                    continue

            # else: sort by run index and pick Nth (repeat)
            pool.sort(key=lambda p: (run_index(os.path.basename(p)), os.path.basename(p)))
            idx = max(0, int(repeat) - 1)
            if idx < len(pool):
                files_by_ch[ch] = (pool[idx], f"1{ch}")
            else:
                files_by_ch[ch] = (pool[-1], f"1{ch}")  # fallback

        return list(files_by_ch.values())

    def _parse_method_report(self, filepath):
        if not filepath or not os.path.exists(filepath):
            return {}

        with open(filepath, encoding='utf-8') as f:
            lines = [line.rstrip("\n") for line in f]

        sections = {}
        section_number = None
        section_title = None
        buffer = []
        for line in lines + [""]:  # Ensure last section flush
            match = re.match(r'^(\d+(?:\.\d+)*),(.*)', line)
            if match:
                if section_number is not None:
                    sections[f"{section_number} {section_title.strip()}"] = buffer
                section_number, section_title = match.groups()
                buffer = [line]
            else:
                buffer.append(line)
        if section_number is not None:
            sections[f"{section_number} {section_title.strip()}"] = buffer

        method_info = {}
        for sec, lines in sections.items():
            data = []
            kv = {}
            sub_section = None
            sub_buffer = []
            for line in lines:
                m = re.match(r'^(\d+\.\d+),(.*)', line)
                if m:
                    if sub_section is not None and sub_buffer:
                        kv[sub_section] = self._parse_section_block(sub_buffer)
                    sub_section = m.group(1) + " " + m.group(2).strip()
                    sub_buffer = [line]
                elif sub_section is not None:
                    sub_buffer.append(line)
                else:
                    data.append(line)
            if sub_section is not None and sub_buffer:
                kv[sub_section] = self._parse_section_block(sub_buffer)
            if data:
                parsed = self._parse_section_block(data)
                if isinstance(parsed, dict):
                    kv.update(parsed)
                elif isinstance(parsed, pd.DataFrame):
                    kv = parsed
            method_info[sec] = kv
        return method_info

    @staticmethod
    def _parse_section_block(lines):
        lines = [l for l in lines if l.strip() != ""]
        if not lines:
            return {}
        # Remove section headers like "1,Method Information"
        if re.match(r'^\d+(?:\.\d+)*,', lines[0]):
            lines = lines[1:]
        # Search for a table: Find first line with >2 commas, treat as header
        table_start = None
        for i, l in enumerate(lines):
            if l.count(",") >= 2:  # crude table header detector
                table_start = i
                break
        if table_start is not None:
            # Scan until first empty line or non-table row
            table_lines = []
            for l in lines[table_start:]:
                if l.strip() == "":
                    break
                table_lines.append(l)
            try:
                df = pd.read_csv(io.StringIO("\n".join(table_lines)))
                # If just one table found, return as DataFrame (for signals)
                return df
            except Exception:
                pass
        # fallback: treat as key-value dict
        d = {}
        for l in lines:
            if "," in l:
                k, *v = l.split(",", 1)
                d[k.strip()] = v[0].strip() if v else ""
        return d

    def _get_dad_wavelength(self, method_dict, dad_channel):
        # unchanged helper
        if not method_dict:
            return None
        def find_wavelength(d):
            if isinstance(d, pd.DataFrame):
                df = d
                for col in df.columns:
                    for row in df[col].astype(str):
                        if dad_channel in row:
                            idx = df.index[df[col] == row]
                            if len(idx) > 0:
                                i = idx[0]
                                try:
                                    val = df.iloc[i, 1]
                                    return int(float(val))
                                except Exception:
                                    continue
            elif isinstance(d, dict):
                for v in d.values():
                    out = find_wavelength(v)
                    if out is not None:
                        return out
            return None
        return find_wavelength(method_dict)

    @staticmethod
    def _extract_table_from_shimadzu_csv(filepath, table_start_marker):
        if not filepath or not os.path.exists(filepath):
            return pd.DataFrame()
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
        marker_idx = None
        for i, line in enumerate(lines):
            if table_start_marker in line:
                marker_idx = i
                break
        if marker_idx is None:
            return pd.DataFrame()
        header_idx = None
        for j in range(marker_idx + 1, len(lines)):
            if lines[j].strip() != "":
                header_idx = j
                break
        if header_idx is None:
            return pd.DataFrame()
        for k in range(header_idx + 1, len(lines)):
            if lines[k].strip() == "":
                table_end = k
                break
        else:
            table_end = len(lines)
        table_data = "".join(lines[header_idx:table_end])
        try:
            df = pd.read_csv(io.StringIO(table_data))
        except Exception as e:
            print(f"Failed to parse table from {filepath}: {e}")
            return pd.DataFrame()
        return df

    @staticmethod
    def _read_quick_report_csv(filepath):
        """
        Normalize a QuickReport CSV into a tidy peak table:
            columns: Name (str), rt_min (float), area (float)
        Tolerant to headers like 'RT [min]', 'RT (min)', 'Area', etc.
        """
        if not filepath or not os.path.exists(filepath):
            return pd.DataFrame()
        try:
            df = pd.read_csv(filepath)
            # find columns
            def _find_col(df, *keywords):
                kws = [k.lower() for k in keywords]
                for c in df.columns:
                    if all(k in str(c).lower() for k in kws):
                        return c
                raise KeyError
            name_col = next((c for c in df.columns if str(c).strip().lower() == "name"), None)
            if name_col is None:
                name_col = _find_col(df, "name")
            try:
                rt_col = _find_col(df, "rt", "min")
            except Exception:
                rt_col = _find_col(df, "rt")
            area_col = _find_col(df, "area")

            out = df[[name_col, rt_col, area_col]].copy()
            out.columns = ["Name", "rt_min", "area"]
            out["rt_min"] = pd.to_numeric(out["rt_min"], errors="coerce")
            out["area"]   = pd.to_numeric(out["area"],   errors="coerce")
            out = out.dropna(subset=["Name"]).reset_index(drop=True)
            return out
        except Exception as e:
            print(f"Failed to parse QuickReport CSV {filepath}: {e}")
            return pd.DataFrame()
