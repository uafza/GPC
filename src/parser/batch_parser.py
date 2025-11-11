import os
import re
import pandas as pd
import io


class HPLCBatchParser:
    def __init__(self, base_folder, metadata_parser=None):
        self.base_folder = base_folder
        self.method_folder = os.path.join(base_folder, 'AqMethod_Report')
        self.chrom_folder = os.path.join(base_folder, 'Chromatograms')
        self.qual_folder = os.path.join(base_folder, 'Qual_Results')
        self.quant_folder = os.path.join(base_folder, 'Quant_Results')
        self.samples = []
        self.metadata_parser = metadata_parser  # pass your MetadataParser instance

    def parse_batch(self):
        barcode_versions = dict()
        for f in os.listdir(self.quant_folder):
            if f.lower().endswith('.csv'):
                bc, repeat = self._extract_barcode_and_repeat(f)
                if not bc:
                    continue
                barcode_versions.setdefault(bc, []).append((f, repeat))
        self.samples = []
        for bc, files in sorted(barcode_versions.items()):
            files = sorted(files, key=lambda x: x[1])
            for (quant_file, repeat) in files:
                quant_path = os.path.join(self.quant_folder, quant_file)
                sample_type = self._get_sample_type_from_quant(quant_path)
                if sample_type.lower() != "sample":
                    continue
                sample = self.parse_sample(bc, repeat=repeat, quant_filename=quant_file)
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
        # e.g., 4000002131_QuantReport_0001.csv
        m = re.match(r'.*_(\d{10})(?:_QuantReport(?:_(\d{4}))?)?\.csv', filename)
        if m:
            barcode = m.group(1)
            repeat = int(m.group(2)) + 1 if m.group(2) else 1
            return barcode, repeat
        # fallback: old pattern
        m = re.search(r'_(\d{10})_', f'_{filename}_')
        return (m.group(1), 1) if m else (None, None)

    @staticmethod
    def _get_sample_type_from_quant(quant_file):
        try:
            with open(quant_file, encoding='utf-8') as f:
                for line in f:
                    if line.startswith("Acq. method:"):
                        parts = line.split(",")
                        for i, part in enumerate(parts):
                            if part.strip().startswith("Type:"):
                                value = parts[i+1].strip() if (i+1)<len(parts) else ""
                                return value
            return ""
        except Exception as e:
            print(f"Failed to parse sample type from {quant_file}: {e}")
            return ""

    @staticmethod
    def _extract_injection_timestamp_from_quant(quant_file):
        try:
            with open(quant_file, encoding='utf-8') as f:
                for line in f:
                    if "Data file:" in line:
                        # e.g. Data file:,20250622 124456_A046_TASK04_RUN01_01_4000002131_33.dx
                        parts = line.split(",")
                        for i, p in enumerate(parts):
                            if "Data file:" in p:
                                datafile = parts[i+1].strip()
                                ts = re.match(r'(\d{8} \d{6})', datafile)
                                if ts:
                                    return ts.group(1)
                    if "Injection date:" in line:
                        # e.g. Injection date:,2025-06-22 12:45:50+02:00
                        parts = line.split(",")
                        for i, p in enumerate(parts):
                            if "Injection date:" in p:
                                inj_date = parts[i+1].strip()
                                # Optionally convert to yyyymmdd HHMMSS
                                dt_match = re.match(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})', inj_date)
                                if dt_match:
                                    return f"{dt_match.group(1)}{dt_match.group(2)}{dt_match.group(3)} {dt_match.group(4)}{dt_match.group(5)}{dt_match.group(6)}"
            return None
        except Exception as e:
            print(f"Failed to get injection timestamp from {quant_file}: {e}")
            return None

    def parse_sample(self, barcode, repeat=1, quant_filename=None):
        sample = {'barcode': barcode, 'repeat': repeat}
        # Method/qual: try to match _000X for repeats
        suffix = f"_{str(repeat-1).zfill(4)}" if repeat > 1 else ""
        quant_file = self._find_file(self.quant_folder, barcode, repeat=repeat)
        method_file = self._find_file(self.method_folder, barcode, repeat=repeat)
        qual_file = self._find_file(self.qual_folder, barcode, repeat=repeat)

        sample['method'] = self._parse_method_report(method_file) if method_file else {}
        sample['quant_results'] = self._extract_table_from_shimadzu_csv(quant_file, table_start_marker="Signal:") if quant_file else pd.DataFrame()
        sample['qual_results'] = self._extract_table_from_shimadzu_csv(qual_file, table_start_marker="Peak Results") if qual_file else pd.DataFrame()

        # Find correct chromatogram for this injection
        inj_ts = self._extract_injection_timestamp_from_quant(quant_file) if quant_file else None
        rid_file = self._find_file(self.chrom_folder, barcode, repeat=repeat, inj_ts=inj_ts, contains='RID1A')
        if rid_file is None:
            print(f"WARNING: Could not find chromatogram for barcode {barcode} repeat {repeat} (expected inj time {inj_ts})")
        sample['chrom_rid'] = self._safe_read_csv(rid_file)

        dad_files = self._find_all_dad_files(self.chrom_folder, barcode, repeat=repeat, inj_ts=inj_ts)
        if not dad_files:
            print(f"WARNING: Could not find DAD chromatograms for barcode {barcode} repeat {repeat} (expected inj time {inj_ts})")
        if dad_files:
            dad_dfs = []
            for dad_file, dad_channel in dad_files:
                df = self._safe_read_csv(dad_file)
                wl = self._get_dad_wavelength(sample['method'], dad_channel)
                label = f'DAD_{dad_channel}_{wl}' if wl else f'DAD_{dad_channel}'
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
        return sample

    def _find_file(self, folder, barcode, repeat=1, inj_ts=None, contains=None, filename=None):
        """
        Finds the correct file for a given barcode and repeat index.
        If repeat > 1, match files ending with _000X (e.g., _0001 for repeat=2).
        If inj_ts is provided, try to match timestamp in filename.
        If filename is provided, return it directly.
        """
        if filename:
            return filename if os.path.exists(filename) else None
        if not os.path.exists(folder):
            return None
        suffix = f"_{str(repeat-1).zfill(4)}" if repeat > 1 else ""
        candidates = []
        for f in os.listdir(folder):
            fname_lower = f.lower()
            # Adjust barcode match for repeat
            if (f'_{barcode}{suffix}_' in f'_{f}_') and (contains is None or contains.lower() in fname_lower) and fname_lower.endswith('.csv'):
                candidates.append(f)
        if inj_ts and candidates:
            # Try to match by timestamp in filename (if present)
            inj_dt = pd.to_datetime(inj_ts, format='%Y%m%d %H%M%S')
            best_file = None
            min_diff = None
            for f in candidates:
                m = re.match(r'^(\d{8} \d{6})', f)
                if m:
                    try:
                        f_dt = pd.to_datetime(m.group(1), format='%Y%m%d %H%M%S')
                        diff = abs((f_dt - inj_dt).total_seconds())
                        if min_diff is None or diff < min_diff:
                            min_diff = diff
                            best_file = f
                    except Exception:
                        continue
            if best_file:
                return os.path.join(folder, best_file)
        if candidates:
            return os.path.join(folder, candidates[0])
        return None


    def _find_chrom_file(self, folder, barcode, contains=None, inj_ts=None):
        # Try to match by timestamp; fallback: first file with barcode
        if not os.path.exists(folder):
            return None
        files = []
        ts_map = {}
        for f in os.listdir(folder):
            if contains and contains.lower() not in f.lower():
                continue
            if f'_{barcode}_' in f'_{f}_' and f.lower().endswith('.csv'):
                files.append(f)
                # Try to extract timestamp: e.g. 20250622 124456
                m = re.match(r'^(\d{8} \d{6})', f)
                if m:
                    ts_map[f] = m.group(1)
        if inj_ts and ts_map:
            # Find file with closest timestamp
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
        # fallback: just pick first
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
        # List all DAD files for given barcode and timestamp (closest for each channel)
        files_by_ch = {}
        for ch in 'ABCDEFGH':
            best_file = None
            min_diff = None
            best_ts = None
            for f in os.listdir(folder):
                if f'DAD1{ch}' not in f:
                    continue
                if f'_{barcode}_' not in f'_{f}_' or not f.lower().endswith('.csv'):
                    continue
                m = re.match(r'^(\d{8} \d{6})', f)
                if inj_ts and m:
                    try:
                        f_dt = pd.to_datetime(m.group(1), format='%Y%m%d %H%M%S')
                        inj_dt = pd.to_datetime(inj_ts, format='%Y%m%d %H%M%S')
                        diff = abs((f_dt - inj_dt).total_seconds())
                        if min_diff is None or diff < min_diff:
                            min_diff = diff
                            best_file = f
                            best_ts = m.group(1)
                    except Exception:
                        continue
                elif not inj_ts and not best_file:
                    best_file = f
            if best_file:
                files_by_ch[ch] = (os.path.join(folder, best_file), f'1{ch}')
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
        # Search all values in the sectioned dict for tables with DAD info
        if not method_dict:
            return None
        # Walk the section dict tree for any DataFrame with wavelength
        def find_wavelength(d):
            if isinstance(d, pd.DataFrame):
                df = d
                for col in df.columns:
                    for row in df[col].astype(str):
                        if dad_channel in row:
                            # Try to find number in next col or after channel
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
