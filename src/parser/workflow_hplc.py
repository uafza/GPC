# workflow_hplc.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from parser.batch_parser import HPLCBatchParser

def run_hplc_workflow(base_folder=".", out_dir="results_hplc"):
    """
    Minimal HPLC path:
      - parse batch with HPLCBatchParser
      - for each sample, export a tidy QuickReport peak table:
          columns: Sample, Name, RT (min), Area
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    parser = HPLCBatchParser(base_folder=base_folder)
    samples = parser.parse_batch()

    wrote = 0
    for s in samples:
        name = f"{s.get('barcode','')}_r{s.get('repeat',1)}"
        qdf = s.get("quickreport")
        if qdf is None or qdf.empty:
            print(f"[skip] {name}: no QuickReport")
            continue
        df = qdf.copy()
        df.insert(0, "Sample", name)
        df.columns = ["Sample", "Name", "RT (min)", "Area"]
        out_csv = out / f"{name}_PEAKS.csv"
        df.to_csv(out_csv, index=False)
        wrote += 1
    print(f"Wrote {wrote} HPLC peak tables to {out.resolve()}")
