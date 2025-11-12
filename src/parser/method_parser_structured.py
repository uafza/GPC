from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import io
import pandas as pd


_SECTION_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)*),(?P<title>.*)$")


@dataclass
class Node:
    num: str
    title: str
    kv: Dict[str, str] = field(default_factory=dict)          # flat key/value pairs in this section
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)  # any tables found in this section
    children: Dict[str, "Node"] = field(default_factory=dict) # nested subsections by number

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.kv:
            out.update(self.kv)
        if self.tables:
            out.update({k: v for k, v in self.tables.items()})
        for k in sorted(self.children, key=_num_sort_key):
            child = self.children[k]
            out[f"{child.num} {child.title}"] = child.to_dict()
        return out


def _num_sort_key(s: str) -> Tuple:
    return tuple(int(x) for x in s.split("."))


def _parse_kv_line_to_pairs(line: str) -> List[Tuple[str, str]]:
    parts = [p.strip() for p in line.split(",")]
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(parts):
        if ":" in parts[i]:
            k, after = parts[i].split(":", 1)
            k = k.strip()
            if after.strip():
                v = after.strip()
                i += 1
            else:
                v = parts[i+1].strip() if i + 1 < len(parts) else ""
                i += 2
            pairs.append((k, v))
        else:
            if pairs and not pairs[-1][1]:
                pairs[-1] = (pairs[-1][0], parts[i])
            i += 1
    return pairs


def _collect_table(lines: List[str], start_idx: int) -> Tuple[pd.DataFrame, int, Optional[str]]:
    """
    Collect a table starting at start_idx. Returns (df, next_index, optional_name).
    Heuristics:
      - If the line contains 'table' (e.g., 'Signal table'), treat it as a table name,
        and expect the next non-empty line to be the header.
      - Else, if the current line looks like a CSV header (>=1 comma), use it as header.
      - Continue until blank line or next section header.
    """
    i = start_idx
    table_name: Optional[str] = None

    # Optional label like "Signal table"
    label_match = re.search(r"table", lines[i], flags=re.IGNORECASE)
    if label_match:
        table_name = lines[i].strip()
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1

    if i >= len(lines):
        return pd.DataFrame(), i, table_name

    header_line = lines[i]
    if header_line.count(",") < 1:
        return pd.DataFrame(), start_idx, None

    table_buf = [header_line]
    i += 1

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            break
        if _SECTION_RE.match(line):
            break
        table_buf.append(line)
        i += 1

    try:
        df = pd.read_csv(io.StringIO("\n".join(table_buf)))
        return df, i, table_name
    except Exception:
        return pd.DataFrame(), start_idx, None


def parse_method_report_text(text: str) -> Dict[str, Any]:
    """
    Parse the raw method report text into a nested dictionary by numbered sections.
    Top-level preamble KV ("Acquisition Method:", "Path:") is folded into '1 Method Information'.
    Each section collects:
      - kv: merged key/value pairs (comma-separated 'Key:,Value' tokens)
      - tables: DataFrames under sensible keys (e.g., 'Signal table', 'Solvent Composition')
      - children: subsections by numbering (e.g., 2.1, 2.1.1)
    """
    lines = [l.rstrip("\n\r") for l in text.splitlines()]

    # ---- preamble KV until first numbered section ----
    preamble_kv: Dict[str, str] = {}
    idx = 0
    while idx < len(lines) and not _SECTION_RE.match(lines[idx]):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if ":" in line:
            for k, v in _parse_kv_line_to_pairs(line):
                if k:
                    preamble_kv[k] = v
        idx += 1

    # ---- section nodes by number ----
    roots: Dict[str, Node] = {}
    stack: List[Node] = []  # hierarchy stack

    def place_node(node: Node):
        while stack and not node.num.startswith(stack[-1].num + "."):
            stack.pop()
        if stack:
            stack[-1].children[node.num] = node
        else:
            roots[node.num] = node
        stack.append(node)

    # scan sections
    while idx < len(lines):
        header_m = _SECTION_RE.match(lines[idx])
        if not header_m:
            idx += 1
            continue

        sec_num = header_m.group("num")
        sec_title = header_m.group("title").strip()
        node = Node(num=sec_num, title=sec_title)
        place_node(node)
        idx += 1

        # collect content lines until next section
        content_start = idx
        while idx < len(lines) and not _SECTION_RE.match(lines[idx]):
            idx += 1
        content = lines[content_start:idx]

        # parse content: mix of KV lines and optional tables
        j = 0
        while j < len(content):
            line = content[j].strip()
            if not line:
                j += 1
                continue

            lower_line = line.lower()

            # explicit table labels
            if lower_line in {"signal table", "solvent composition"}:
                df, j_next, _ = _collect_table(content, j)
                if not df.empty:
                    node.tables[line.strip()] = df  # exact key: "Signal table" or "Solvent Composition"
                    j = j_next
                    continue

            # generic table header detection
            looks_like_table_header = ("," in line) and (":" not in line) and (line.count(",") >= 1)
            has_table_word = "table" in lower_line
            if has_table_word or looks_like_table_header:
                df, j_next, tname = _collect_table(content, j)
                if not df.empty:
                    key = (tname or sec_title).strip()
                    node.tables[key] = df
                    j = j_next
                    continue

            # KV line
            if ":" in line:
                for k, v in _parse_kv_line_to_pairs(line):
                    if k:
                        node.kv[k] = v
            j += 1

    # fold preamble into "1 Method Information"
    if preamble_kv:
        one = roots.get("1")
        if one is None:
            one = Node(num="1", title="Method Information")
            roots["1"] = one
        for k, v in preamble_kv.items():
            one.kv.setdefault(k, v)

    out: Dict[str, Any] = {}
    for k in sorted(roots, key=_num_sort_key):
        node = roots[k]
        out[f"{node.num} {node.title}"] = node.to_dict()
    return out


def parse_method_report_file(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return parse_method_report_text(text)
