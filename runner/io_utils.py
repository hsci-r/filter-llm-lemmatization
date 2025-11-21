# runner/io_utils.py

import os
import re
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

import csv

from collections import defaultdict


def read_corpus(corpus):
    songs = defaultdict(list)
    with open(corpus) as csvfile:
        reader = csv.reader(csvfile)
        for r in reader:
            if r[0] == 'poem_id':
                continue
            songs[r[0]].append(r[-1])
    print("%d songs" % len(songs))
    return songs




def slugify(s):
    """
    Turn a string into a filesystem-safe slug.
    Example: "GPT-4o Mini" -> "gpt-4o-mini"
    """
    return re.sub(r'[^a-z0-9]+', '-', str(s).lower()).strip('-')


def make_run_dir(runs_base, model, run_label, args_dict):
    
    """
    Create run directory with subfolders and args.json.

    Structure:
      runs/<model>__<promptset>__<timestamp>/
        input/
        json/
        reports/
        tables/
        logs/

    Returns: Path to run_dir
    """
    model = Path(model).name
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_base) / f"{run_label}" / f"{slugify(model)}_{ts}"
    (run_dir / "input").mkdir(parents=True, exist_ok=False)
    (run_dir / "json").mkdir(parents=True, exist_ok=False)
    (run_dir / "reports").mkdir(parents=True, exist_ok=False)
    (run_dir / "tables").mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)

    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

    return run_dir


def artifact_prefix(run_dir, poem_id, stage, chunk_no=None, attempt_no=None):
    """
    Build a consistent filename stem for artifacts.
    Example: "kalevipoeg_stage3_000_01"
    """
    parts = [slugify(poem_id), slugify(stage)]
    if chunk_no is not None:
        parts.append(f"{int(chunk_no):03d}")
    if attempt_no is not None:
        parts.append(f"{int(attempt_no):02d}")
    return "_".join(parts)


def format_messages_for_log(messages):
    """
    Pretty-print a chat message list for humans.

    Produces blocks like:
    ----- 1. SYSTEM -----
    ...
    ----- 2. USER -----
    ...

    Flattens list-style 'content' (with {"type":"text","text":...}) into plain text.
    """
    out = []
    for idx, m in enumerate(messages, 1):
        role = str(m.get("role", "")).upper()
        content = m.get("content", "")
        # flatten list content (OpenAI-style content blocks)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                else:
                    parts.append(str(part))
            content = "\n".join(parts)
        out.append("----- %d. %s -----" % (idx, role))
        out.append(str(content))
    return "\n\n".join(out) + "\n"



def save_text(run_dir, subdir, stem, suffix, text):
    """
    Save plain text to run_dir/<subdir>/<stem>_<suffix>.txt
    Returns: Path to saved file.
    """
    p = Path(run_dir) / subdir / f"{stem}_{suffix}.txt"
    p.write_text(text or "", encoding="utf-8")
    return p


def save_json(run_dir, stem, obj_json_str):
    """
    Save raw JSON string to run_dir/json/<stem>.json
    """
    p = Path(run_dir) / "json" / f"{stem}.json"
    with open(p, "w", encoding="utf-8") as f:
        f.write(obj_json_str)
    return p


def _strip_parens(text):
    return text


    #"""Remove all parenthetical groups like '(...)' from a string."""
    #s = text
    #while True:
    #    new_s = re.sub(r"\s*\([^()]*\)", "", s)
    #    if new_s == s:
    #        break
    #    s = new_s
    #return s.strip()


def save_table_csv(run_dir, poem_id, df):
    """
    Save per-poem table DataFrame to run_dir/tables/<poem>.csv
    Strips (...) parenthetical groups from data rows only (headers untouched).
    """
    out = Path(run_dir) / "tables" / f"{slugify(poem_id)}.csv"

    # copy df, clean only the cell values, not the column names
    clean_df = df.copy()
    for col in clean_df.columns:
        clean_df[col] = clean_df[col].map(_strip_parens)

    # group columns by a normalized header key
    groups = {}
    for col in list(clean_df.columns):
        key = re.sub(r"\s+", " ", str(col)).strip().lower()
        groups.setdefault(key, []).append(col)

    # collapse duplicate groups by taking first non-null per row (left to right)
    collapsed = pd.DataFrame(index=clean_df.index)
    try:
      for key, cols in groups.items():
          s = clean_df[cols[0]]
          for c in cols[1:]:
              s = s.combine_first(clean_df[c])
          collapsed[key] = s
    except Exception as e:
        print(e)
        print(clean_df)

    collapsed.to_csv(out, index=False)

    return out



def log_line(run_dir, text):
    """
    Append a log line to run_dir/logs/run.log
    """
    p = Path(run_dir) / "logs" / "run.log"
    with open(p, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    return p
