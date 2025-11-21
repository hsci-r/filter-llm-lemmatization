# evaluation.py

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from Levenshtein import ratio, distance

warnings.filterwarnings("ignore")

# -----------------------
# CLI
# -----------------------




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-a", "--annotation",
        type=str, required=True,
        help="CSV file with manual annotations",
    )
    p.add_argument(
        "-o", "--output",
        type=str, required=True,
        help="Folder with model output; multiple folders separated by ','.",
    )
    p.add_argument(
        "-u", "--use",
        choices=["exact", "distance"], default="exact",
        help="Which measures to use for best/worst comparison.",
    )
    p.add_argument(
        "-r", "--result",
        type=str, default="eval.csv",
        help="Path to write the main .csv result.",
    )
    p.add_argument(
        "--spacy-model",
        type=str, default="en_core_web_lg",
        help="spaCy model for semantic similarity.",
    )
    p.add_argument(
    "--outdir",
    type=str,
    default=None,
    help="Directory to store all outputs. If not set: "
         "with one -o folder → <basename>_eval; otherwise → stem of -r.",
    )

    p.add_argument(
    "--precision",
    type=int,
    default=3,
    help="Number of decimal places to show in outputs (default: 3).",
    )   

    return p.parse_args()


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column (case-insensitive), or None."""
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def _run_id_from_path(path: str) -> str:
    """Infer a run id from a table file path."""
    d = os.path.dirname(path)
    base = os.path.basename(d)
    if base == "tables":  # climb one level up if needed
        d = os.path.dirname(d)
        base = os.path.basename(d)
    return base or "run"


# -----------------------
# IO helpers
# -----------------------




def float_fmt(n: int) -> str:
    return f"%.{n}f"


def resolve_outdir(args, model_folders: list[str]) -> str:
    """Choose an output directory based on --outdir, -o, and -r."""
    if args.outdir:  # explicit wins
        outdir = args.outdir
    else:
        if len(model_folders) == 1 and model_folders[0]:
            base = os.path.basename(os.path.normpath(model_folders[0]))
            outdir = f"{base}_eval"
        else:
            # fall back to stem of -r
            stem = os.path.splitext(os.path.basename(args.result))[0]
            outdir = stem or "eval_results"
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _rename_if_present(df, mapping):
    """Rename columns only if the source key exists (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    actual = {}
    for src_lower, dst in mapping.items():
        if src_lower in lower_map:
            actual[lower_map[src_lower]] = dst
    return df.rename(columns=actual)

def read_ground_truth(path: str) -> pd.DataFrame:
    """Read and normalize ground-truth annotations.
    'root' is optional; if absent, metrics that use it will just be 0.0.
    """
    gt = pd.read_csv(path)

    # Normalize common header variants (case-insensitive keys on the left)
    gt = _rename_if_present(gt, {
        # canonical
        "poem_id": "poem_id",
        "word": "word",
        "word_normalised": "word_normalised",
        "word normalised": "word_normalised",
        "word_normalized": "word_normalised",
        "word normalized": "word_normalised",
        "word_lemmatised (local)": "word_lemmatised (local)",
        "word lemmatised (local)": "word_lemmatised (local)",
        "lemma_standard": "lemma_standard",
        "lemma standard": "lemma_standard",
        "lemma_standard (finnish/estonian & minority languages if needed)": "lemma_standard",
        "english": "English",

        # 'root' comes in many flavors; map any of them to 'root'
        "root": "root",
        "root (est/fi)": "root",
        "root (est/fi": "root",                 # seen typo without closing paren
        "etymological root": "root",
        "etymology root": "root",
    })

    # Keep only rows with English filled (if present)
    if "English" in gt.columns:
        gt = gt[gt["English"].notna()]

    # Required (minimal) columns; 'root' is OPTIONAL
    required = [
        "poem_id",
        "word",
        "word_normalised",
        "word_lemmatised (local)",
        "lemma_standard",
        "English",
    ]
    missing = [c for c in required if c not in gt.columns]
    if missing:
        raise ValueError(f"Ground truth missing columns: {missing}\nAvailable: {list(gt.columns)}")

    # If 'root' is missing, create an empty placeholder so downstream code never KeyErrors
    if "root" not in gt.columns:
        gt["root"] = ""

    return gt

def read_model_output(path: str) -> pd.DataFrame | None:
    """Read one model output CSV and normalize column names."""
    try:
        df = pd.read_csv(path, keep_default_na=False, na_values=[])
    except pd.errors.EmptyDataError:
        print(f"EmptyData: {path}")
        return None

    df = df.fillna("")
    # normalize columns: strip asterisks, lower, map common variants
    df.columns = df.columns.str.replace("*", "", regex=False).str.lower()

    rename_map = {
        "lemma (modern finnish)": "lemma modern",
        "lemma (modern estonian)": "lemma modern",
        "lemma (modern)": "lemma modern",
        "lemma (mod.)": "lemma modern",
        "lemma (modern swedish)": "lemma modern",
        "normalized": "normalized orthography",
        "normalised": "normalized orthography",
        "lemma (orig.)": "lemma (original)",
        "english": "english translation",
        "english lemma": "english translation",
    }
    # handle expected canonical column casing later; keep lowercase here
    df = df.rename(columns=rename_map)
    return df

def read_dir(dirpath: str, song_ids: set[str]) -> dict[str, str]:
    """List CSVs in a directory that match known poem IDs."""
    if not os.path.isdir(dirpath):
        return {}
    out = {}
    for f in os.listdir(dirpath):
        if f.endswith(".csv"):
            s = f[:-4]
            if s in song_ids:
                out[s] = os.path.join(dirpath, f)
    return out

# -----------------------
# String/metric helpers
# -----------------------

def exact(df: pd.DataFrame, c1: str, c2: str) -> float:
    try:
        return float((df[c1] == df[c2]).mean())
    except KeyError:
        return 0.0

def lev_distance(df: pd.DataFrame, c1: str, c2: str) -> float:
    # average Levenshtein ratio (higher is better)
    try:
        return float(df.apply(lambda x: ratio(str(x[c1]), str(x[c2])), axis=1).mean())
    except KeyError:
        return 0.0

def lev_thr(df: pd.DataFrame, c1: str, c2: str, thr: int = 1) -> float:
    try:
        vals = df.apply(lambda x: distance(str(x[c1]), str(x[c2])) <= thr, axis=1)
        return float(vals.mean())
    except KeyError:
        return 0.0

def compare_strings(df: pd.DataFrame, c1: str, c2: str) -> tuple[float, float, float]:
    try:
        return exact(df, c1, c2), lev_distance(df, c1, c2), lev_thr(df, c1, c2)
    except KeyError as e:
        print("KeyError in compare_strings:", e, "available:", list(df.columns))
        return 0.0, 0.0, 0.0

# -----------------------
# Semantics with spaCy
# -----------------------

_nlp = None
_STOP = None

def ensure_nlp(model_name: str):
    global _nlp, _STOP
    if _nlp is None:
        import spacy
        _nlp = spacy.load(model_name)
        _STOP = {s.lower() for s in _nlp.Defaults.stop_words}

def similarity(t1: str, t2: str) -> float:
    """Cosine similarity of mean token vectors excluding stopwords.
    Safeguards against empty vectors.
    """
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0

    doc1 = _nlp(t1)
    doc2 = _nlp(t2)

    v1 = np.zeros(doc1.vocab.vectors_length or 300, dtype="float32")
    v2 = np.zeros(doc2.vocab.vectors_length or 300, dtype="float32")

    n1 = 0
    for tok in doc1:
        if tok.text.lower() not in _STOP and tok.has_vector:
            v1 += tok.vector
            n1 += 1
    n2 = 0
    for tok in doc2:
        if tok.text.lower() not in _STOP and tok.has_vector:
            v2 += tok.vector
            n2 += 1

    if n1 > 0:
        v1 /= n1
    if n2 > 0:
        v2 /= n2

    nrm1 = np.linalg.norm(v1)
    nrm2 = np.linalg.norm(v2)
    if nrm1 == 0.0 or nrm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (nrm1 * nrm2))

def compare_semantics(df: pd.DataFrame, c1: str, c2: str) -> float:
    try:
        return float(df.apply(lambda x: similarity(str(x[c1]), str(x[c2])), axis=1).mean())
    except KeyError:
        print("compare_semantics: missing", c1, c2, "in", list(df.columns))
        return 0.0

def compare_translations(df: pd.DataFrame, c1: str, c2: str) -> tuple[float, float]:
    try:
        return exact(df, c1, c2), compare_semantics(df, c1, c2)
    except KeyError:
        print("COMPARE_TRANSLATIONS", list(df.columns), c1, c2)
        return 0.0, 0.0

# -----------------------
# Poem-level evaluation
# -----------------------

def align_on_original(gt_poem: pd.DataFrame, model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """Align rows by comparing GT 'word' with model 'original form' (case-insensitive, '*' stripped).
    If partial prefix matches, truncate both to the longest matching prefix.
    """
    # Allow common model col variants (we normalized to lowercase earlier)
    candidates = [
        "original form", "original_form", "original", "orig", "orig form",
        "original word", "originalword"
    ]
    model_cols = {c.lower(): c for c in model_df.columns}
    orig_col = next((model_cols[c] for c in model_cols if c in candidates), None)
    if orig_col is None:
        print("model_output.columns:", list(model_df.columns))
        return None, None

    gt_list = gt_poem["word"].astype(str).tolist()
    model_list = (
        pd.Series(model_df[orig_col]).astype(str).str.replace("*", "", regex=False).tolist()
    )

    m = min(len(gt_list), len(model_list))
    i = 0
    for i in range(m):
        try:
            if gt_list[i].lower() != model_list[i].lower():
                break
        except Exception as e:
            print("Alignment error at", i, gt_list[i], e)
            break
    # if all matched, i == m-1 at loop end; adjust
    if i == m - 1 and m > 0 and gt_list[i].lower() == model_list[i].lower():
        i = m

    if i == 0:
        print("Found 0 aligned tokens; skipping.")
        return None, None

    return gt_poem.iloc[:i].reset_index(drop=True), model_df.iloc[:i].reset_index(drop=True)

def evaluate_poem(gt_poem: pd.DataFrame, model_output: pd.DataFrame) -> list | None:
    gt_aligned, model_aligned = align_on_original(gt_poem, model_output)
    if gt_aligned is None:
        return None


    # ----- build full-length joint (GT length), padding model rows with "" -----
    true_len = len(gt_poem)
    aligned_len = len(model_aligned)
    if true_len == 0:
        return None

    # use the full GT, and pad the model side if it is shorter
    gt_full = gt_poem.reset_index(drop=True)
    model_full = model_aligned.reset_index(drop=True)
    if aligned_len < true_len:
        pad_rows = true_len - aligned_len
        pad_df = pd.DataFrame([{c: "" for c in model_full.columns}] * pad_rows)
        model_full = pd.concat([model_full, pad_df], ignore_index=True)

    poem_joint = pd.concat([gt_full, model_full], axis=1)
    length = true_len

    # Expectation: GT uses canonical column names (see read_ground_truth)
    # Model expectation: normalized columns (lowercase in read_model_output)
    # For model-side access, use the exact strings below (lowercase).
    norm_exact, norm_distance, norm_lev1 = compare_strings(
        poem_joint, "word_normalised", "normalized orthography"
    )
    local_exact, local_distance, local_lev1 = compare_strings(
        poem_joint, "word_lemmatised (local)", "lemma (original)"
    )
    standard_exact, standard_distance, standard_lev1 = compare_strings(
        poem_joint, "lemma_standard", "lemma modern"
    )
    root_exact, root_distance, root_lev1 = compare_strings(
        poem_joint, "root", "etymological root"
    )
    translation_exact, translation_semantic = compare_translations(
        poem_joint, "English", "english translation"
    )

    return [
        length,
        norm_exact, norm_distance, norm_lev1,
        local_exact, local_distance, local_lev1,
        standard_exact, standard_distance, standard_lev1,
        root_exact, root_distance, root_lev1,
        translation_exact, translation_semantic,
    ]


def evaluate_poem_perword(
    gt_poem: pd.DataFrame,
    model_df: pd.DataFrame,
    poem_id: str,
    run_id: str,
) -> list[dict] | None:
    """
    Build per-token rows with distances and model outputs.
    Returns a list of dicts or None if alignment fails.
    """
    gt_aligned, model_aligned = align_on_original(gt_poem, model_df)
    if gt_aligned is None:
        return None

    # Use full GT length, pad model if needed (same as evaluate_poem)
    true_len = len(gt_poem)
    model_full = model_aligned.reset_index(drop=True)
    if len(model_full) < true_len:
        pad_df = pd.DataFrame([{c: "" for c in model_full.columns}] * (true_len - len(model_full)))
        model_full = pd.concat([model_full, pad_df], ignore_index=True)
    gt_full = gt_poem.reset_index(drop=True)
    joint = pd.concat([gt_full, model_full], axis=1)

    # Identify GT ids (optional; synthesize if missing)
    verse_col = _pick_col(gt_full, ["verse_id", "verse", "line_id", "line"])
    word_col  = _pick_col(gt_full, ["word_id", "token_id", "position", "index"])

    verse_ids = joint[verse_col].tolist() if verse_col else ["" for _ in range(len(joint))]
    if word_col:
        word_ids = joint[word_col].tolist()
    else:
        # 1-based token index within poem if not provided
        word_ids = list(range(1, len(joint) + 1))

    # Model-side canonical columns
    norm_col     = _pick_col(joint, ["normalized orthography"])
    local_col    = _pick_col(joint, ["lemma (original)"])
    standard_col = _pick_col(joint, ["lemma modern"])
    root_col     = _pick_col(joint, ["etymological root"])
    trans_col    = _pick_col(joint, ["english translation"])

    # GT-side canonical columns
    gt_norm     = "word_normalised"
    gt_local    = "word_lemmatised (local)"
    gt_standard = "lemma_standard"
    gt_root     = "root"
    gt_english  = "English"

    rows = []
    for i, row in joint.iterrows():
        g_norm     = str(row.get(gt_norm, ""))
        g_local    = str(row.get(gt_local, ""))
        g_standard = str(row.get(gt_standard, ""))
        g_root     = str(row.get(gt_root, ""))
        g_english  = str(row.get(gt_english, ""))

        m_norm     = str(row.get(norm_col, ""))     if norm_col     else ""
        m_local    = str(row.get(local_col, ""))    if local_col    else ""
        m_standard = str(row.get(standard_col, "")) if standard_col else ""
        m_root     = str(row.get(root_col, ""))     if root_col     else ""
        m_trans    = str(row.get(trans_col, ""))    if trans_col    else ""

        # distances per token
        nd = ratio(g_norm,     m_norm)     if (g_norm or m_norm)     else 0.0
        ld = ratio(g_local,    m_local)    if (g_local or m_local)   else 0.0
        sd = ratio(g_standard, m_standard) if (g_standard or m_standard) else 0.0
        rd = ratio(g_root,     m_root)     if (g_root or m_root)     else 0.0
        ts = similarity(g_english, m_trans)  # protected against empties inside similarity()

        rows.append({
            "run_id": run_id,
            "poem_id": poem_id,
            "verse_id": verse_ids[i],
            "word_id": word_ids[i],
            "llm_norm": m_norm,
            "norm_distance": float(nd),
            "llm_local_lemma": m_local,
            "local_distance": float(ld),
            "llm_standard": m_standard,
            "standard_distance": float(sd),
            "llm_root": m_root,
            "root_distance": float(rd),
            "translation_semantic": float(ts),
        })
    return rows



# -----------------------
# Aggregation
# -----------------------

def best_worst(model_res: pd.DataFrame, use: str) -> pd.DataFrame:
    if use == "distance":
        cols = ["norm_distance", "local_distance", "standard_distance", "root_distance", "translation_semantic"]
    else:  # "exact"
        cols = ["norm_exact", "local_exact", "standard_exact", "root_exact", "translation_semantic"]

    rows = []
    for c in cols:
        # idxmin/idxmax on a Series → index label
        worst_idx = model_res[c].idxmin()
        best_idx = model_res[c].idxmax()
        rows.append([
            c,
            float(model_res[c].mean()),
            model_res.loc[worst_idx, "song_id"],
            float(model_res.loc[worst_idx, c]),
            model_res.loc[best_idx, "song_id"],
            float(model_res.loc[best_idx, c]),
        ])

    combined = model_res[cols].mean(axis=1)
    comb_min = combined.idxmin()
    comb_max = combined.idxmax()
    rows.append([
        "combined",
        float(combined.mean()),
        model_res.loc[comb_min, "song_id"],
        float(combined.loc[comb_min]),
        model_res.loc[comb_max, "song_id"],
        float(combined.loc[comb_max]),
    ])

    return pd.DataFrame(rows, columns=["field", "avg", "worst_id", "worst_val", "best_id", "best_val"])


# -----------------------
# Error Analysis
# -----------------------

# -----------------------
# Worst-lists + mistakes
# -----------------------

def write_top10_worst_lists(result: pd.DataFrame, base_out: str, float_format: str) -> None:
    worst_std = result.nsmallest(10, "standard_distance")[["song_id", "standard_distance"]]
    worst_std.to_csv(base_out.replace(".csv", "_worst10_standard.csv"), index=False, float_format=float_format)

    worst_tr = result.nsmallest(10, "translation_semantic")[["song_id", "translation_semantic"]]
    worst_tr.to_csv(base_out.replace(".csv", "_worst10_translation.csv"), index=False, float_format=float_format)

    worst_loc = result.nsmallest(10, "local_distance")[["song_id", "local_distance"]]
    worst_loc.to_csv(base_out.replace(".csv", "_worst10_original.csv"), index=False, float_format=float_format)

def collect_standard_lemma_errors(
    gt_poem: pd.DataFrame,
    model_df: pd.DataFrame,
    song_id: str,
) -> list[dict]:
    """
    Return per-token errors for standard lemma:
    GT:  'lemma_standard'
    Model: 'lemma modern'
    """
    gt_aligned, model_aligned = align_on_original(gt_poem, model_df)
    if gt_aligned is None:
        return []

    joint = pd.concat([gt_aligned, model_aligned], axis=1)

    # Guard columns quietly; if missing, no errors to record.
    need = {"word", "lemma_standard", "lemma modern"}
    if not need.issubset(set(joint.columns)):
        return []

    errs = []
    for _, row in joint.iterrows():
        inp = str(row["word"])
        gold = str(row["lemma_standard"])
        pred = str(row["lemma modern"])
        # record only mismatches (case-sensitive, like your exact())
        if pred != gold:
            errs.append({
                "input word": inp,
                "correct lemma": gold,
                "incorrect model output": pred,
                "song_id": song_id,
            })
    return errs


def write_standard_lemma_mistake_dictionary(errors: list[dict], base_out: str, float_format: str) -> None:
    if not errors:
        empty = pd.DataFrame(
            columns=["input word", "correct lemma", "incorrect model output", "list of document ids", "count"]
        )
        empty.to_csv(base_out.replace(".csv", "_standard_lemma_mistakes.csv"), index=False, float_format=float_format)
        return

    df = pd.DataFrame(errors)
    grouped = (
        df.groupby(["input word", "correct lemma", "incorrect model output"])
          .agg(
              count=("song_id", "size"),
              doc_ids=("song_id", lambda x: sorted(set(map(str, x))))
          )
          .reset_index()
    )
    grouped = grouped.sort_values("count", ascending=False)
    grouped["list of document ids"] = grouped["doc_ids"].apply(lambda xs: ",".join(xs))
    grouped = grouped.drop(columns=["doc_ids"])
    grouped = grouped[["input word", "correct lemma", "incorrect model output", "list of document ids", "count"]]
    grouped.to_csv(base_out.replace(".csv", "_standard_lemma_mistakes.csv"), index=False, float_format=float_format)



def find_all_csvs(root_dir: str, song_ids: set[str]) -> dict[str, str]:
    """Recursively find all CSV files below root_dir whose basenames match known poem IDs."""
    out = {}
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".csv"):
                stem = f[:-4]
                if stem in song_ids:
                    out[stem] = os.path.join(dirpath, f)
    return out


def summarize_matches_per_folder(out_paths: dict[str, str]) -> None:
    """Print how many CSV matches were found per model run folder, showing full paths."""
    folder_counts: dict[str, dict[str, int | str]] = {}
    for path in out_paths.values():
        d = os.path.dirname(path)
        base = os.path.basename(d)
        # go one level up if 'tables'
        if base == "tables":
            d = os.path.dirname(d)
            base = os.path.basename(d)
        folder_counts[d] = folder_counts.get(d, {"name": base, "count": 0})
        folder_counts[d]["count"] += 1

    if folder_counts:
        print("\nMatches per folder:")
        for full_path, info in sorted(folder_counts.items(), key=lambda x: x[0].lower()):
            print(f"  {info['name']:35s} {info['count']:>5d}   {full_path}")
        print()

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    
    float_format = float_fmt(args.precision)
    pd.options.display.float_format = ("{:.%df}" % args.precision).format   
    
    ensure_nlp(args.spacy_model)

    gt = read_ground_truth(args.annotation)
    
    
    # Keep ONLY poems where ALL rows have English filled
    grp = gt.groupby(gt["poem_id"].astype(str))["English"]
    fully_mask = grp.apply(lambda s: (s.astype(str).str.strip() != "").all())
    fully_ids = set(fully_mask[fully_mask].index.astype(str))

    # Filter GT to fully annotated poems only
    gt = gt[gt["poem_id"].astype(str).isin(fully_ids)].copy()

    
    song_ids = set(gt.poem_id.astype(str).tolist())
    print(f"Fully annotated poems (all English present): {len(song_ids)}")

    # model_folders = [p for p in args.output.split(",") if p]
    # folder_paths = [read_dir(f, song_ids) for f in model_folders]
    # out_paths = {k: v for fp in folder_paths for k, v in fp.items()}
    
    
    if "," in args.output:
        roots = [p for p in args.output.split(",") if p]
    else:
        roots = [args.output]

    out_paths = {}
    for root in roots:
        print(f"Searching under: {root}")
        found = find_all_csvs(root, song_ids)
        out_paths.update(found)
        
    print(f"Total matching CSVs found: {len(out_paths)}")
    summarize_matches_per_folder(out_paths)
    
    print(f"Found in model results: {len(out_paths)}")

    # NEW: make an output directory
    out_dir = resolve_outdir(args, roots)

    print("out_dir:", out_dir)
    
    base_name = os.path.splitext(os.path.basename(args.result))[0]
    result_path = os.path.join(out_dir, base_name + ".csv")

    missed = sorted([s for s in song_ids if s not in out_paths])

    evaluations = []
    broken = []
    std_lemma_errors: list[dict] = []
    
    # map poem_id -> true annotated length (as int)
    poem_lengths = (gt.groupby("poem_id").size())
    poem_lengths.index = poem_lengths.index.astype(str)
    poem_lengths = poem_lengths.to_dict()

    per_word_rows = []  # <--- NEW LINE (define the list early)


    for sid, path in out_paths.items():
        print(sid)
        try:
            df = read_model_output(path)
            if df is None:
                broken.append(sid)
                continue
            res = evaluate_poem(gt[gt.poem_id.astype(str) == sid], df)
            if res is None:
                broken.append(sid)
                continue
            
            run_id = _run_id_from_path(path)
            per = evaluate_poem_perword(
                gt_poem=gt[gt.poem_id.astype(str) == sid],
                model_df=df,
                poem_id=sid,
                run_id=run_id,
            )
            if per:
                per_word_rows.extend(per)
            
            
            evaluations.append([sid] + res)
            errs = collect_standard_lemma_errors(
                gt_poem=gt[gt.poem_id.astype(str) == sid],
                model_df=df,
                song_id=sid,
            )
            if errs:
                std_lemma_errors.extend(errs)
                
        except Exception as e:
            print(f"Skipping {sid}: {e}")
            broken.append(sid)
            continue
        
        
    # ensure a row for EVERY annotated poem:
    def _zeros_row(sid: str) -> list:
        L = int(poem_lengths.get(str(sid), 0))
        return [sid, L] + [0.0] * 14  # 14 metric columns after 'length'

    for sid in missed:
        evaluations.append(_zeros_row(sid))

    for sid in broken:
        evaluations.append(_zeros_row(sid))
        
        
        
    columns = [
        "song_id",
        "length",
        "norm_exact", "norm_distance", "norm_lev1",
        "local_exact", "local_distance", "local_lev1",
        "standard_exact", "standard_distance", "standard_lev1",
        "root_exact", "root_distance", "root_lev1",
        "translation_exact", "translation_semantic",
    ]
    result = pd.DataFrame(columns=columns, data=evaluations)
    result.to_csv(result_path, index=False, float_format=float_format)

    bw = best_worst(result, args.use)
    bw.to_csv(result_path.replace(".csv", "_best_worst.csv"), index=False, float_format=float_format)


    write_top10_worst_lists(result, result_path, float_format)
    write_standard_lemma_mistake_dictionary(std_lemma_errors, result_path, float_format)


    perword_cols = [
        "run_id", "poem_id", "verse_id", "word_id",
        "llm_norm", "norm_distance",
        "llm_local_lemma", "local_distance",
        "llm_standard", "standard_distance",
        "llm_root", "root_distance",
        "translation_semantic",
    ]
    perword_path = result_path.replace(".csv", "_perword.csv")
    
    if per_word_rows:
        pd.DataFrame(per_word_rows)[perword_cols].to_csv(
            perword_path, index=False, float_format=float_format
        )
        print(f"Wrote per-word table: {perword_path}")
    else:
        print("Per-word table: no aligned rows to write.")


    with open(result_path.replace(".csv", "_avg.txt"), "w", encoding="utf-8") as out:
        print(f"Manual: {len(song_ids)} Model: {len(out_paths)} Broken: {len(broken)}", file=out)
        
        means = result[columns[1:]].mean(numeric_only=True).round(args.precision)
        print(means, file=out)
        print("Missed:", missed, file=out)
        print("Broken:", broken, file=out)
        
    print("\nEvaluation complete.")
    print(f"All results saved under:\n  {os.path.abspath(out_dir)}\n")


if __name__ == "__main__":
    main()
