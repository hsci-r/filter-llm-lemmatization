# runner/pipeline.py

import asyncio
from pathlib import Path

from runner.chunking import run_chunked_stage
from runner.io_utils import save_table_csv, log_line
from runner.reports import write_poem_report

from runner.io_utils import log_line

from runner.prompts import expand_task_prompts


# ---------- tiny local helpers (no external deps) ----------

def _load_text(p):
    return Path(p).read_text(encoding="utf-8")

def _load_parts(paths):
    return [_load_text(p) for p in paths]



def _assemble_parts_in_order(paths, input_block,
                             prev_output=None,
                             input_marker="INPUT",
                             prev_marker="PREV",
                             sep="\n\n"):
    parts = _load_parts(paths)
    body = sep.join(parts)

    # PREV handling
    if prev_output is not None and prev_marker in body:
        body = body.replace(prev_marker, prev_output)
    else:
        # drop stray PREV if no prev_output
        if prev_marker in body:
            body = body.replace(prev_marker, "")
            while sep + sep in body:
                body = body.replace(sep + sep, sep)
            body = body.strip()

    # INPUT handling (required)
    if input_marker in body:
        body = body.replace(input_marker, input_block)
    else:
        # append as last resort (we’re enforcing INPUT via config/discipline,
        # but this keeps you from silently losing input)
        if input_block:
            body = body + (sep if body else "") + input_block

    return body


def _df_to_markdown_table(df):
    """
    Build a simple GitHub-style markdown table from a pandas DataFrame.
    Assumes df has column names and stringifiable cells.
    """
    if df is None or getattr(df, "empty", True):
        return ""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = []
    for _, row in df.iterrows():
        cells = [str(row[c]) if row[c] is not None else "" for c in cols]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


# ---------- public API used by scripts/run_pipeline.py ----------




async def run_pipeline(dataset, cfg, async_client, sem, run_dir, args):
    """
    Run the configured steps for all poems.
    poems are processed in parallel (bounded by `sem`).
    Steps *within a poem* remain sequential to preserve PREV->NEXT dependencies.
    """
    # top-level overrides (fall back to CLI if not present)
    top = cfg.get("overrides", {}) or {}
    top_model       = top.get("model", getattr(args, "model", "")) or getattr(args, "model", "")
    top_temp        = float(top.get("temperature", getattr(args, "temperature", 0.0)))
    top_max_tokens  = int(top.get("max_tokens", getattr(args, "max_tokens", 8000)))
    top_chunk_size  = int(top.get("chunk_size",  getattr(args, "chunk_size", 100)))
    top_max_retries = int(top.get("max_retries", getattr(args, "max_retries", 2)))

    steps = cfg.get("steps", []) or []
    if not steps:
        raise RuntimeError("pipeline.json has no 'steps'")

    # Load system message text: allow str OR list of files
    lang = getattr(args, "lang", None)
    sys_spec = cfg.get("system", "") or ""
    system_text = ""
    try:
        if isinstance(sys_spec, list):
            sys_paths = expand_task_prompts(sys_spec, lang)
            parts = [Path(p).read_text(encoding="utf-8") for p in sys_paths]
            system_text = "\n\n".join(t.strip() for t in parts if t is not None)
        else:
            p = str(sys_spec)
            if lang:
                p = p.replace("{lang}", lang)
            if p:
                system_text = Path(p).read_text(encoding="utf-8")
    except Exception as e:
        log_line(run_dir, f"WARN\tsystem not loaded: {e}")
        system_text = ""
    async def _run_one_poem(poem_id, text):
        """Run all steps for a single poem under the semaphore."""
        async with sem:
            prev_output = None
            last_df = None
            stage_records = []

            for idx, step in enumerate(steps):
                name = step.get("name", f"stage_{idx+1}")

                # ---- prompts (task_prompts only) ----
                raw_task_prompts = step.get("task_prompts")
                if not isinstance(raw_task_prompts, list) or not raw_task_prompts:
                            raise RuntimeError("Step '%s' must define a non-empty 'task_prompts' array." % name)

                # Substitute {lang} inside prompt paths (simple, global)
                lang = getattr(args, "lang", None)
                task_prompts = expand_task_prompts(raw_task_prompts, lang)

                # --- split invariant vs input prompts ---
                invariant_paths = [p for p in task_prompts
                                   if not p.endswith("/input.txt") and not p.endswith("\\input.txt") and not p.endswith("/prev.txt") and not p.endswith("\\prev.txt")]
                input_paths     = [p for p in task_prompts
                                   if p.endswith("/input.txt") or p.endswith("\\input.txt") or p.endswith("/prev.txt") or p.endswith("\\prev.txt")]

                # invariant blocks: one block per file (no INPUT substitution)
                invariant_blocks = []
                for p in invariant_paths:
                    try:
                        invariant_blocks.append({"type": "text", "text": Path(p).read_text(encoding="utf-8")})
                    except Exception as e:
                        log_line(run_dir, f"WARN\tprompt not loaded: {p}: {e}")

                # input block: allow PREV/INPUT substitution using existing helper
                input_text_block = _assemble_parts_in_order(
                    input_paths if input_paths else [],
                    input_block=text,
                    prev_output=prev_output
                )
                user_blocks = list(invariant_blocks)
                user_blocks.append({"type": "text", "text": input_text_block})


                # ---- per-step overrides → top-level → CLI ----
                ov          = step.get("overrides", {}) or {}
                model       = ov.get("model", top_model)
                temperature = float(ov.get("temperature", top_temp))
                max_tokens  = int(ov.get("max_tokens", top_max_tokens))

                # ---- chunking (always chunked) ----
                ch         = step.get("chunking", {}) or {}
                csize      = int(ch.get("chunk_size",  top_chunk_size))
                mretries   = int(ch.get("max_retries", top_max_retries))
                cn_path    = ch.get("chunk_notice", "")
                chunk_notice_text = ""
                if cn_path:
                    try:
                        chunk_notice_text = Path(cn_path).read_text(encoding="utf-8")
                    except Exception as e:
                        log_line(run_dir, f"WARN\tchunk_notice not loaded: {e}")
                        chunk_notice_text = ""

                val        = step.get("validation", {}) or {}
                enforce_fc = bool(val.get("enforce_first_column", True))
                min_cols   = int(val.get("min_table_cols", 7))
                val_mode   = str(val.get("mode", "first_column_words")).lower()

                log_line(
                    run_dir,
                    f"STEP\t{poem_id}:{name}\tchunk_size={csize}\tmax_retries={mretries}\t"
                    f"chunk_notice={'yes' if chunk_notice_text else 'no'}\t"
                    f"enforce_first_column={'yes' if enforce_fc else 'no'} "
                )

                poem_id_ret, df, combined_text = await run_chunked_stage(
                    poem_id=poem_id,
                    text=text,
                    stage_name=name,
                    prompt_text=user_blocks,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    chunk_size=csize,
                    max_retries=mretries,
                    think=getattr(args, "think", False),
                    async_client=async_client,
                    run_dir=run_dir,
                    chunk_notice_text=chunk_notice_text,
                    enforce_first_column=enforce_fc,
                    min_table_cols=min_cols,
                    validation_mode=val_mode,
                    system_text=system_text      
                )

                last_df = df
                stage_records.append({
                    "name": name,
                    "df": df,
                    "raw": combined_text or ""
                })

                # feed previous stage output to next stage PREV
                prev_fmt = (step.get("prev_format") or "raw").lower()
                if prev_fmt in ("table_markdown", "table_md"):
                    table_md = _df_to_markdown_table(df)
                    prev_output = table_md if table_md.strip() else (combined_text or "")
                else:
                    prev_output = combined_text or ""

            return (poem_id, last_df, stage_records)

    # ----- launch all poems concurrently (bounded by sem) -----
    tasks = [asyncio.create_task(_run_one_poem(poem_id, text)) for poem_id, text in dataset]
    results = await asyncio.gather(*tasks)
    return results



def persist_pipeline_results(run_dir, results):
    """
    results: list[(poem_id, df)]
    Saves per-poem CSVs (if any), writes a simple report, and logs outcome.
    """

    for item in results:
        # Back-compat: accept either (poem_id, df) or (poem_id, df, stages)
        poem_id, df, stages = item
        if hasattr(df, "empty") and not df.empty:
            save_table_csv(run_dir, poem_id, df)
            write_poem_report(run_dir, poem_id, df, stages=stages)  # pass stages
            log_line(run_dir, f"OK\t{poem_id}\t{len(df)} rows")
        else:
            write_poem_report(run_dir, poem_id, df, stages=stages)  # still write a report showing stage outputs
            log_line(run_dir, f"EMPTY\t{poem_id}")
