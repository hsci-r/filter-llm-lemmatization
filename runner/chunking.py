# runner/chunking.py
"""
Chunked, validation-driven stage (Session A).
- Default: Claude models use Message Batches (cheaper, async).
- Others: realtime path via llm.raw_completion.

Design:
- Small helpers for: messages, artifacts, batch enqueue/poll, result handling.
- run_chunked_stage is a thin dispatcher.
"""

import asyncio
import os
import json
import pandas as pd
import traceback as _tb
import re, hashlib
from .conversation import normalize_conversation
from .parsing import extract_table, validate_first_column, table_to_dataframe
from .io_utils import artifact_prefix, save_text, save_json, format_messages_for_log
from . import llm  # routing + batch helpers

def _norm(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()

def _validate_first_column_verses(rows, verses, strict=True):
     if not rows:
         return False
     data_rows = rows[1:]  # skip header
     if not data_rows:
         return False
     col0 = [_norm(r[0]) for r in data_rows if r and len(r) > 0]
     exp  = [_norm(v) for v in verses]
     if strict:
         return len(col0) == len(exp) and all(c == e for c, e in zip(col0, exp))
     if len(col0) < len(exp):
         return False
     return all(any(e in c for c in col0) for e in exp)

def _make_retry_reminder_verses(verses):
    return (
        "Reminder: Your previous table did not match the required verses.\n"
        "Please output a table with EXACTLY these verses in the FIRST COLUMN, "
        "in the same order:\n"
        f"{verses}\n"
        "Do not add or skip rows. Include one row per verse."
    )

def _make_instruction_verses(chunk_no, verses):
    head = "Next chunk." if chunk_no > 0 else ""
    return (
        f"\n{head} Create a verse-by-verse translation table. "
        "The FIRST COLUMN must list exactly these verses in this order:\n"
        f"{verses}\n"
        "Do not add extra rows; include one row per verse."
    )

def split_verses(text, chunk_size):
    verses = [v for v in str(text).split("\n") if v.strip()]
    if not verses:
        return [[]]
    if len(verses) <= chunk_size:
        return [verses]
    return [verses[i:i + chunk_size] for i in range(0, len(verses), chunk_size)]

def _build_basic_messages_v1(system_text, prompt_text):
    sys = (system_text or "").strip() or ""
    # prompt_text may be a string or a list of {"type":"text","text":...} blocks
    content = prompt_text if isinstance(prompt_text, list) else [{'type': 'text', 'text': prompt_text}]
    return [
        {'role': 'system', 'content': sys},
        {'role': 'user', 'content': content}
    ]

def _make_instruction_v1(chunk_no, chunk_words):
    if chunk_no == 0:
        instruction = (
            "\n I expect the first column in the table to contain all these words, "
            f"in this order: {chunk_words}"
            " All columns should be filled for every word."
        )
    else:
        instruction = (
            "\nNext chunk. I expect the first column in the table to contain all these words, "
            f"in this order: {chunk_words}"
            " All columns should be filled for every word."
        )
    return instruction

def _make_retry_reminder_v1():
    return (
        "Reminder: Your previous table did not match exactly."
        "Please output a table whose FIRST COLUMN equals EXACTLY the given word sequence, "
        "same length and order, with no extra/missing rows."
        " ALL COLUMNS should be filled for every word even if some entries are identical."
    )



def _is_claude(model):
    return (model or "").strip().lower().startswith("claude")

def _dump_json(obj):
    try:
        return obj.model_dump_json()
    except Exception:
        try:
            return json.dumps(obj)
        except Exception:
            return "{}"

def _build_chunk_messages(basic_messages,
                          mode,
                          enforce_first_column,
                          chunk_no,
                          chunk_words,
                          chunk_verses,
                          attempt,
                          keep_blocks=False):

    """
    Returns normalized messages for one chunk attempt.
    Adds retry reminders and per-chunk instruction.
    """
    messages = list(basic_messages)

    # Retry reminder
    if attempt > 0:
        if enforce_first_column and mode == "first_column_words":
            messages.append({'role': 'user', 'content': _make_retry_reminder_v1()})
        elif mode.startswith("verse_"):
            messages.append({'role': 'user', 'content': _make_retry_reminder_verses(chunk_verses)})

    # Per-chunk instruction
    instruction = ""
    if mode == "first_column_words":
        if enforce_first_column:
            instruction = _make_instruction_v1(chunk_no, chunk_words)
    elif mode in ("verse_exact", "verse_loose"):
        instruction = _make_instruction_verses(chunk_no, chunk_verses)

    if instruction:
        messages.append({'role': 'user', 'content': instruction})

    return normalize_conversation(messages, keep_blocks=keep_blocks)
    
    
def _extract_anthropic_text(result_obj):

    """
    Pull plain text from an Anthropic messages result for both:
      - item.result.message.content  (common in batch results)
      - item.result.content          (single-call shape)
    Works for Pydantic objects or plain dicts.
    """
    def _get(obj, name):
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name)
        return None

    if result_obj is None:
        return ""

    # Prefer nested message.content if present
    msg = _get(result_obj, "message")
    content = None
    if msg is not None:
        content = _get(msg, "content")
    if content is None:
        content = _get(result_obj, "content")
    if not content:
        return ""

    parts = []
    for blk in content or []:
        # blk may be a Pydantic object or a dict
        btype = getattr(blk, "type", None) if not isinstance(blk, dict) else blk.get("type")
        if btype == "text":
            text = getattr(blk, "text", None) if not isinstance(blk, dict) else blk.get("text")
            if text:
                parts.append(str(text))
    return "".join(parts)

def _validate_rows(table_rows, mode, enforce_first_column, chunk_words, chunk_verses):
    if not table_rows:
        return False
    if mode == "first_column_words":
        return validate_first_column(table_rows, chunk_words) if enforce_first_column else True
    if mode == "verse_exact":
        return _validate_first_column_verses(table_rows, chunk_verses, strict=True)
    if mode == "verse_loose":
        return _validate_first_column_verses(table_rows, chunk_verses, strict=False)
    return True

def _rows_to_df(table_rows, poem_id, chunk_no):
    if not table_rows:
        return pd.DataFrame()
    try:
        return table_to_dataframe(table_rows, poem_id=poem_id, chunk_no=chunk_no)
    except Exception as e:
        print("DataFrame construction failed for", poem_id, "chunk", chunk_no, ":", e)
        _tb.print_exc()
        return pd.DataFrame()


# ----------------------- Claude batch path -----------------------

async def _run_stage_claude_batch(poem_id,
                                  text,
                                  stage_name,
                                  prompt_text,
                                  model,
                                  temperature,
                                  max_tokens,
                                  chunk_size,
                                  max_retries,
                                  run_dir,
                                  chunk_notice_text,
                                  enforce_first_column,
                                  min_table_cols,
                                  validation_mode,
                                  system_text,
                                  async_client):
    """
    Batch-by-default for Claude:
    - attempt 0 for all chunks in one batch
    - optional follow-up batches for failures until max_retries
    """
    # Base messages (v1-style) + optional global notice
    basic_messages = _build_basic_messages_v1(system_text, prompt_text)
    verse_chunks = split_verses(text, chunk_size)
    if len(verse_chunks) > 1 and chunk_notice_text:
        basic_messages.append({'role': 'user', 'content': chunk_notice_text})

    mode = (validation_mode or "first_column_words").lower()
    chunks_words  = [(" ".join(vc)).split() for vc in verse_chunks]
    chunks_verses = ["\n".join(vc).split("\n") for vc in verse_chunks]

    # Helper: build batch requests for a subset of chunk indices at a given attempt
    def _build_batch_requests(indices, attempt):
        
        reqs = []
        for i in indices:
            normalized = _build_chunk_messages(
                basic_messages, mode, enforce_first_column, i, chunks_words[i], chunks_verses[i], attempt, keep_blocks=True
            )
            stem = artifact_prefix(run_dir, poem_id, stage_name, chunk_no=i, attempt_no=attempt)
            save_text(run_dir, "input", stem, "inp", format_messages_for_log(normalized))
            params = llm._build_anthropic_params(normalized, model, max_tokens, temperature)
            
            # --- Build a Claude-safe custom_id (<=64, only [A-Za-z0-9_-]) ---
            # Keep a parseable suffix "--{i}--{attempt}" so we can recover indices later.

            suffix = f"--{i}--{attempt}"            # stays intact for parsing
            # sanitize prefix (poem_id + stage_name), replace illegal chars with "_"
            prefix_raw = f"{poem_id}--{stage_name}"
            prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(prefix_raw)).strip("_") or "id"
            # add a short hash to preserve uniqueness when we truncate
            h = hashlib.sha1(prefix.encode("utf-8")).hexdigest()[:8]
            core = f"{prefix}--{h}"
            # ensure total length <= 64 by truncating the core (suffix kept intact)
            max_core_len = max(1, 64 - len(suffix))
            core = core[:max_core_len]
            custom_id = f"{core}{suffix}"
            reqs.append({"custom_id": custom_id, "params": params})
            
        return reqs

    anth = async_client.get("anthropic")
    if anth is None:
        raise RuntimeError("Anthropic client missing")

    # State across attempts
    per_chunk_texts = [""] * len(verse_chunks)
    per_chunk_rows  = [None] * len(verse_chunks)
    done            = [False] * len(verse_chunks)

    attempt = 0
    while True:
        # Build and submit batch for all not-done chunks
        pending = [i for i, d in enumerate(done) if not d]
        if not pending:
            break

        requests = _build_batch_requests(pending, attempt)
        batch_id, batch_json = await llm.claude_create_batch(requests, anth)
        # Save batch envelope once per attempt
        save_json(run_dir, artifact_prefix(run_dir, poem_id, stage_name, attempt_no=attempt), batch_json)

        # Poll until finished
        while True:
            status, status_json = await llm.claude_retrieve_batch(batch_id, anth)
            if status in ("ended", "failed", "cancelled"):
                break
            await asyncio.sleep(30)
            
        # Fetch results
        results = await llm.claude_batch_results(batch_id, anth)

        ## Debug
        
        # print(f"Batch {batch_id} returned {len(results)} items")
        # print(results)
        # if results[:1]:
        #     r0 = results[0]
        #     ro = r0.get("result")
        #     print("result has keys:",
        #     list(ro.__dict__.keys()) if hasattr(ro, "__dict__") else (list(ro.keys()) if isinstance(ro, dict) else type(ro)))
        #     print(ro)
        #     
        # 
        # # --- Debug: detect missing results ---
        # 
        # got = set()
        # for item in results:
        #     cid = item.get("custom_id") or ""
        #     try:
        #         _, idx_str, att_str = cid.rsplit("--", 2)
        #         got.add(int(idx_str))
        #     except Exception as e:
        #         print(e)
        #         pass
        # 
        # missing = [i for i in pending if i not in got]
        # if missing:
        #     print("Batch completed with missing results for indices:", missing)
        # --- End debug ---



        # Map and validate
        for item in results:
            cid = item.get("custom_id", "")
            try:
                # We only rely on the suffix we constructed: "--{idx}--{attempt}"
                _, idx_str, att_str = cid.rsplit("--", 2)
                idx = int(idx_str)
                att = int(att_str)    
                
                
            except Exception:
                continue

            result_obj = item.get("result")

            # Inspect usage to confirm cache behavior
            usage = getattr(result_obj, "usage", None)
            if usage is None:
            # some SDKs nest usage under result.message
                msg = getattr(result_obj, "message", None)
                usage = getattr(msg, "usage", None)

            if usage is not None:
                # These fields typically exist on Claude 3.7:
                # cache_creation_input_tokens, cache_read_input_tokens, input_tokens, output_tokens
                try:
                    cc = getattr(usage, "cache_creation_input_tokens", None)
                    cr = getattr(usage, "cache_read_input_tokens", None)
                    it = getattr(usage, "input_tokens", None)
                    ot = getattr(usage, "output_tokens", None)
                    print(f"USAGE idx={idx} att={att} "
                          f"cache_create={cc} cache_read={cr} "
                          f"in={it} out={ot}")
                except Exception as e:
                    print(e)
                    pass

            
            raw_json = _dump_json(result_obj)
            content = _extract_anthropic_text(result_obj)

            stem = artifact_prefix(run_dir, poem_id, stage_name, chunk_no=idx, attempt_no=att)
            save_json(run_dir, stem, raw_json)
            save_text(run_dir, "input", stem, "out", content)

            # Parse table & validate
            rows = extract_table(content, min_cols=min_table_cols)
            ok = _validate_rows(rows, mode, enforce_first_column, chunks_words[idx], chunks_verses[idx])
            if ok:
                per_chunk_rows[idx]  = rows
                per_chunk_texts[idx] = content
                done[idx] = True
            else:
                # keep for retry if attempts left
                done[idx] = False

        if all(done) or attempt >= max_retries:
            break
        attempt += 1

    # Build final DataFrame and combined raw text
    per_chunk_dfs = [_rows_to_df(per_chunk_rows[i], poem_id, i) for i in range(len(verse_chunks))]
    if any((not d.empty) for d in per_chunk_dfs):
        final_df = pd.concat([d for d in per_chunk_dfs if not d.empty], ignore_index=True)
    else:
        final_df = pd.DataFrame()

    combined_text = "\n\n".join([t for t in per_chunk_texts if t])
    return (poem_id, final_df, combined_text)


# ----------------------- realtime path (OpenAI/vLLM) -----------------------

async def _run_stage_realtime(poem_id,
                              text,
                              stage_name,
                              prompt_text,
                              model,
                              temperature,
                              max_tokens,
                              chunk_size,
                              max_retries,
                              run_dir,
                              chunk_notice_text,
                              enforce_first_column,
                              min_table_cols,
                              validation_mode,
                              system_text,
                              async_client):
    """
    Original v1-style realtime path (also used for non-Claude models).
    """
    basic_messages = _build_basic_messages_v1(system_text, prompt_text)
    verse_chunks = split_verses(text, chunk_size)
    if len(verse_chunks) > 1 and chunk_notice_text:
        basic_messages.append({'role': 'user', 'content': chunk_notice_text})

    mode = (validation_mode or "first_column_words").lower()
    chunks_words  = [(" ".join(vc)).split() for vc in verse_chunks]
    chunks_verses = ["\n".join(vc).split("\n") for vc in verse_chunks]

    async def _process_one_chunk(i):
        attempt = 0
        best_content = ""
        rows = None

        while True:
            normalized = _build_chunk_messages(
                basic_messages, mode, enforce_first_column, i, chunks_words[i], chunks_verses[i], attempt)
            
            stem = artifact_prefix(run_dir, poem_id, stage_name, chunk_no=i, attempt_no=attempt)
            save_text(run_dir, "input", stem, "inp", format_messages_for_log(normalized))

            try:
                content, raw_json = await llm.raw_completion(
                    messages=normalized,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    async_client=async_client
                )
            except Exception as e:
                print("Exception while processing", poem_id, "chunk", i, ":", e)
                _tb.print_exc()
                if attempt >= max_retries:
                    break
                attempt += 1
                continue

            save_json(run_dir, stem, raw_json)
            best_content = content
            save_text(run_dir, "input", stem, "out", content)

            new_rows = extract_table(content, min_cols=min_table_cols)
            if new_rows:
                rows = new_rows

            ok = _validate_rows(rows, mode, enforce_first_column, chunks_words[i], chunks_verses[i]) if rows else False
            if ok or attempt >= max_retries:
                break
            attempt += 1

        df = _rows_to_df(rows, poem_id, i)
        return (i, df, best_content)

    tasks = [asyncio.create_task(_process_one_chunk(i)) for i in range(len(chunks_words))]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])

    per_chunk_dfs   = [r[1] for r in results]
    per_chunk_texts = [r[2] for r in results]

    if any((not d.empty) for d in per_chunk_dfs):
        try:
            final_df = pd.concat([d for d in per_chunk_dfs if not d.empty], ignore_index=True)
        except Exception as e:
            print(e)
            print(per_chunk_dfs)
            final_df = pd.DataFrame()
    else:
        final_df = pd.DataFrame()

    combined_text = "\n\n".join([t for t in per_chunk_texts if t])
    return (poem_id, final_df, combined_text)


# ----------------------- public dispatcher -----------------------

async def run_chunked_stage(poem_id,
                            text,
                            stage_name,
                            prompt_text,
                            model,
                            temperature,
                            max_tokens,
                            chunk_size,
                            max_retries,
                            think,
                            async_client,
                            run_dir,
                            chunk_notice_text=None,
                            enforce_first_column=True,
                            min_table_cols=7,
                            validation_mode="first_column_words",
                            system_text=None):
    """
    Thin dispatcher:
    - Claude → batch mode
    - Others → realtime mode
    """
    if _is_claude(model):
        return await _run_stage_claude_batch(
            poem_id, text, stage_name, prompt_text, model, temperature, max_tokens,
            chunk_size, max_retries, run_dir, chunk_notice_text or "",
            enforce_first_column, min_table_cols, validation_mode or "first_column_words",
            system_text or "", async_client
        )
    else:
        return await _run_stage_realtime(
            poem_id, text, stage_name, prompt_text, model, temperature, max_tokens,
            chunk_size, max_retries, run_dir, chunk_notice_text or "",
            enforce_first_column, min_table_cols, validation_mode or "first_column_words",
            system_text or "", async_client
        )
