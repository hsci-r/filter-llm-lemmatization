# scripts/run_pipeline.py
"""JSON-only pipeline runner (always uses --pipeline <config.json>)."""

import argparse
import asyncio
import os, sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import shutil
import json
from asyncio import Semaphore

from runner.io_utils import make_run_dir, read_corpus, log_line
from runner.llm import make_async_client
from runner.pipeline import run_pipeline, persist_pipeline_results  # dispatcher + result persister

def parse_args():
    ap = argparse.ArgumentParser(description="JSON-only pipeline runner.")
    ap.add_argument("-c", "--corpus", required=True, help="Path to filter-pipeline CSV (first col id, last col line)")
    ap.add_argument("--pipeline", required=True, help="Path to pipeline.json (required)")
    ap.add_argument("-m", "--model", required=True, help="Model name or path")
    ap.add_argument("-i", "--id", default="", help="Comma-separated poem ids to process (optional)")
    ap.add_argument("-u", "--base_url", default="http://localhost:8000/v1/", help="Model base URL")
    ap.add_argument("-k", "--api_key", default="123", help="API key")
    ap.add_argument("-M", "--max_connections", type=int, default=1024, help="Max concurrent connections to the model client")
    ap.add_argument("--runs_base", default="runs", help="Base directory to create run folders")
    ap.add_argument("--run_label", default="pipeline", help="Label used in run folder name")
    # Keep these as fallbacks if JSON omits them
    ap.add_argument("-t", "--temperature", type=float, default=0.0)
    ap.add_argument("--chunk_size", type=int, default=100)
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("-T", "--think", action="store_true", help="Set if model outputs a 'thinking trace'")
    ap.add_argument(    "--lang",    default="",    help="Language code for prompt path substitution (e.g., fi, et). Used to replace {lang} in prompt paths.")

    args = ap.parse_args()
    if not args.pipeline.lower().endswith(".json"):
        ap.error("--pipeline must be a .json file")
    return args

def _abspath_list(base, paths):
    return [str((base / p).resolve()) if not Path(p).is_absolute() else p for p in paths]

def _abspath_maybe(base, p):
    if not p:
        return ""
    p = Path(p)
    return str(p if p.is_absolute() else (base / p).resolve())

def load_pipeline_cfg(cfg_path):
    """JSON-only loader. Relative paths in the JSON are resolved relative to the JSON file location."""
    p = Path(cfg_path).resolve()
    text = p.read_text(encoding="utf-8")
    cfg = json.loads(text or "{}")
    base = p.parent

    # Normalize top-level paths
    if "system" in cfg:
        if isinstance(cfg["system"], list):
            cfg["system"] = _abspath_list(base, cfg["system"])
        else:
            cfg["system"] = _abspath_maybe(base, cfg["system"])

    # Normalize per-step paths
    for step in cfg.get("steps", []) or []:
        if "task_prompts" in step:
            step["task_prompts"] = _abspath_list(base, step.get("task_prompts", []))
        ch = step.get("chunking")
        if isinstance(ch, dict) and ch.get("chunk_notice"):
            ch["chunk_notice"] = _abspath_maybe(base, ch["chunk_notice"])


    return cfg

async def main_async(args):
    # Load config first (so we can name the run directory using the model override if present)
    cfg = load_pipeline_cfg(args.pipeline)

    # Pick a model label for the run dir from JSON if available; otherwise a generic tag
    model_label = ""
    ov = cfg.get("overrides", {})
    if isinstance(ov, dict):
        model_label = ov.get("model", "") or ""
    model_label = Path(model_label).name or "model-from-json"

    # Create run dir and snapshot JSON for provenance
    run_dir = make_run_dir(args.runs_base, args.model, args.run_label, args.__dict__)
    log_line(run_dir, "MODE\tpipeline_json")
    try:
        shutil.copyfile(Path(args.pipeline).resolve(), Path(run_dir) / "pipeline.json")
    except Exception as e:
        log_line(run_dir, f"WARN\tcould not copy pipeline.json: {e}")

    # Init client & semaphore (pipeline may use it)
    client = make_async_client(args.base_url, args.api_key, args.max_connections)
    sem = Semaphore(args.max_connections)

    # Build dataset (poem_id -> full text)
    songs = read_corpus(args.corpus)
    wanted = set([s.strip() for s in args.id.split(",") if s.strip()]) if args.id else set(songs.keys())
    dataset = []
    for poem_id in wanted:
        if poem_id not in songs:
            print(poem_id, "not found in", args.corpus)
            continue
        dataset.append((poem_id, "\n".join(songs[poem_id])))

    # Run the pipeline dispatcher
    results = await run_pipeline(dataset, cfg, client, sem, run_dir, args)
    persist_pipeline_results(run_dir, results)

    print("Done", args.corpus, str(run_dir))

def main():
    asyncio.run(main_async(parse_args()))

if __name__ == "__main__":
    main()
