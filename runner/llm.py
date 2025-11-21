# runner/llm.py
"""
Unified LLM helpers: one client dict, backend chosen by model name.

- No type hints, no OOP, no inner functions, no imports inside functions.
- Keep signatures used by scripts/run_pipeline.py:
    * make_async_client(base_url, api_key, max_connections)
    * get_completion(system_text, user_text, async_client, semaphore,
                     run_dir, stem, temperature, model, max_tokens)

Routing rule:
- If model name starts with "claude" (case-insensitive) -> Anthropic messages API
- Otherwise -> OpenAI-compatible chat.completions

Also exports:
    * raw_completion(messages, model, temperature, max_tokens, async_client)
      → returns (assistant_text, raw_json_string) with NO logging/IO.
"""

import json
import httpx
import openai
from asyncio import Semaphore

try:
    import anthropic
except Exception:
    anthropic = None  # allowed; we error only if we need it

from .conversation import normalize_conversation
from .io_utils import save_text, save_json, format_messages_for_log


# --------------------- client creation ---------------------

def make_async_client(base_url, api_key, max_connections):
    """
    Build a single 'client' object (a dict) that holds both backends.
    The dispatcher will pick based on the model name.
    """
    client = {}

    # OpenAI/vLLM-compatible client
    client["openai"] = openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(max_connections=max_connections)
        )
    )

    # Anthropic client (optional)
    if anthropic is not None:
        client["anthropic"] = anthropic.AsyncAnthropic(api_key=api_key)
    else:
        client["anthropic"] = None

    return client


# --------------------- small helpers ---------------------

# --- BATCH HELPERS (Claude only, no logging) ---

def _is_claude(model):
    return (model or "").strip().lower().startswith("claude")

def _build_anthropic_params(normalized, model, max_tokens, temperature):
    """
    Build Anthropic messages params from a normalized message list that may
    contain list-style block content. We merge:
      - System: all blocks → one single text block (cacheable)
      - First user message: all BUT the last block → one single text block (cacheable)
                           last block kept separate (NOT cacheable) for per-chunk/input
      - Subsequent user messages: passed through unchanged
    This guarantees <= 2 cache-tagged blocks total and avoids the 4-block limit.
    """
    # ---------- System (merge to one block, cacheable) ----------
    sys_blocks = None
    if normalized and normalized[0].get("role") == "system":
        scontent = normalized[0].get("content") or ""
        if isinstance(scontent, list):
            # merge all system blocks with a clear separator
            merged = "\n\n".join(
                str(b.get("text", "")) if isinstance(b, dict) else str(b)
                for b in scontent
            )
        else:
            merged = str(scontent)
        sys_blocks = [{
            "type": "text",
            "text": merged,
            "cache_control": {"type": "ephemeral"},
        }]

    # ---------- Users ----------
    user_msgs = []
    first_user_done = False
    for m in normalized:
        if (m.get("role") or "").lower() != "user":
            continue

        content = m.get("content") or []
        if not isinstance(content, list):
            content = [{"type": "text", "text": str(content)}]

        if not first_user_done:
            # Split prefix (invariant) vs. tail (variable)
            if len(content) == 0:
                user_msgs.append({"role": "user", "content": []})
            elif len(content) == 1:
                # Only a single block → treat as variable tail (no cache tag)
                user_msgs.append({"role": "user", "content": [dict(content[0]) ]})
            else:
                prefix = content[:-1]  # invariant – merge & cache
                tail   = content[-1:]  # variable – keep untagged

                merged_prefix = "\n\n".join(
                    str(b.get("text", "")) if isinstance(b, dict) else str(b)
                    for b in prefix
                ).strip()

                out_blocks = []
                if merged_prefix:
                    out_blocks.append({
                        "type": "text",
                        "text": merged_prefix,
                        "cache_control": {"type": "ephemeral"},
                    })
                # keep the last block as-is (no cache tag)
                out_blocks += [dict(tail[0])]

                user_msgs.append({"role": "user", "content": out_blocks})
            first_user_done = True
        else:
            # Pass through subsequent user messages unchanged
            user_msgs.append({"role": "user", "content": [dict(b) for b in content]})

    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": sys_blocks,
        "messages": user_msgs if user_msgs else [{"role": "user", "content": []}],
        # Do NOT set extra_headers here; batch API rejects it.
    }


async def claude_create_batch(requests, client):
    if anthropic is None:
        raise RuntimeError("Anthropic SDK not installed")
    # requests: list of {"custom_id": "...", "params": {... messages params ...}}
    resp = await client.messages.batches.create(requests=requests)
    try:
        return resp.id, resp.model_dump_json()
    except Exception:
        import json as _json
        return getattr(resp, "id", ""), _json.dumps(resp)

async def claude_retrieve_batch(batch_id, client):
    if anthropic is None:
        raise RuntimeError("Anthropic SDK not installed")
    resp = await client.messages.batches.retrieve(batch_id)
    # New SDK exposes `processing_status` (e.g., "in_progress", "ended")
    status = getattr(resp, "processing_status", None)
    if status is None:  # very old SDK fallback
        status = getattr(resp, "status", "")
    # Use the model's serializer; never json.dumps(resp)
    if hasattr(resp, "model_dump_json"):
        js = resp.model_dump_json()
    elif hasattr(resp, "model_dump"):
        js = json.dumps(resp.model_dump())
    else:
        js = str(resp)
    return status, js


async def claude_batch_results(batch_id, client):
    if anthropic is None:
        raise RuntimeError("Anthropic SDK not installed")
    # Correct API: results() returns an async iterator of batch items
    stream = await client.messages.batches.results(batch_id)
    out = []
    async for item in stream:
        try:
            out.append({"custom_id": item.custom_id, "result": item.result})
        except Exception:
            # best-effort fallback
            try:
                out.append(json.loads(item.model_dump_json()))
            except Exception:
                out.append({"raw": str(item)})
    return out

def _collapse_for_anthropic(messages):
    """
    Convert OpenAI-style messages into (system_text, user_text) for Anthropic.
    Assistant/tool roles are ignored for this single-turn path.
    """
    sys_buf = []
    user_buf = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content", "")

        if isinstance(content, list):
            parts = []
            for it in content:
                if isinstance(it, dict) and "text" in it:
                    parts.append(str(it["text"]))
                else:
                    parts.append(str(it))
            text = "".join(parts)
        else:
            text = str(content) if content is not None else ""

        if role == "system" and text.strip():
            sys_buf.append(text.strip())
        elif role == "user" and text:
            user_buf.append(text)

    return "\n".join(sys_buf).strip(), "\n\n".join(user_buf)


def _openai_extract(resp):
    """
    Return (assistant_text, raw_json_string) from an OpenAI response.
    """
    try:
        text = (resp.choices[0].message.content or "") if resp and resp.choices else ""
    except Exception:
        text = ""
    try:
        raw = resp.model_dump_json()
    except Exception:
        try:
            raw = json.dumps(resp)
        except Exception:
            raw = "{}"
    return text, raw


def _anthropic_extract(resp):
    """
    Return (assistant_text, raw_json_string) from an Anthropic response.
    """
    try:
        parts = []
        for blk in (resp.content or []):
            if getattr(blk, "type", None) == "text":
                parts.append(getattr(blk, "text", "") or "")
        text = "".join(parts)
    except Exception:
        text = ""
    try:
        raw = resp.model_dump_json()
    except Exception:
        try:
            raw = json.dumps(resp)
        except Exception:
            raw = "{}"
    return text, raw


# --------------------- core backend calls ---------------------

async def call_openai_chat(messages, model, max_tokens, temperature, client):
    """
    OpenAI/vLLM chat.completions call.
    Returns (assistant_text, raw_json_string).
    """
    resp = await client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return _openai_extract(resp)


async def call_anthropic_messages(messages, model, max_tokens, temperature, client):
    raise NonImplementedError("Should never be here")


# --------------------- dispatcher (by model name) ---------------------

async def _dispatch_by_model(messages, model, max_tokens, temperature, client):
    """
    Choose backend from model name.
    """
    m = (model or "").strip().lower()
    if m.startswith("claude"):
        anth = client.get("anthropic")
        if anth is None:
            raise RuntimeError("Anthropic client missing (and required by model name).")
        return await call_anthropic_messages(messages, model, max_tokens, temperature, anth)

    # default path: OpenAI-compatible
    oai = client.get("openai")
    if oai is None:
        raise RuntimeError("OpenAI client missing.")
    return await call_openai_chat(messages, model, max_tokens, temperature, oai)


# --------------------- public raw API (no logging) ---------------------

async def raw_completion(messages,
                         model,
                         temperature,
                         max_tokens,
                         async_client):
    """
    Routed single call with NO file I/O or logging.
    Returns (assistant_text, raw_json_string).
    Use this from chunking/loops that already handle artifacts.
    """
    return await _dispatch_by_model(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        client=async_client
    )


# --------------------- public entry ---------------------

async def get_completion(system_text,
                         user_text,
                         async_client,
                         semaphore,
                         run_dir,
                         stem,
                         temperature,
                         model,
                         max_tokens):
    """
    Single-call path used by pipeline stages.

    - Builds normalized messages
    - Logs readable input
    - Dispatches by model name using the single 'async_client' dict
    - Logs raw JSON + text output
    - Returns assistant text
    """
    messages = [
        {"role": "system", "content": system_text or ""},
        {"role": "user",   "content": user_text or ""},
    ]
    normalized = normalize_conversation(messages)

    # Save input (pretty)
    save_text(run_dir, "input", stem, "inp", format_messages_for_log(normalized))

    async with semaphore:
        content, raw_json = await _dispatch_by_model(
            messages=normalized,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            client=async_client
        )

    # Save outputs
    save_json(run_dir, stem, raw_json)
    save_text(run_dir, "input", stem, "out", content or "")

    return content or ""
