# runner/prompts.py
"""
- Keep it literal. Files are read verbatim (UTF-8).
- Markers:
    INPUT  -> replaced with the poem/input block
    PREV   -> replaced with previous model output (sequential mode)
"""

from pathlib import Path



def _sub_lang(path, lang):
    return path.replace("{lang}", lang) if (lang and isinstance(path, str)) else path

def expand_task_prompts(task_prompts, lang):
    """
    Return a flat list of file paths with {lang} substituted.
    Accepts either strings or {"path": "..."} objects.
    Backward compatible: plain strings still work unchanged.
    """
    out = []
    for it in (task_prompts or []):
        if isinstance(it, str):
            out.append(_sub_lang(it, lang))
        elif isinstance(it, dict):
            p = _sub_lang(it.get("path", ""), lang)
            if p:
                out.append(p)
        # else ignore
    return out



def load_part(path):
    p = Path(path)
    return p.read_text(encoding="utf-8")


def load_parts(paths):
    """
    paths: list[str|Path]
    returns: list[str]
    """
    return [load_part(p) for p in paths]


def assemble_user_text(parts, input_block, prev_output=None,
                       input_marker="INPUT", prev_marker="PREV", sep="\n\n"):
    """
    Assemble a user message from already-loaded strings.

    - Joins parts with sep.
    - Replaces INPUT marker with input_block.
    - If prev_output is not None, replaces PREV with prev_output.
      If prev_output is None and prev_marker exists, leaves it as-is (explicit).
    """
    body = sep.join(parts)

    if input_marker in body:
        body = body.replace(input_marker, input_block)
    else:
        # fallback: append input at the end if no marker present
        if input_block:
            body = body + (sep if body else "") + input_block

    if prev_output is not None and prev_marker in body:
        body = body.replace(prev_marker, prev_output)

    return body


# -----------------------
# MODE 1: MERGED (single call)
# -----------------------

def build_merged_prompt(system_path, user_part_paths, input_block,
                        input_marker="INPUT", sep="\n\n"):
    """
    Returns (system_text, user_text)
    """
    system_text = load_part(system_path) if system_path else ""
    parts = load_parts(user_part_paths)
    user_text = assemble_user_text(parts, input_block,
                                   prev_output=None,
                                   input_marker=input_marker,
                                   prev_marker="PREV",
                                   sep=sep)
    return system_text, user_text


# -----------------------
# MODE 2: SEQUENTIAL (multi-call)
# -----------------------

def build_sequential_steps(system_path, step_part_paths_list, input_block,
                           input_marker="INPUT", sep="\n\n"):
    """
    Prepare step-by-step (system_text, user_text) pairs.

    step_part_paths_list: list of steps, each step is a list of file paths.
                          Example: [
                            ["prompts/context.txt", "prompts/task_translate.txt"],
                            ["prompts/task_improve.txt"],
                            ["prompts/analysis_prompt.txt"]
                          ]

    Returns: list of dicts:
      [{"system": str, "user": str}, ...]
    NOTE: PREV is NOT substituted here; do it at runtime so you can
          pass the previous step's actual model output.
    """
    system_text = load_part(system_path) if system_path else ""
    steps = []
    for paths in step_part_paths_list:
        parts = load_parts(paths)
        # leave PREV untouched here (prev_output=None)
        user_text = assemble_user_text(parts, input_block,
                                       prev_output=None,
                                       input_marker=input_marker,
                                       prev_marker="PREV",
                                       sep=sep)
        steps.append({"system": system_text, "user": user_text})
    return steps


def substitute_prev_in_step(step_user_text, prev_output, prev_marker="PREV"):
    """
    Small helper for runtime: replace PREV in the step's user text
    with the actual previous output.
    """
    if prev_output is None:
        return step_user_text
    if prev_marker in step_user_text:
        return step_user_text.replace(prev_marker, prev_output)
    return step_user_text
