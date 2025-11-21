# runner/conversation.py
"""
Conversation helpers:
- normalize_conversation: normalizer to flatten/merge messages (
"""



def normalize_conversation(messages, keep_blocks=False):
    """
    Ensure conversation alternates user/assistant after optional system.
    - Keeps the first system message if present.
    - Groups consecutive same-role messages together.
    - If keep_blocks is False: flattens list content to strings (current behavior).
      If keep_blocks is True: preserves content lists (concats block lists).
    """
    normalized = []
    
    # keep optional system
    if messages and messages[0]["role"] == "system":
        normalized.append(messages[0])
        messages = messages[1:]

    # flatten content
    flat_msgs = []
    if not keep_blocks:
        # flatten content to strings (legacy behavior)
        for m in messages:
            if isinstance(m["content"], list):
                text = "".join(
                    part["text"] if isinstance(part, dict) and part.get("type") == "text" else str(part)
                    for part in m["content"]
                )
            else:
                text = str(m["content"])
            flat_msgs.append({"role": m["role"], "content": text})
    else:
        # preserve blocks; normalize each content to a list of text blocks
        def to_blocks(c):
            if c is None:
                return []
            if isinstance(c, list):
                out = []
                for it in c:
                    if isinstance(it, dict) and it.get("type") == "text":
                        out.append({"type": "text", "text": str(it.get("text", ""))})
                    else:
                        out.append({"type": "text", "text": str(it)})
                return out
            # string → single block
            return [{"type": "text", "text": str(c)}]
        for m in messages:
            flat_msgs.append({"role": m["role"], "content": to_blocks(m.get("content"))})

    # enforce alternation

    if keep_blocks:
        # Do NOT merge consecutive same-role messages — we need the first USER
        # message (invariant+input) separate from later retry/instruction blocks
        normalized.extend(flat_msgs)
    else:
        # legacy: merge consecutive same-role messages into one text blob
        last_role = None
        buffer = ""
        for m in flat_msgs:
            if m["role"] == last_role:
                buffer += "\n" + m["content"]
            else:
                if buffer:
                    normalized.append({"role": last_role, "content": buffer})
                last_role = m["role"]
                buffer = m["content"]
        if buffer:
            normalized.append({"role": last_role, "content": buffer})


    return normalized

