# runner/reports.py

from pathlib import Path
from .io_utils import slugify

def _df_to_markdown_table(df, max_rows=None):
    """
    Render a DataFrame to a compact GitHub-style markdown table.
    Returns '' if df is empty/None.
    If max_rows is None, render all rows; otherwise cap to max_rows.
    """
    if df is None or getattr(df, "empty", True):
        return ""
    df_show = df.fillna("")
    if isinstance(max_rows, int) and max_rows > 0 and len(df_show) > max_rows:
        df_show = df_show.head(max_rows)
    cols = [str(c) for c in df_show.columns]
    lines = [
        "|" + "|".join(cols) + "|",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df_show.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)

def write_poem_report(run_dir, poem_id, df, stages=None, 
                      stage_preview_rows=25, final_preview_rows=None):
    """
    Per-poem Markdown report including per-stage outputs (tables + raw).
    Writes runs/.../reports/<poem_id>.md

    stage_preview_rows: cap for per-stage preview (int or None for all)
    final_preview_rows: cap for final table (None = show ALL)
    """
    p = Path(run_dir) / "reports" / f"{slugify(poem_id)}.md"
    lines = [f"# {poem_id}", ""]

    n = int(getattr(df, "shape", (0, 0))[0]) if df is not None else 0
    lines += [f"Final rows: {n}", ""]

    if stages:
        for rec in stages:
            name = rec.get("name", "stage")
            s_df = rec.get("df", None)
            raw  = (rec.get("raw", "") or "").strip()

            lines += [f"## Stage: {name}", ""]

            md = _df_to_markdown_table(s_df, max_rows=stage_preview_rows)
            if md:
                lines += [md, ""]
                # Add a collapsible full table if we truncated
                if isinstance(stage_preview_rows, int) and s_df is not None and len(s_df) > stage_preview_rows:
                    full_md = _df_to_markdown_table(s_df, max_rows=None)
                    lines += [
                        "<details>",
                        f"<summary>Show full table ({len(s_df)} rows)</summary>",
                        "",
                        full_md,
                        "",
                        "</details>",
                        "",
                    ]
            else:
                lines += ["[no table]", ""]

            if raw:
                lines += [
                    "<details>",
                    f"<summary>Raw model output ({len(raw)} chars)</summary>",
                    "",
                    "```",
                    raw,
                    "```",
                    "</details>",
                    "",
                ]

    # Final table
    lines += ["## Final table", ""]
    final_md = _df_to_markdown_table(df, max_rows=final_preview_rows)  # None => all rows
    lines += [final_md or "[no table]", ""]

    p.write_text("\n".join(lines), encoding="utf-8")
    return p
