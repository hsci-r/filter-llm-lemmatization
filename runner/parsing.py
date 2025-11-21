# runner/parsing.py
"""
Parsing & validation utilities.

Functions:
- strip_until_think(text)          ← v2 version (TODO: paste your exact code)
- extract_table(response_text, min_cols=7)
                                   ← v1 logic + pre-strip (TODO: paste your loop)
- validate_first_column(rows, expected_words)
- table_to_dataframe(rows)
"""

import pandas as pd
import re
import unicodedata

def strip_until_think(text):
    """
    Keep everything AFTER the closing </think> tag (case-insensitive).
    If no tag is found, return the original text.
    Handles stray spaces like </think   > and trims leading newlines/space.
    """
    if not text:
        return ""
    s = str(text)
    last = None
    for m in re.finditer(r"</\s*think\s*>", s, flags=re.I):
        last = m
    if last:
        return s[last.end():].lstrip()  # drop tag + any leading whitespace/newlines
    return s.strip()



def extract_table(response, min_cols=7):
    response = strip_until_think(response)
    table = []
    header_len = None

    for l in response.split("\n"):
        l = l.strip()
        if not l.startswith("|"):
            continue
        # skip markdown separator like |---|---|
        if l.startswith("|-") or l.startswith("| -"):
            continue

        # split cells, drop leading/trailing pipe, trim spaces
        cells = [x.strip() for x in l.strip().strip("|").split("|")]

        # drop trailing empty cells that come from a final pipe
        while cells and cells[-1] == "":
            cells.pop()

        if not cells:
            continue

        if not table:
            # first table line must be a header; enforce min_cols only here
            if min_cols and len(cells) < min_cols:
                # not a real header — keep scanning until we find one
                continue
            table.append(cells)
            header_len = len(cells)
            continue

        # data rows: KEEP even if short; we'll pad in table_to_dataframe
        # if longer than header, truncate now to avoid accidental extra cols
        if header_len is not None:
            if len(cells) > header_len:
                cells = cells[:header_len]
        table.append(cells)

    return table



def _norm_token(s):
    """Normalize for comparison: NFKC, collapse spaces, strip, lowercase."""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def validate_first_column(rows, expected_words, normalize=True):
    """
    Validate that the first column of DATA ROWS (rows[1:]) matches expected_words
    in length and order.

    - rows: list[list[str]] where rows[0] is the header row.
    - expected_words: iterable of tokens to match against the first column.
    - normalize=True applies forgiving comparison:
        * Unicode NFKC normalization
        * whitespace collapsed & trimmed
        * case-insensitive
    """
    if not rows or len(rows) < 2:
        return False

    first_col = []
    for r in rows[1:]:
        if not isinstance(r, list) or len(r) == 0:
            first_col.append("")
        else:
            first_col.append(r[0])

    if normalize:
        first_col_norm = [_norm_token(x) for x in first_col]
        expected_norm = [_norm_token(x) for x in expected_words]
        return first_col_norm == expected_norm
    else:
        return list(first_col) == list(expected_words)




    

def table_to_dataframe(rows, poem_id=None, chunk_no=None):
    """Convert list-of-lists rows into a DataFrame, padding/trimming so
    every row has same number of columns as header."""
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    ncols = len(header)
    data = []
    for r in rows[1:]:
        # normalize length
        if len(r) < ncols:
            r = r + [""] * (ncols - len(r))
        elif len(r) > ncols:
            r = r[:ncols]
        data.append(r)

    try:
        return pd.DataFrame(data, columns=header)
    except Exception as e:
        print(f"DataFrame construction failed for {poem_id} chunk {chunk_no}: {e}")
        # fall back: at least return the rows as plain DataFrame
        return pd.DataFrame(rows)
