import numpy as np
import pandas as pd
from typing import Any


def fmt_float(v: Any, decimals: int = 4) -> str:
    if isinstance(v, (float, np.floating)) and np.isfinite(v):
        return f"{v:.{decimals}f}"
    return str(v)


def df_to_text(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "(empty)"
    if len(df) > max_rows:
        head = df.head(max_rows // 2)
        tail = df.tail(max_rows // 2)
        ellipsis_row = pd.DataFrame([["..."] * len(df.columns)], columns=df.columns)
        df = pd.concat([head, ellipsis_row, tail], ignore_index=True)
    return df.to_string(index=True)


def section(title: str, content: str) -> str:
    bar = "─" * (len(title) + 4)
    return f"\n┌{bar}┐\n│  {title}  │\n└{bar}┘\n{content}\n"
