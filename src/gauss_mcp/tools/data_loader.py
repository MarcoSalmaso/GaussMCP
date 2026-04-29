import json

import pandas as pd

from gauss_mcp.app import mcp
from gauss_mcp.state import datasets


def _dataset_info(df: pd.DataFrame, name: str) -> str:
    dtype_lines = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    mem_kb = df.memory_usage(deep=True).sum() / 1024
    return (
        f"Dataset '{name}' loaded.\n"
        f"Shape  : {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Memory : {mem_kb:.1f} KB\n"
        f"Columns:\n{dtype_lines}"
    )


@mcp.tool()
def load_csv(
    file_path: str,
    dataset_name: str,
    separator: str = ",",
    encoding: str = "utf-8",
    parse_dates: bool = True,
) -> str:
    """Load a CSV file into memory as a named dataset.

    Args:
        file_path: Absolute or relative path to the CSV file.
        dataset_name: Name to assign to this dataset for subsequent analysis.
        separator: Column delimiter (default: comma).
        encoding: File encoding (default: utf-8).
        parse_dates: Attempt to parse date columns automatically.
    """
    try:
        df = pd.read_csv(
            file_path,
            sep=separator,
            encoding=encoding,
            parse_dates=parse_dates,
        )
        datasets[dataset_name] = df
        return _dataset_info(df, dataset_name)
    except Exception as e:
        return f"Error loading CSV: {e}"


@mcp.tool()
def load_excel(
    file_path: str,
    dataset_name: str,
    sheet_name: str = "0",
    parse_dates: bool = True,
) -> str:
    """Load an Excel file (.xlsx, .xls) into memory as a named dataset.

    Args:
        file_path: Absolute or relative path to the Excel file.
        dataset_name: Name to assign to this dataset for subsequent analysis.
        sheet_name: Sheet name or index (default: first sheet "0").
        parse_dates: Attempt to parse date columns automatically.
    """
    try:
        sn: int | str = int(sheet_name) if sheet_name.isdigit() else sheet_name
        df = pd.read_excel(file_path, sheet_name=sn, parse_dates=parse_dates)
        datasets[dataset_name] = df
        return _dataset_info(df, dataset_name)
    except Exception as e:
        return f"Error loading Excel: {e}"


@mcp.tool()
def load_from_sql(
    connection_string: str,
    query: str,
    dataset_name: str,
) -> str:
    """Execute a SQL SELECT query against a PostgreSQL database and store the result as a dataset.

    Args:
        connection_string: PostgreSQL connection string, e.g.
            'postgresql://user:password@host:5432/dbname'.
        query: SQL SELECT query to execute.
        dataset_name: Name to assign to this dataset for subsequent analysis.
    """
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(connection_string)
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        datasets[dataset_name] = df
        return _dataset_info(df, dataset_name)
    except ImportError:
        return "Error: sqlalchemy not installed. Run: pip install sqlalchemy psycopg2-binary"
    except Exception as e:
        return f"Error loading from SQL: {e}"


@mcp.tool()
def load_from_json(
    data: str,
    dataset_name: str,
) -> str:
    """Load data passed inline as a JSON string into memory as a named dataset.

    Accepts JSON in:
    - Records format: [{"col": val, ...}, ...]
    - Column-oriented format: {"col": [val, ...], ...}

    Args:
        data: JSON string representing the data.
        dataset_name: Name to assign to this dataset for subsequent analysis.
    """
    try:
        parsed = json.loads(data)
        if isinstance(parsed, (list, dict)):
            df = pd.DataFrame(parsed)
        else:
            return "Error: JSON must be a list of records or a dict of columns."
        datasets[dataset_name] = df
        return _dataset_info(df, dataset_name)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    except Exception as e:
        return f"Error loading from JSON: {e}"


@mcp.tool()
def list_datasets() -> str:
    """List all datasets currently loaded in memory with their shape and column names."""
    if not datasets:
        return "No datasets loaded. Use load_csv, load_excel, load_from_sql, or load_from_json."
    lines = ["Loaded datasets:\n"]
    for name, df in datasets.items():
        preview_cols = list(df.columns[:6])
        suffix = ", ..." if len(df.columns) > 6 else ""
        lines.append(
            f"  '{name}': {df.shape[0]}×{df.shape[1]}  "
            f"[{', '.join(preview_cols)}{suffix}]"
        )
    return "\n".join(lines)


@mcp.tool()
def preview_dataset(dataset_name: str, rows: int = 10) -> str:
    """Show the first N rows of a loaded dataset.

    Args:
        dataset_name: Name of the dataset.
        rows: Number of rows to show (default: 10).
    """
    if dataset_name not in datasets:
        return f"Dataset '{dataset_name}' not found. Use list_datasets() to see available datasets."
    df = datasets[dataset_name]
    return f"Dataset '{dataset_name}' — first {rows} rows:\n\n{df.head(rows).to_string()}"


@mcp.tool()
def drop_dataset(dataset_name: str) -> str:
    """Remove a dataset from memory.

    Args:
        dataset_name: Name of the dataset to remove.
    """
    if dataset_name not in datasets:
        return f"Dataset '{dataset_name}' not found. Available: {list(datasets.keys())}"
    del datasets[dataset_name]
    return f"Dataset '{dataset_name}' removed from memory."
