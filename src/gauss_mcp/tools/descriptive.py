import numpy as np
import pandas as pd
from scipy import stats

from gauss_mcp.app import mcp
from gauss_mcp.state import datasets


def _get(name: str) -> tuple[pd.DataFrame | None, str | None]:
    if name not in datasets:
        return None, f"Dataset '{name}' not found. Use list_datasets() to see available datasets."
    return datasets[name], None


@mcp.tool()
def describe(
    dataset_name: str,
    columns: list[str] | None = None,
) -> str:
    """Compute descriptive statistics: count, mean, std, min, percentiles (5/25/50/75/95),
    max, skewness, kurtosis, missing count and %.

    Args:
        dataset_name: Name of the dataset to analyze.
        columns: Column names to include (default: all numeric columns).
    """
    df, err = _get(dataset_name)
    if err:
        return err

    num = df.select_dtypes(include=[np.number])
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return f"Columns not found: {missing}"
        num = df[columns].select_dtypes(include=[np.number])

    if num.empty:
        return "No numeric columns found."

    out = num.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    out["skewness"] = num.skew()
    out["kurtosis"] = num.kurt()
    out["missing"] = df[num.columns].isnull().sum()
    out["missing%"] = (df[num.columns].isnull().mean() * 100).round(2)

    return f"Descriptive statistics — '{dataset_name}':\n\n{out.round(4).to_string()}"


@mcp.tool()
def frequency_table(
    dataset_name: str,
    column: str,
    bins: int | None = None,
    normalize: bool = False,
) -> str:
    """Compute a frequency table for a column.

    For categorical/string columns counts each unique value.
    For numeric columns optionally bins values into intervals.

    Args:
        dataset_name: Name of the dataset.
        column: Column name to analyze.
        bins: Number of bins for numeric data (activates binning).
        normalize: Return proportions instead of counts (default: False).
    """
    df, err = _get(dataset_name)
    if err:
        return err
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {list(df.columns)}"

    series = df[column].dropna()

    if pd.api.types.is_numeric_dtype(series) and bins:
        freq = pd.cut(series, bins=bins).value_counts(sort=False, normalize=normalize)
        freq.index = freq.index.astype(str)
    else:
        freq = series.value_counts(normalize=normalize, sort=True)

    value_label = "proportion" if normalize else "count"
    result = pd.DataFrame({"value": freq.index, value_label: freq.values})
    title = f"Frequency table — '{column}'" + (" (proportions)" if normalize else "")
    return f"{title}:\n\n{result.to_string(index=False)}"


@mcp.tool()
def correlation_matrix(
    dataset_name: str,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> str:
    """Compute the pairwise correlation matrix between numeric columns.

    Args:
        dataset_name: Name of the dataset.
        columns: Column names to include (default: all numeric columns).
        method: 'pearson', 'spearman', or 'kendall'.
    """
    df, err = _get(dataset_name)
    if err:
        return err

    if method not in ("pearson", "spearman", "kendall"):
        return "method must be one of: pearson, spearman, kendall"

    num = df.select_dtypes(include=[np.number])
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return f"Columns not found: {missing}"
        num = df[columns].select_dtypes(include=[np.number])

    if num.shape[1] < 2:
        return "At least 2 numeric columns required."

    corr = num.corr(method=method).round(4)
    return f"Correlation matrix ({method}) — '{dataset_name}':\n\n{corr.to_string()}"


@mcp.tool()
def distribution_fit(
    dataset_name: str,
    column: str,
    distributions: list[str] | None = None,
) -> str:
    """Fit probability distributions to a numeric column and rank them by KS goodness-of-fit.

    Args:
        dataset_name: Name of the dataset.
        column: Numeric column to fit.
        distributions: scipy.stats distribution names to test
            (default: norm, expon, gamma, lognorm, beta, weibull_min).
    """
    df, err = _get(dataset_name)
    if err:
        return err
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {list(df.columns)}"

    series = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return f"Column '{column}' is not numeric."

    data = series.values
    dist_names = distributions or ["norm", "expon", "gamma", "lognorm", "beta", "weibull_min"]

    results = []
    for name in dist_names:
        try:
            dist = getattr(stats, name)
            params = dist.fit(data)
            ks_stat, p_value = stats.kstest(data, name, args=params)
            log_lik = float(np.sum(dist.logpdf(data, *params)))
            aic = 2 * len(params) - 2 * log_lik
            param_str = ", ".join(f"{p:.4f}" for p in params)
            results.append({
                "distribution": name,
                "ks_stat": round(ks_stat, 4),
                "p_value": round(p_value, 4),
                "aic": round(aic, 2),
                "params": param_str,
            })
        except Exception:
            continue

    if not results:
        return "Could not fit any of the requested distributions."

    results.sort(key=lambda x: x["ks_stat"])
    out = pd.DataFrame(results)
    return (
        f"Distribution fit — '{column}' (n={len(data)}, ranked by KS statistic):\n\n"
        f"{out.to_string(index=False)}\n\n"
        "Note: lower KS stat = better fit; p > 0.05 means the distribution cannot be rejected at 5%."
    )
