import numpy as np
import pandas as pd

from gauss_mcp.app import mcp
from gauss_mcp.state import datasets


def _get(name: str) -> tuple[pd.DataFrame | None, str | None]:
    if name not in datasets:
        return None, f"Dataset '{name}' not found. Use list_datasets() to see available datasets."
    return datasets[name], None


def _to_series(df: pd.DataFrame, value_column: str, date_column: str | None) -> pd.Series:
    series = df[value_column].dropna()
    if date_column and date_column in df.columns:
        idx = pd.to_datetime(df.loc[series.index, date_column])
        series = series.copy()
        series.index = idx
        series = series.sort_index()
    return series


@mcp.tool()
def decompose_time_series(
    dataset_name: str,
    value_column: str,
    date_column: str | None = None,
    model: str = "additive",
    period: int | None = None,
) -> str:
    """Decompose a time series into trend, seasonal, and residual components
    using statsmodels seasonal_decompose.

    Args:
        dataset_name: Name of the dataset.
        value_column: Numeric column with the time series values.
        date_column: Optional datetime column to use as the index.
        model: 'additive' or 'multiplicative' (default: additive).
        period: Seasonal period (e.g., 12=monthly, 7=weekly). Auto-inferred when omitted.
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        return "statsmodels not installed. Run: pip install statsmodels"

    df, err = _get(dataset_name)
    if err:
        return err
    if value_column not in df.columns:
        return f"Column '{value_column}' not found."

    series = _to_series(df, value_column, date_column)

    if model not in ("additive", "multiplicative"):
        return "model must be 'additive' or 'multiplicative'."
    if model == "multiplicative" and (series <= 0).any():
        return "Multiplicative model requires all-positive values."

    if period is None:
        freq_map = {"MS": 12, "M": 12, "QS": 4, "Q": 4, "W": 52, "D": 7, "H": 24}
        freq_str = getattr(series.index, "freqstr", None) or ""
        period = freq_map.get(freq_str[:2], 2)

    min_obs = 2 * period
    if len(series) < min_obs:
        return f"Need ≥{min_obs} observations for period={period}. Got {len(series)}."

    result = seasonal_decompose(series, model=model, period=period)

    trend = result.trend.dropna()
    seasonal = result.seasonal.dropna()
    resid = result.resid.dropna()
    explained = (1 - resid.var() / series.var()) * 100

    return (
        f"Time series decomposition ({model}) — '{value_column}', period={period}\n"
        f"  n={len(series)}  mean={series.mean():.4f}  std={series.std():.4f}\n\n"
        f"Trend (first/last 5 non-NaN values):\n"
        f"{pd.concat([trend.head(5), trend.tail(5)]).round(4).to_string()}\n\n"
        f"Seasonal pattern (one period):\n"
        f"{seasonal.iloc[:period].round(4).to_string()}\n\n"
        f"Residuals:\n"
        f"  Mean={resid.mean():.4f}  Std={resid.std():.4f}  "
        f"Min={resid.min():.4f}  Max={resid.max():.4f}\n"
        f"  Variance explained by trend+seasonal: {explained:.1f}%"
    )


@mcp.tool()
def stationarity_test(
    dataset_name: str,
    column: str,
    test: str = "both",
    date_column: str | None = None,
) -> str:
    """Test time series stationarity with ADF and/or KPSS tests.

    ADF H₀: unit root present (non-stationary) — reject → stationary.
    KPSS H₀: series is stationary — reject → non-stationary.

    Args:
        dataset_name: Name of the dataset.
        column: Numeric column to test.
        test: 'adf', 'kpss', or 'both' (default).
        date_column: Optional datetime column to sort by before testing.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError:
        return "statsmodels not installed. Run: pip install statsmodels"

    df, err = _get(dataset_name)
    if err:
        return err
    if column not in df.columns:
        return f"Column '{column}' not found."

    series = _to_series(df, column, date_column).values
    parts = []

    if test in ("adf", "both"):
        res = adfuller(series, autolag="AIC")
        stat, p, lags, nobs, crit = res[0], res[1], res[2], res[3], res[4]
        decision = "Stationary (reject unit root H₀)" if p < 0.05 else "Non-stationary (fail to reject unit root H₀)"
        parts.append(
            f"Augmented Dickey-Fuller:\n"
            f"  ADF stat={stat:.4f}  p={p:.4f}  lags={lags}  n={nobs}\n"
            f"  Critical values: 1%={crit['1%']:.4f}  5%={crit['5%']:.4f}  10%={crit['10%']:.4f}\n"
            f"  → {decision}"
        )

    if test in ("kpss", "both"):
        try:
            res = kpss(series, regression="c", nlags="auto")
            stat, p, lags, crit = res[0], res[1], res[2], res[3]
            decision = "Non-stationary (reject stationarity H₀)" if p < 0.05 else "Stationary (fail to reject stationarity H₀)"
            parts.append(
                f"KPSS:\n"
                f"  KPSS stat={stat:.4f}  p={p:.4f}  lags={lags}\n"
                f"  Critical values: 10%={crit['10%']:.4f}  5%={crit['5%']:.4f}  "
                f"2.5%={crit['2.5%']:.4f}  1%={crit['1%']:.4f}\n"
                f"  → {decision}"
            )
        except Exception as e:
            parts.append(f"KPSS failed: {e}")

    if not parts:
        return f"Unknown test '{test}'. Use: adf, kpss, both."

    return f"Stationarity tests — '{column}' (n={len(series)})\n\n" + "\n\n".join(parts)


@mcp.tool()
def autocorrelation(
    dataset_name: str,
    column: str,
    nlags: int = 20,
    include_pacf: bool = True,
    date_column: str | None = None,
) -> str:
    """Compute the Autocorrelation Function (ACF) and optionally the Partial ACF (PACF).
    Significant lags are flagged — use them to guide ARIMA order selection.

    Args:
        dataset_name: Name of the dataset.
        column: Numeric time series column.
        nlags: Number of lags (default: 20).
        include_pacf: Also compute PACF (default: True).
        date_column: Optional datetime column to sort by.
    """
    try:
        from statsmodels.tsa.stattools import acf, pacf
    except ImportError:
        return "statsmodels not installed. Run: pip install statsmodels"

    df, err = _get(dataset_name)
    if err:
        return err
    if column not in df.columns:
        return f"Column '{column}' not found."

    series = _to_series(df, column, date_column).values
    nlags = min(nlags, len(series) // 2 - 1)

    acf_vals, acf_ci = acf(series, nlags=nlags, alpha=0.05)
    acf_df = pd.DataFrame({
        "lag": range(len(acf_vals)),
        "ACF": acf_vals.round(4),
        "CI_lower": acf_ci[:, 0].round(4),
        "CI_upper": acf_ci[:, 1].round(4),
    })
    acf_df["significant"] = ~((acf_df["CI_lower"] <= 0) & (acf_df["CI_upper"] >= 0))
    sig_acf = acf_df.loc[acf_df["significant"] & (acf_df["lag"] > 0), "lag"].tolist()

    output = (
        f"ACF — '{column}' (n={len(series)}, nlags={nlags})\n\n"
        f"{acf_df.to_string(index=False)}\n\n"
        f"Significant ACF lags (excl. 0): {sig_acf or 'none'}"
    )

    if include_pacf:
        try:
            pacf_vals, pacf_ci = pacf(series, nlags=nlags, alpha=0.05)
            pacf_df = pd.DataFrame({
                "lag": range(len(pacf_vals)),
                "PACF": pacf_vals.round(4),
                "CI_lower": pacf_ci[:, 0].round(4),
                "CI_upper": pacf_ci[:, 1].round(4),
            })
            pacf_df["significant"] = ~((pacf_df["CI_lower"] <= 0) & (pacf_df["CI_upper"] >= 0))
            sig_pacf = pacf_df.loc[pacf_df["significant"] & (pacf_df["lag"] > 0), "lag"].tolist()
            output += (
                f"\n\nPACF:\n{pacf_df.to_string(index=False)}\n\n"
                f"Significant PACF lags (excl. 0): {sig_pacf or 'none'}"
            )
        except Exception as e:
            output += f"\n\nPACF computation failed: {e}"

    return output


@mcp.tool()
def fit_arima(
    dataset_name: str,
    column: str,
    order: list[int] = [1, 1, 1],
    seasonal_order: list[int] | None = None,
    date_column: str | None = None,
) -> str:
    """Fit an ARIMA or SARIMA model to a time series column and report fit statistics.

    Args:
        dataset_name: Name of the dataset.
        column: Numeric time series column.
        order: (p, d, q) ARIMA order — default [1, 1, 1].
        seasonal_order: (P, D, Q, s) seasonal order, e.g. [1, 1, 1, 12] for monthly SARIMA.
        date_column: Optional datetime column to sort by.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return "statsmodels not installed. Run: pip install statsmodels"

    df, err = _get(dataset_name)
    if err:
        return err
    if column not in df.columns:
        return f"Column '{column}' not found."
    if len(order) != 3:
        return "order must have exactly 3 elements: [p, d, q]"

    series = _to_series(df, column, date_column)

    try:
        kwargs: dict = {"order": tuple(order)}
        if seasonal_order:
            if len(seasonal_order) != 4:
                return "seasonal_order must have 4 elements: [P, D, Q, s]"
            kwargs["seasonal_order"] = tuple(seasonal_order)

        fit = ARIMA(series, **kwargs).fit()

        coef_df = pd.DataFrame({
            "coef": fit.params,
            "std_err": fit.bse,
            "z": fit.tvalues,
            "p-value": fit.pvalues,
        }).round(4)

        return (
            f"ARIMA{tuple(order)}"
            + (f"×{tuple(seasonal_order)}" if seasonal_order else "")
            + f" — '{column}'\n"
            f"  n={len(series)}\n\n"
            f"Coefficients:\n{coef_df.to_string()}\n\n"
            f"Fit:\n"
            f"  AIC={fit.aic:.4f}  BIC={fit.bic:.4f}  HQIC={fit.hqic:.4f}\n"
            f"  Log-likelihood={fit.llf:.4f}\n\n"
            f"Residuals:\n"
            f"  Mean={fit.resid.mean():.4f}  Std={fit.resid.std():.4f}"
        )
    except Exception as e:
        return f"ARIMA fitting failed: {e}"


@mcp.tool()
def forecast_arima(
    dataset_name: str,
    column: str,
    steps: int = 10,
    order: list[int] = [1, 1, 1],
    seasonal_order: list[int] | None = None,
    confidence: float = 0.95,
    date_column: str | None = None,
) -> str:
    """Fit an ARIMA/SARIMA model and generate a multi-step ahead forecast with confidence intervals.

    Args:
        dataset_name: Name of the dataset.
        column: Numeric time series column.
        steps: Number of future periods to forecast (default: 10).
        order: (p, d, q) ARIMA order — default [1, 1, 1].
        seasonal_order: (P, D, Q, s) seasonal order for SARIMA.
        confidence: Confidence level for prediction intervals (default: 0.95).
        date_column: Optional datetime column to sort by.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return "statsmodels not installed. Run: pip install statsmodels"

    df, err = _get(dataset_name)
    if err:
        return err
    if column not in df.columns:
        return f"Column '{column}' not found."

    series = _to_series(df, column, date_column)

    try:
        kwargs: dict = {"order": tuple(order)}
        if seasonal_order:
            kwargs["seasonal_order"] = tuple(seasonal_order)

        fit = ARIMA(series, **kwargs).fit()
        forecast = fit.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        ci = forecast.conf_int(alpha=1 - confidence)

        fcast_df = pd.DataFrame({
            "step": range(1, steps + 1),
            "forecast": mean.values.round(4),
            f"CI_lower_{int(confidence*100)}%": ci.iloc[:, 0].values.round(4),
            f"CI_upper_{int(confidence*100)}%": ci.iloc[:, 1].values.round(4),
        })

        return (
            f"ARIMA{tuple(order)} forecast — '{column}'\n"
            f"  Training n={len(series)}  last observed={series.iloc[-1]:.4f}\n"
            f"  AIC={fit.aic:.2f}  BIC={fit.bic:.2f}\n\n"
            f"Last 5 observed values:\n{series.tail(5).round(4).to_string()}\n\n"
            f"Forecast ({steps} steps, {int(confidence*100)}% CI):\n"
            f"{fcast_df.to_string(index=False)}"
        )
    except Exception as e:
        return f"Forecasting failed: {e}"
