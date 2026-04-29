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
def t_test(
    dataset_name: str,
    column: str,
    test_type: str = "one_sample",
    group_column: str | None = None,
    popmean: float = 0.0,
    alternative: str = "two-sided",
) -> str:
    """Perform a t-test on a numeric column.

    test_type options:
    - 'one_sample' : Test if the column mean equals popmean.
    - 'two_sample' : Compare means of two groups defined by group_column (2 unique values).
      Auto-applies Welch correction when variances are unequal (Levene test).
    - 'paired'     : Paired test between column and a second numeric column (group_column = second col name).

    Args:
        dataset_name: Name of the dataset.
        column: Primary numeric column.
        test_type: 'one_sample', 'two_sample', or 'paired'.
        group_column: Group label column (two_sample) or second numeric column (paired).
        popmean: Null hypothesis mean for one_sample test (default: 0).
        alternative: 'two-sided', 'less', or 'greater'.
    """
    df, err = _get(dataset_name)
    if err:
        return err
    if column not in df.columns:
        return f"Column '{column}' not found."

    series = df[column].dropna()

    if test_type == "one_sample":
        stat, pval = stats.ttest_1samp(series, popmean, alternative=alternative)
        ci = stats.t.interval(0.95, df=len(series) - 1, loc=series.mean(), scale=stats.sem(series))
        return (
            f"One-sample t-test: '{column}' vs μ₀={popmean}\n"
            f"  n={len(series)}  mean={series.mean():.4f}  sd={series.std():.4f}\n"
            f"  t={stat:.4f}  df={len(series)-1}\n"
            f"  p-value ({alternative})={pval:.4f}\n"
            f"  95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]\n"
            f"  → {'Reject H₀' if pval < 0.05 else 'Fail to reject H₀'} at α=0.05"
        )

    elif test_type == "two_sample":
        if not group_column or group_column not in df.columns:
            return "group_column required for two_sample test."
        groups = df.groupby(group_column)[column].apply(lambda x: x.dropna().values)
        if len(groups) != 2:
            return f"two_sample requires exactly 2 groups, found {len(groups)}: {list(groups.index)}"
        g1_name, g2_name = groups.index
        g1, g2 = groups.iloc[0], groups.iloc[1]
        levene_f, levene_p = stats.levene(g1, g2)
        equal_var = levene_p > 0.05
        stat, pval = stats.ttest_ind(g1, g2, equal_var=equal_var, alternative=alternative)
        test_name = "Student's t" if equal_var else "Welch's t"
        return (
            f"Two-sample t-test ({test_name}): '{column}' by '{group_column}'\n"
            f"  '{g1_name}': n={len(g1)}  mean={g1.mean():.4f}  sd={g1.std():.4f}\n"
            f"  '{g2_name}': n={len(g2)}  mean={g2.mean():.4f}  sd={g2.std():.4f}\n"
            f"  Levene: F={levene_f:.4f}  p={levene_p:.4f}  "
            f"→ {'equal variances assumed' if equal_var else 'Welch correction applied'}\n"
            f"  t={stat:.4f}\n"
            f"  p-value ({alternative})={pval:.4f}\n"
            f"  → {'Reject H₀' if pval < 0.05 else 'Fail to reject H₀'} at α=0.05"
        )

    elif test_type == "paired":
        if not group_column or group_column not in df.columns:
            return "group_column must be the name of the second numeric column for paired test."
        series2 = df[group_column].dropna()
        n = min(len(series), len(series2))
        stat, pval = stats.ttest_rel(series.iloc[:n], series2.iloc[:n], alternative=alternative)
        diff = series.iloc[:n].values - series2.iloc[:n].values
        ci = stats.t.interval(0.95, df=n - 1, loc=diff.mean(), scale=stats.sem(diff))
        return (
            f"Paired t-test: '{column}' vs '{group_column}'\n"
            f"  n pairs={n}\n"
            f"  Mean difference={diff.mean():.4f}  sd={diff.std():.4f}\n"
            f"  t={stat:.4f}  df={n-1}\n"
            f"  p-value ({alternative})={pval:.4f}\n"
            f"  95% CI of difference=[{ci[0]:.4f}, {ci[1]:.4f}]\n"
            f"  → {'Reject H₀' if pval < 0.05 else 'Fail to reject H₀'} at α=0.05"
        )

    return f"Unknown test_type '{test_type}'. Use: one_sample, two_sample, paired."


@mcp.tool()
def anova(
    dataset_name: str,
    value_column: str,
    group_column: str,
    post_hoc: bool = True,
) -> str:
    """Perform a one-way ANOVA to compare means across groups.
    Optionally runs Tukey's HSD post-hoc test when the overall result is significant.

    Args:
        dataset_name: Name of the dataset.
        value_column: Numeric column with measured values.
        group_column: Categorical column defining the groups.
        post_hoc: Run Tukey's HSD if ANOVA is significant (default: True).
    """
    df, err = _get(dataset_name)
    if err:
        return err
    for col in [value_column, group_column]:
        if col not in df.columns:
            return f"Column '{col}' not found."

    groups = df.groupby(group_column)[value_column].apply(lambda x: x.dropna().values)
    if len(groups) < 2:
        return "ANOVA requires at least 2 groups."

    f_stat, p_val = stats.f_oneway(*groups.values)

    group_stats = df.groupby(group_column)[value_column].agg(["count", "mean", "std"]).round(4)
    group_stats.columns = ["n", "mean", "std"]

    result = (
        f"One-way ANOVA: '{value_column}' by '{group_column}'\n\n"
        f"Group statistics:\n{group_stats.to_string()}\n\n"
        f"  F={f_stat:.4f}  p-value={p_val:.4f}\n"
        f"  → {'Reject H₀ — at least one group mean differs' if p_val < 0.05 else 'Fail to reject H₀'} at α=0.05\n"
    )

    if post_hoc and p_val < 0.05:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            mask = df[value_column].notna()
            tukey = pairwise_tukeyhsd(df.loc[mask, value_column], df.loc[mask, group_column])
            result += f"\nTukey's HSD post-hoc:\n{tukey.summary()}"
        except ImportError:
            result += "\nInstall statsmodels for post-hoc Tukey's HSD."

    return result


@mcp.tool()
def chi_square_test(
    dataset_name: str,
    col1: str,
    col2: str,
) -> str:
    """Chi-square test of independence between two categorical columns.
    Reports χ², degrees of freedom, p-value, and Cramér's V effect size.

    Args:
        dataset_name: Name of the dataset.
        col1: First categorical column.
        col2: Second categorical column.
    """
    df, err = _get(dataset_name)
    if err:
        return err
    for col in [col1, col2]:
        if col not in df.columns:
            return f"Column '{col}' not found."

    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    effect = "small" if cramers_v < 0.1 else "medium" if cramers_v < 0.3 else "large"
    warning = "" if (expected >= 5).all() else "\n⚠ Some expected frequencies < 5 — chi-square may be unreliable."

    return (
        f"Chi-square test of independence: '{col1}' × '{col2}'\n\n"
        f"Contingency table:\n{contingency.to_string()}\n\n"
        f"  χ²={chi2:.4f}  df={dof}  p-value={p:.4f}\n"
        f"  Cramér's V={cramers_v:.4f}  ({effect} effect)\n"
        f"  → {'Reject H₀ — variables are dependent' if p < 0.05 else 'Fail to reject H₀ — variables appear independent'} at α=0.05"
        f"{warning}"
    )


@mcp.tool()
def linear_regression(
    dataset_name: str,
    target: str,
    features: list[str],
    intercept: bool = True,
) -> str:
    """Fit an OLS linear regression model and report coefficients, R², F-test, AIC/BIC,
    and a residuals normality check.

    Args:
        dataset_name: Name of the dataset.
        target: Dependent variable (numeric column).
        features: List of independent variable column names.
        intercept: Include an intercept (default: True).
    """
    df, err = _get(dataset_name)
    if err:
        return err

    try:
        import statsmodels.api as sm
    except ImportError:
        return "statsmodels not installed. Run: pip install statsmodels"

    for col in [target] + features:
        if col not in df.columns:
            return f"Column '{col}' not found."

    sub = df[[target] + features].dropna()
    y = sub[target]
    X = sub[features]
    if intercept:
        X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    coef_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p-value": model.pvalues,
        "CI_lower_95": model.conf_int()[0],
        "CI_upper_95": model.conf_int()[1],
    }).round(4)

    resid = model.resid
    sample = resid.values[:5000]
    _, sw_p = stats.shapiro(sample)

    return (
        f"OLS Linear Regression: '{target}' ~ {' + '.join(features)}\n"
        f"  n={int(model.nobs)}  df_resid={int(model.df_resid)}\n\n"
        f"Coefficients:\n{coef_df.to_string()}\n\n"
        f"Fit:\n"
        f"  R²={model.rsquared:.4f}  Adj R²={model.rsquared_adj:.4f}\n"
        f"  F={model.fvalue:.4f}  p={model.f_pvalue:.4f}\n"
        f"  AIC={model.aic:.2f}  BIC={model.bic:.2f}\n\n"
        f"Residuals:\n"
        f"  Mean={resid.mean():.4f}  Std={resid.std():.4f}\n"
        f"  Shapiro-Wilk: p={sw_p:.4f} → {'residuals appear normal' if sw_p > 0.05 else 'residuals non-normal'}"
    )


@mcp.tool()
def normality_test(
    dataset_name: str,
    column: str,
    test: str = "all",
) -> str:
    """Test whether a numeric column follows a normal distribution.

    Args:
        dataset_name: Name of the dataset.
        column: Numeric column to test.
        test: 'shapiro' (Shapiro-Wilk, best for n≤5000), 'ks' (Kolmogorov-Smirnov),
              'dagostino' (D'Agostino-Pearson), or 'all' (run all three, default).
    """
    df, err = _get(dataset_name)
    if err:
        return err
    if column not in df.columns:
        return f"Column '{column}' not found."

    series = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return f"Column '{column}' is not numeric."

    data = series.values
    lines = []

    def _row(label: str, stat: float, p: float) -> str:
        decision = "Fail to reject H₀ (appears normal)" if p > 0.05 else "Reject H₀ (not normal)"
        return f"  {label}: stat={stat:.4f}  p={p:.4f}  → {decision}"

    if test in ("shapiro", "all"):
        sample = np.random.choice(data, 5000, replace=False) if len(data) > 5000 else data
        label = "Shapiro-Wilk (sample n=5000)" if len(data) > 5000 else "Shapiro-Wilk"
        s, p = stats.shapiro(sample)
        lines.append(_row(label, s, p))

    if test in ("ks", "all"):
        z = (data - data.mean()) / data.std()
        s, p = stats.kstest(z, "norm")
        lines.append(_row("Kolmogorov-Smirnov", s, p))

    if test in ("dagostino", "all"):
        s, p = stats.normaltest(data)
        lines.append(_row("D'Agostino-Pearson", s, p))

    if not lines:
        return f"Unknown test '{test}'. Use: shapiro, ks, dagostino, all."

    header = (
        f"Normality tests — '{column}' (n={len(data)})\n"
        f"  Mean={data.mean():.4f}  Std={data.std():.4f}  "
        f"Skew={stats.skew(data):.4f}  Kurt={stats.kurtosis(data):.4f}\n"
    )
    return header + "\n".join(lines)
