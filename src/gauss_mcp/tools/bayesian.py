import json

import numpy as np
import pandas as pd
from scipy import stats

from gauss_mcp.app import mcp
from gauss_mcp.state import datasets, mcmc_traces


def _get(name: str) -> tuple[pd.DataFrame | None, str | None]:
    if name not in datasets:
        return None, f"Dataset '{name}' not found. Use list_datasets() to see available datasets."
    return datasets[name], None


@mcp.tool()
def bayesian_update(
    prior_type: str,
    prior_params: str,
    likelihood_type: str,
    observations: list[float],
) -> str:
    """Closed-form Bayesian update using conjugate priors.

    Supported conjugate pairs and their prior_params JSON keys:
    - prior='beta'   + likelihood='binomial' : {"alpha": 1, "beta": 1}
      Observations should be 0/1 values (or a single [successes, trials] pair).
    - prior='normal' + likelihood='normal'   : {"mu": 0, "sigma": 1, "likelihood_sigma": 1}
      Estimates the mean of a normal population with known variance.
    - prior='gamma'  + likelihood='poisson'  : {"alpha": 1, "beta": 1}
      Estimates the Poisson rate (events per period). Observations = event counts per period.

    Args:
        prior_type: 'beta', 'normal', or 'gamma'.
        prior_params: JSON string with prior hyper-parameters (see above).
        likelihood_type: 'binomial', 'normal', or 'poisson'.
        observations: List of observed values.
    """
    try:
        params = json.loads(prior_params)
    except json.JSONDecodeError as e:
        return f"prior_params is not valid JSON: {e}"

    obs = np.array(observations, dtype=float)
    n = len(obs)

    # Beta-Binomial
    if prior_type == "beta" and likelihood_type == "binomial":
        a0 = float(params.get("alpha", 1))
        b0 = float(params.get("beta", 1))
        successes = int(obs.sum())
        failures = n - successes
        a1 = a0 + successes
        b1 = b0 + failures
        mean = a1 / (a1 + b1)
        mode = (a1 - 1) / (a1 + b1 - 2) if a1 > 1 and b1 > 1 else None
        var = (a1 * b1) / ((a1 + b1) ** 2 * (a1 + b1 + 1))
        lo, hi = stats.beta.ppf([0.025, 0.975], a1, b1)
        mode_str = f"{mode:.4f}" if mode is not None else "undefined (α or β ≤ 1)"
        return (
            f"Bayesian Update — Beta-Binomial\n\n"
            f"Prior  : Beta(α={a0}, β={b0})  mean={a0/(a0+b0):.4f}\n"
            f"Data   : n={n}  successes={successes}  failures={failures}\n\n"
            f"Posterior: Beta(α={a1}, β={b1})\n"
            f"  Mean     = {mean:.4f}\n"
            f"  Mode     = {mode_str}\n"
            f"  Std Dev  = {np.sqrt(var):.4f}\n"
            f"  95% CrI  = [{lo:.4f}, {hi:.4f}]\n\n"
            f"Interpretation: estimated proportion {mean:.4f} "
            f"(95% CrI [{lo:.4f}, {hi:.4f}]) after {n} observations."
        )

    # Normal-Normal (known likelihood variance)
    if prior_type == "normal" and likelihood_type == "normal":
        mu0 = float(params.get("mu", 0))
        s0 = float(params.get("sigma", 1))
        s_lik = float(params.get("likelihood_sigma", 1))
        x_bar = obs.mean()
        post_var = 1.0 / (1.0 / s0**2 + n / s_lik**2)
        post_mu = post_var * (mu0 / s0**2 + n * x_bar / s_lik**2)
        post_std = np.sqrt(post_var)
        lo, hi = stats.norm.ppf([0.025, 0.975], post_mu, post_std)
        return (
            f"Bayesian Update — Normal-Normal\n\n"
            f"Prior  : N(μ={mu0}, σ={s0})\n"
            f"Likelihood: N(θ, σ_known={s_lik})\n"
            f"Data   : n={n}  x̄={x_bar:.4f}\n\n"
            f"Posterior: N(μ={post_mu:.4f}, σ={post_std:.4f})\n"
            f"  Mean     = {post_mu:.4f}\n"
            f"  Std Dev  = {post_std:.4f}\n"
            f"  95% CrI  = [{lo:.4f}, {hi:.4f}]\n\n"
            f"Interpretation: estimated mean {post_mu:.4f} "
            f"(95% CrI [{lo:.4f}, {hi:.4f}]) after {n} observations."
        )

    # Gamma-Poisson
    if prior_type == "gamma" and likelihood_type == "poisson":
        a0 = float(params.get("alpha", 1))
        b0 = float(params.get("beta", 1))
        total = obs.sum()
        a1 = a0 + total
        b1 = b0 + n
        mean = a1 / b1
        mode = (a1 - 1) / b1 if a1 > 1 else 0.0
        var = a1 / b1**2
        lo, hi = stats.gamma.ppf([0.025, 0.975], a1, scale=1.0 / b1)
        return (
            f"Bayesian Update — Gamma-Poisson\n\n"
            f"Prior  : Gamma(α={a0}, β={b0})  mean rate={a0/b0:.4f}\n"
            f"Data   : n={n} periods  total events={int(total)}  observed rate={total/n:.4f}\n\n"
            f"Posterior: Gamma(α={a1}, β={b1})\n"
            f"  Mean     = {mean:.4f}\n"
            f"  Mode     = {mode:.4f}\n"
            f"  Std Dev  = {np.sqrt(var):.4f}\n"
            f"  95% CrI  = [{lo:.4f}, {hi:.4f}]\n\n"
            f"Interpretation: estimated Poisson rate {mean:.4f} "
            f"(95% CrI [{lo:.4f}, {hi:.4f}]) after {n} periods."
        )

    return (
        f"Unsupported conjugate pair: prior='{prior_type}', likelihood='{likelihood_type}'.\n"
        f"Supported pairs:\n"
        f"  beta   + binomial\n"
        f"  normal + normal\n"
        f"  gamma  + poisson"
    )


@mcp.tool()
def mcmc_sample(
    model_type: str,
    dataset_name: str,
    column: str | None = None,
    target: str | None = None,
    features: list[str] | None = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    trace_name: str = "default",
) -> str:
    """Run MCMC sampling with PyMC for Bayesian inference.

    Requires PyMC: pip install 'gauss-mcp[bayesian]'  (or pip install pymc arviz)

    model_type options:
    - 'normal_mean'        : Estimate μ and σ of a normal population. Requires column.
    - 'proportion'         : Estimate Bernoulli p from binary (0/1) data. Requires column.
    - 'linear_regression'  : Bayesian OLS with weakly informative priors. Requires target + features.

    Args:
        model_type: 'normal_mean', 'proportion', or 'linear_regression'.
        dataset_name: Name of the dataset.
        column: Column for normal_mean and proportion models.
        target: Target column for linear_regression.
        features: Feature columns for linear_regression.
        draws: Posterior draws per chain (default: 1000).
        tune: Tuning (warm-up) steps per chain (default: 1000).
        chains: Number of independent chains (default: 2).
        trace_name: Key under which to store the trace for posterior_summary().
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        return (
            "PyMC is not installed.\n"
            "Install with:  pip install 'gauss-mcp[bayesian]'\n"
            "Or:            pip install pymc arviz"
        )

    if dataset_name not in datasets:
        return f"Dataset '{dataset_name}' not found. Use list_datasets()."
    df = datasets[dataset_name]

    try:
        if model_type == "normal_mean":
            if not column or column not in df.columns:
                return f"column='{column}' not found in dataset."
            data = df[column].dropna().values.astype(float)
            with pm.Model():
                mu = pm.Normal("mu", mu=float(data.mean()), sigma=float(data.std()) * 2)
                sigma = pm.HalfNormal("sigma", sigma=float(data.std()) * 2)
                pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
                trace = pm.sample(
                    draws=draws, tune=tune, chains=chains,
                    progressbar=False, return_inferencedata=True,
                )

        elif model_type == "proportion":
            if not column or column not in df.columns:
                return f"column='{column}' not found in dataset."
            data = df[column].dropna().values.astype(float)
            if not np.isin(data, [0, 1]).all():
                return f"Column '{column}' must contain only 0 and 1 for the proportion model."
            with pm.Model():
                p = pm.Beta("p", alpha=1, beta=1)
                pm.Bernoulli("obs", p=p, observed=data.astype(int))
                trace = pm.sample(
                    draws=draws, tune=tune, chains=chains,
                    progressbar=False, return_inferencedata=True,
                )

        elif model_type == "linear_regression":
            if not target or not features:
                return "target and features are required for linear_regression."
            for col in [target] + features:
                if col not in df.columns:
                    return f"Column '{col}' not found."
            sub = df[[target] + features].dropna()
            y = sub[target].values.astype(float)
            X = sub[features].values.astype(float)
            # Standardise predictors for better sampling geometry
            X_mu = X.mean(axis=0)
            X_std = X.std(axis=0) + 1e-8
            X_s = (X - X_mu) / X_std
            with pm.Model():
                alpha = pm.Normal("alpha", mu=float(y.mean()), sigma=float(y.std()) * 2)
                betas = pm.Normal("betas", mu=0, sigma=1, shape=len(features))
                sigma = pm.HalfNormal("sigma", sigma=float(y.std()))
                mu_model = alpha + pm.math.dot(X_s, betas)
                pm.Normal("obs", mu=mu_model, sigma=sigma, observed=y)
                trace = pm.sample(
                    draws=draws, tune=tune, chains=chains,
                    progressbar=False, return_inferencedata=True,
                )

        else:
            return (
                f"Unknown model_type '{model_type}'.\n"
                "Available: normal_mean, proportion, linear_regression"
            )

        mcmc_traces[trace_name] = trace
        summary = az.summary(trace, round_to=4)
        return (
            f"MCMC complete — model='{model_type}', trace saved as '{trace_name}'\n"
            f"  draws={draws}  tune={tune}  chains={chains}\n\n"
            f"Posterior summary:\n{summary.to_string()}\n\n"
            f"Run posterior_summary('{trace_name}') for convergence diagnostics."
        )

    except Exception as e:
        return f"MCMC sampling failed: {e}"


@mcp.tool()
def posterior_summary(trace_name: str = "default") -> str:
    """Display the posterior summary and convergence diagnostics (R̂, ESS) for a stored MCMC trace.

    Args:
        trace_name: Name of the trace saved by mcmc_sample() (default: 'default').
    """
    try:
        import arviz as az
    except ImportError:
        return "arviz not installed. Run: pip install arviz"

    if trace_name not in mcmc_traces:
        available = list(mcmc_traces.keys())
        return (
            f"Trace '{trace_name}' not found.\n"
            f"Available traces: {available or ['none — run mcmc_sample() first']}"
        )

    trace = mcmc_traces[trace_name]
    try:
        summary = az.summary(trace, round_to=4)

        diag = []
        if "r_hat" in summary.columns:
            r_hat_max = summary["r_hat"].max()
            if r_hat_max < 1.01:
                diag.append(f"  R̂ max={r_hat_max:.4f}  — GOOD convergence (< 1.01)")
            elif r_hat_max < 1.05:
                diag.append(f"  R̂ max={r_hat_max:.4f}  — MARGINAL (1.01–1.05), consider more tuning")
            else:
                diag.append(f"  R̂ max={r_hat_max:.4f}  — POOR (> 1.05), increase tune= or reparametrize")

        if "ess_bulk" in summary.columns:
            ess_min = summary["ess_bulk"].min()
            status = "OK" if ess_min > 400 else "LOW — consider more draws"
            diag.append(f"  ESS bulk min={ess_min:.0f}  — {status}")

        return (
            f"Posterior summary — '{trace_name}':\n\n"
            f"{summary.to_string()}\n\n"
            f"Convergence diagnostics:\n" + "\n".join(diag)
        )
    except Exception as e:
        return f"Failed to summarize trace: {e}"


@mcp.tool()
def list_traces() -> str:
    """List all MCMC traces currently stored in memory."""
    if not mcmc_traces:
        return "No MCMC traces stored. Run mcmc_sample() first."
    lines = ["Stored MCMC traces:\n"]
    for name, trace in mcmc_traces.items():
        try:
            n_chains = trace.posterior.sizes.get("chain", "?")
            n_draws = trace.posterior.sizes.get("draw", "?")
            vars_ = list(trace.posterior.data_vars)
            lines.append(f"  '{name}': chains={n_chains}  draws={n_draws}  vars={vars_}")
        except Exception:
            lines.append(f"  '{name}': (trace info unavailable)")
    return "\n".join(lines)
