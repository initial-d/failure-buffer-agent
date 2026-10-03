"""Leakage metrics for Study 1.  All comparisons are within-instance (paired)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def paired_bootstrap_ci(d: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def sign_flip_pvalue(d: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    """Paired permutation test (random sign flips) of mean(d) == 0."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1, 1], (n_perm, len(d)))
    return float(((np.abs((flips * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n_perm + 1))


def summarize_diff(d, label: str) -> dict:
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    lo, hi = paired_bootstrap_ci(d)
    return {"metric": label, "n": len(d), "mean": float(d.mean()) if len(d) else np.nan,
            "ci_low": lo, "ci_high": hi, "p_perm": sign_flip_pvalue(d),
            "mean_abs": float(np.abs(d).mean()) if len(d) else np.nan,
            "frac_nonzero": float((np.abs(d) > 1e-9).mean()) if len(d) else np.nan}


def wide_main(df: pd.DataFrame, value: str = "p") -> pd.DataFrame:
    """Rows: (model, instance, ambiguity, explicit); columns: low/medium/high (+ none)."""
    main = df[df.condition_type.isin(["main", "exog"])]
    w = main.pivot_table(index=["model", "domain", "instance_id", "ambiguity", "indep"],
                         columns="buffer_level", values=value, aggfunc="first").reset_index()
    none = df[df.condition_type == "none"][["model", "instance_id", "ambiguity", value]] \
        .rename(columns={value: "none"})
    return w.merge(none, on=["model", "instance_id", "ambiguity"], how="left")


def bbl_table(df: pd.DataFrame, by=("model", "ambiguity", "indep")) -> pd.DataFrame:
    """Signed BBL = p(H)-p(L); absolute; range over L/M/H; also H-none and L-none (direction)."""
    rows = []
    for value in ["p", "mid", "width"]:
        w = wide_main(df, value)
        for keys, g in w.groupby(list(by)):
            keys = keys if isinstance(keys, tuple) else (keys,)
            base = dict(zip(by, keys), value=value)
            rows.append({**base, **summarize_diff(g["high"] - g["low"], "H-L")})
            if value == "p":
                rng = g[["low", "medium", "high"]].max(1) - g[["low", "medium", "high"]].min(1)
                rows.append({**base, **summarize_diff(rng, "range_LMH")})
                rows.append({**base, **summarize_diff(g["high"] - g["none"], "H-none")})
                rows.append({**base, **summarize_diff(g["low"] - g["none"], "L-none")})
    return pd.DataFrame(rows)


def control_table(df: pd.DataFrame, control_ambiguity: str = "sparse") -> pd.DataFrame:
    """Noise leakage, valence, positive/retry controls, negative buffer vs low."""
    rows = []
    d = df[df.ambiguity == control_ambiguity]
    for model, g in d.groupby("model"):
        piv = g.pivot_table(index="instance_id", columns="cond_key", values="p", aggfunc="first")
        noise_cols = [c for c in piv.columns if c.startswith("noise")]

        def add(label, series):
            rows.append({"model": model, **summarize_diff(series, label)})

        if noise_cols:
            add("noise_range (codename)", piv[noise_cols].max(1) - piv[noise_cols].min(1))
        add("buffer_range_LMH (implicit)", piv[["main_low_imp", "main_medium_imp", "main_high_imp"]].max(1)
            - piv[["main_low_imp", "main_medium_imp", "main_high_imp"]].min(1))
        add("valence - none", piv["valence"] - piv["none"])
        add("positive_control - none", piv["positive_control"] - piv["none"])
        add("retry_control - none", piv["retry_control"] - piv["none"])
        add("negative_buffer - low(imp)", piv["negative_buffer"] - piv["main_low_imp"])
        add("negative_buffer - high(imp)", piv["negative_buffer"] - piv["main_high_imp"])
    return pd.DataFrame(rows)


def add_cond_key(df: pd.DataFrame) -> pd.DataFrame:
    def key(r):
        if r.condition_type == "main":
            return f"main_{r.buffer_level}_{'exp' if r.explicit_independence else 'imp'}"
        if r.condition_type == "exog":
            return f"exog_{r.buffer_level}"
        if r.condition_type == "noise":
            return f"noise_{r.condition_id.split('__')[3]}"
        return r.condition_type
    df = df.copy()
    df["cond_key"] = df.apply(key, axis=1)
    df["indep"] = np.where(df.condition_type == "exog", "exog", np.where(df.explicit_independence, "exp", "imp"))
    return df
