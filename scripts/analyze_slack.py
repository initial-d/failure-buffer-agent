"""Epistemic slack as a continuous quantity, measured from the models' own answers.

Idea. Instead of treating evidence as three discrete types, use the width of the model's own 90%
interval under the buffer-free condition as a continuous measure of how much the evidence leaves open.
Wider interval = more slack. Then ask whether the size of the buffer effect increases with slack,
pooling all three evidence types into one gradient.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_study1 as a1  # noqa: E402
from fba.metrics.leakage import summarize_diff  # noqa: E402
import yaml  # noqa: E402


def build():
    cfg = yaml.safe_load(open(ROOT / "configs/study1.yaml"))
    a1.MODELS = None
    df = a1.load(cfg)
    keep = df[df.condition_type.isin(["none", "main", "exog"])].copy()
    base = keep[keep.condition_type == "none"][["model", "instance_id", "ambiguity", "p", "width", "lo", "hi"]] \
        .rename(columns={"p": "p_none", "width": "slack", "lo": "lo_none", "hi": "hi_none"})
    for ind in ("imp", "exog", "exp"):
        g = keep[keep.indep == ind].pivot_table(index=["model", "instance_id", "ambiguity"],
                                                columns="buffer_level", values="p")
        if g.empty:
            continue
        g = g.join(base.set_index(["model", "instance_id", "ambiguity"]), how="inner")
        g["d"] = g["high"] - g["low"]
        g["ind"] = ind
        yield ind, g.reset_index()


def main():
    frames = {ind: g for ind, g in build()}
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    rows = []
    for ind, g in frames.items():
        for m, gm in g.groupby("model"):
            if len(gm) < 30:
                continue
            fit = smf.ols("d ~ slack", gm).fit()
            fit_amb = smf.ols("d ~ slack + C(ambiguity)", gm).fit()
            rows.append({"indep": ind, "model": m, "n": len(gm), "slope": fit.params["slack"],
                         "se": fit.bse["slack"], "p": fit.pvalues["slack"], "r2": fit.rsquared,
                         "slope_ctrl_amb": fit_amb.params["slack"], "p_ctrl_amb": fit_amb.pvalues["slack"]})
    res = pd.DataFrame(rows)
    out = ROOT / "results/statistics/slack"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "slack_slopes.csv", index=False)
    print("=== BBL ~ interval width under no buffer (slope in p per unit width) ===")
    print(res[res.indep != "exp"].to_string(index=False))
    pooled = pd.concat([g.assign(ind=ind) for ind, g in frames.items() if ind == "exog"])
    fit = smf.ols("d ~ slack + C(model) + C(ambiguity)", pooled).fit()
    print(f"\nPooled (random allocation, model and evidence dummies): slope={fit.params['slack']:.3f} "
          f"(se {fit.bse['slack']:.3f}, p={fit.pvalues['slack']:.2g}), n={int(fit.nobs)}")
    print(f"mean slack by evidence: "
          f"{pooled.groupby('ambiguity').slack.mean().round(3).to_dict()}")
    # binned means for the figure
    bins = [0, .2, .3, .4, .5, .6, .7, 1]
    pooled["bin"] = pd.cut(pooled.slack, bins)
    bm = pooled.groupby(["ind", "bin"], observed=True).d.agg(["mean", "sem", "size"]).reset_index()
    bm.to_csv(out / "slack_binned.csv", index=False)
    print("\n=== Binned means (random allocation) ===")
    print(bm[bm["ind"] == "exog"].to_string(index=False))


if __name__ == "__main__":
    main()
