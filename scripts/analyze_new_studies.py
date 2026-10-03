"""Analysis of the four new studies: evidence-sample gradient, deliberation budget,
the deployment agent, and the fresh confirmatory replication."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fba.metrics.leakage import add_cond_key, summarize_diff, wide_main  # noqa: E402
import analyze_study1 as a1  # noqa: E402
import yaml  # noqa: E402

NAME = {"gpt-4o-mini": "GPT-4o-mini", "qwen3.5-27b": "Qwen3.5-27B", "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash",
        "qwen-turbo": "Qwen-Turbo", "gemini-2.5-flash": "Gemini-2.5-Flash", "glm-5.3": "GLM-5.3",
        "kimi-k3": "Kimi-K3", "gpt-5.6-terra": "GPT-5.6-Terra", "claude-sonnet-5": "Claude Sonnet 5"}
OUT = ROOT / "results/statistics"
pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)


def load_study(study, value="p"):
    conds = pd.DataFrame([json.loads(l) for l in open(ROOT / "data/generated" / study / "conditions.jsonl")])
    conds = conds.drop(columns=[c for c in ("slots", "system", "prompt", "meta", "sequence") if c in conds.columns])
    frames = []
    for f in sorted((ROOT / "results/raw" / study).glob("*.jsonl")):
        ok = {}
        for l in open(f):
            r = json.loads(l)
            if r["status"] == "ok":
                ok[r["condition_id"]] = r["parsed_response"]
        frames.append(pd.DataFrame([{"condition_id": k, "model": f.stem,
                                     "p": v.get("success_probability"),
                                     "deploy": v.get("deploy")} for k, v in ok.items()]))
    df = pd.concat(frames).merge(conds, on="condition_id")
    df["mid"] = df.p + 0.0
    return df


def gradient():
    df = load_study("gradient")
    print("\n" + "=" * 78 + "\nSTUDY A: evidence-sample gradient\n" + "=" * 78)
    rows = []
    for (m, n), g in df.groupby(["model", "n_cases"]):
        w = g.pivot_table(index="instance_id", columns="buffer_level", values="p")
        rows.append({"model": m, "n_cases": n, **summarize_diff(w["high"] - w["low"], "BBL")})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "gradient.csv", index=False)
    print(res.pivot(index="model", columns="n_cases", values="mean").round(3).to_string())
    print("\n  significance (p_perm<.05 shown as *):")
    print(res.assign(sig=np.where(res.p_perm < .05, "*", "")).pivot(index="model", columns="n_cases",
                                                                    values="sig").fillna("").to_string())

    # decay: does the effect shrink with log(n)? and by how much per doubling
    print("\n=== Slope of BBL on log2(n) and the n at which the effect is halved ===")
    out = []
    for m, g in res.groupby("model"):
        g = g[g.n_cases > 0]                     # n=0 has no count; handled separately
        fit = smf.ols("mean ~ np.log2(n_cases)", g).fit()
        base = res[(res.model == m) & (res.n_cases == 2)]["mean"].iloc[0]
        q = res[(res.model == m) & (res.n_cases == 0)]["mean"].iloc[0]
        out.append({"model": NAME.get(m, m), "slope_per_doubling": fit.params["np.log2(n_cases)"],
                    "p": fit.pvalues["np.log2(n_cases)"], "bbl_n2": base, "bbl_qualitative(n=0)": q,
                    "ratio_qual_over_n2": q / base if base else np.nan})
    print(pd.DataFrame(out).to_string(index=False))
    # normative deviation by n
    nm = df[df.normative_probability.notna()].copy()
    nm["err"] = nm.p - nm.normative_probability
    print("\n=== deviation from Laplace posterior mean, by n (mean over models) ===")
    print(nm.groupby("n_cases").err.agg(["mean", "std"]).round(3).to_string())
    return res


def budget():
    df = load_study("budget")
    print("\n" + "=" * 78 + "\nSTUDY B: deliberation budget\n" + "=" * 78)
    order = ["direct", "b32", "b128", "b512"]
    rows = []
    for (m, amb, b), g in df.groupby(["model", "ambiguity", "budget"]):
        w = g.pivot_table(index="instance_id", columns="buffer_level", values="p")
        rows.append({"model": m, "ambiguity": amb, "budget": b, **summarize_diff(w["high"] - w["low"], "BBL")})
    res = pd.DataFrame(rows)
    res["budget"] = pd.Categorical(res.budget, order, ordered=True)
    res.to_csv(OUT / "budget.csv", index=False)
    for amb in ("sparse", "qualitative"):
        print(f"\n--- {amb}: BBL by budget")
        t = res[res.ambiguity == amb].pivot(index="model", columns="budget", values="mean")
        p = res[res.ambiguity == amb].pivot(index="model", columns="budget", values="p_perm")
        print(t.round(3).to_string())
        print("   p-values:"); print(p.round(4).to_string())
    print("\n=== Within model, BBL direct vs 128 (paired over scenarios) ===")
    out = []
    for (m, amb), g in df.groupby(["model", "ambiguity"]):
        w = g.assign(v=g.p).pivot_table(index="instance_id", columns=["budget", "buffer_level"], values="v")
        for a, b in [("direct", "b32"), ("direct", "b128"), ("direct", "b512"), ("b32", "b512")]:
            d = (w[(b, "high")] - w[(b, "low")]) - (w[(a, "high")] - w[(a, "low")])
            s = summarize_diff(d, f"{a}->{b}")
            out.append({"model": NAME.get(m, m), "ambiguity": amb, "contrast": f"{a}->{b}", **s})
    o = pd.DataFrame(out)
    o.to_csv(OUT / "budget_paired.csv", index=False)
    print(o[["model", "ambiguity", "contrast", "mean", "ci_low", "ci_high", "p_perm"]].round(3).to_string(index=False))
    return res


def deploy():
    df = load_study("deploy")
    print("\n" + "=" * 78 + "\nSTUDY C: deployment agent\n" + "=" * 78)
    bel = df[(df.question_type == "belief") & df.p.notna()].copy()
    dec = df[(df.question_type == "deploy") & df.deploy.notna()].copy()
    # deploy answers are stored as strings by the parser; coerce
    dec["deploy"] = dec.deploy.astype(str).str.lower().isin(["true", "yes"])
    order = ["none", "snapshot", "canary", "killswitch"]
    bel["safeguard"] = pd.Categorical(bel.safeguard, order, ordered=True)
    dec["safeguard"] = pd.Categorical(dec.safeguard, order, ordered=True)
    b = bel.groupby(["model", "safeguard"], observed=True).p.agg(["mean", "size"]).reset_index()
    d = dec.groupby(["model", "safeguard"], observed=True).deploy.agg(["mean", "size"]).reset_index()
    print("=== P(patch correct) by safeguard ===")
    print(b.pivot(index="model", columns="safeguard", values="mean").round(3).to_string())
    print("\n=== deploy rate by safeguard ===")
    print(d.pivot(index="model", columns="safeguard", values="mean").round(3).to_string())
    print("\n=== paired contrasts: best safeguard minus none ===")
    rows = []
    for m, g in bel.groupby("model"):
        w = g.pivot_table(index="instance_id", columns="safeguard", values="p")
        for s in order[1:]:
            rows.append({"model": NAME.get(m, m), "question": "belief", "contrast": f"{s}-none",
                         **summarize_diff(w[s] - w["none"], "")})
    for m, g in dec.groupby("model"):
        w = g.pivot_table(index="instance_id", columns="safeguard", values="deploy")
        for s in order[1:]:
            rows.append({"model": NAME.get(m, m), "question": "deploy", "contrast": f"{s}-none",
                         **summarize_diff(w[s].astype(float) - w["none"].astype(float), "")})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "deploy.csv", index=False)
    print(r[["model", "question", "contrast", "mean", "ci_low", "ci_high", "p_perm"]].round(3).to_string(index=False))
    # belief error against the normative base rate
    bel["err"] = bel.p - bel.normative_probability
    print("\n=== belief error vs the normative test-failure rate ===")
    print(bel.groupby(["model", "safeguard"], observed=True).err.mean().unstack().round(3).to_string())
    # does the deploy decision track the stated belief?
    j = bel[["model", "instance_id", "safeguard", "p"]].merge(
        dec[["model", "instance_id", "safeguard", "deploy"]], on=["model", "instance_id", "safeguard"])
    print("\n=== deploy rate by stated-belief bucket ===")
    j["bucket"] = pd.cut(j.p, [0, .3, .5, .7, .9, 1.001])
    print(j.pivot_table(index=["model", "bucket"], values="deploy", observed=True).astype(float).round(2).to_string())
    return r


def confirmatory():
    print("\n" + "=" * 78 + "\nFRESH CONFIRMATORY SET (new scenarios, pre-specified cells)\n" + "=" * 78)
    study = "confirmatory"
    conds = pd.DataFrame([json.loads(l) for l in open(ROOT / "data/generated" / study / "conditions.jsonl")])
    conds = conds.drop(columns=["slots", "system", "prompt"])
    frames = []
    for f in sorted((ROOT / "results/raw" / study).glob("*.jsonl")):
        ok = {}
        for l in open(f):
            r = json.loads(l)
            if r["status"] == "ok":
                ok[r["condition_id"]] = r["parsed_response"]
        frames.append(pd.DataFrame([{"condition_id": k, "model": f.stem,
                                     "p": v["success_probability"], "width": v["upper_bound"] - v["lower_bound"]}
                                    for k, v in ok.items()]))
    df = add_cond_key(pd.concat(frames).merge(conds, on="condition_id"))
    rows = []
    for (m, amb), g in df[df.condition_type == "exog"].groupby(["model", "ambiguity"]):
        w = g.pivot_table(index="instance_id", columns="buffer_level", values="p")
        s = summarize_diff(w["high"] - w["low"], "BBL (random allocation)")
        ww = g.pivot_table(index="instance_id", columns="buffer_level", values="width")
        rows.append({"model": NAME.get(m, m), "ambiguity": amb, **s,
                     "width_change": (ww["high"] - ww["low"]).mean()})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "confirmatory.csv", index=False)
    print(r[["model", "ambiguity", "n", "mean", "ci_low", "ci_high", "p_perm", "width_change"]].round(4).to_string(index=False))
    print("\n=== baseline (no buffer) probability, for reference ===")
    base = df[df.condition_type == "none"].groupby(["model", "ambiguity"]).p.mean().unstack()
    print(base.round(3).to_string())
    # exploration vs confirmation comparison, same models
    old = a1.load(yaml.safe_load(open(ROOT / "configs/study1.yaml")))
    print("\n=== exploration (Study 1) vs confirmation, random allocation, qualitative ===")
    cmp_rows = []
    for m in ["gpt-4o-mini", "qwen3.5-27b", "deepseek-v4.1-flash"]:
        o = old[(old.model == m) & (old.ambiguous if False else True)]
        ow = o[o.condition_type.isin(["main", "exog"])]
        if ow.empty:
            continue
        ow = ow[(ow.ambiguity == "qualitative") & (ow.indep == "exog")]
        wid = ow.pivot_table(index="instance_id", columns="buffer_level", values="p")
        so = summarize_diff(wid["high"] - wid["low"], "")
        nc = r[(r.model == NAME[m]) & (r.ambiguity == "qualitative")]
        if nc.empty:
            continue
        cmp_rows.append({"model": NAME[m], "exploration": so["mean"], "exploration_n": so["n"],
                         "confirmatory": nc["mean"].iloc[0], "confirmatory_n": nc["n"].iloc[0],
                         "difference": nc["mean"].iloc[0] - so["mean"]})
    print(pd.DataFrame(cmp_rows).round(4).to_string(index=False))
    return r


if __name__ == "__main__":
    gradient()
    budget()
    deploy()
    confirmatory()
