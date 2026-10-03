"""Analysis of the three round-2 studies: deployment v2 (two channels), mitigation arms,
and robustness of the manipulation surface."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fba.metrics.leakage import summarize_diff  # noqa: E402

NAME = {"gpt-4o-mini": "GPT-4o-mini", "qwen3.5-27b": "Qwen3.5-27B",
        "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash", "qwen-turbo": "Qwen-Turbo"}
MODELS3 = ["gpt-4o-mini", "qwen3.5-27b", "deepseek-v4.1-flash"]
ST = ROOT / "results/statistics"
pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)


def load(study):
    conds = pd.DataFrame([json.loads(l) for l in open(ROOT / "data/generated" / study / "conditions.jsonl")])
    conds = conds.drop(columns=[c for c in ("system", "prompt", "meta") if c in conds.columns])
    rows = []
    for f in sorted((ROOT / "results/raw" / study).glob("*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r["status"] != "ok":
                continue
            rows.append({"condition_id": r["condition_id"], "model": f.stem, **r["parsed_response"]})
    df = pd.DataFrame(rows).merge(conds, on="condition_id")
    if "deploy" in df.columns:
        df["deploy"] = df.deploy.astype("boolean")
    if "proceed" in df.columns:
        df["proceed"] = df.proceed.astype("boolean")
    return df


def deploy2():
    df = load("deploy2")
    print("\n" + "=" * 80 + "\nDEPLOYMENT v2\n" + "=" * 80)
    bel = df[(df.question_type == "belief") & df.success_probability.notna()].copy()
    dec = df[df.question_type == "deploy"].copy()
    given = df[df.question_type.str.startswith("deploy_given")].copy()
    given["given_p"] = given.question_type.str.replace("deploy_given", "").astype(float)
    order = ["none", "snapshot", "canary", "killswitch"]
    for c in (bel, dec, given):
        c["safeguard"] = pd.Categorical(c.safeguard, order, ordered=True)

    print("--- base rates by model (all safeguards pooled)")
    print(pd.DataFrame({
        "P(patch correct), mean": bel.groupby("model").success_probability.mean(),
        "deploy rate": dec.groupby("model").deploy.mean(),
        "n belief": bel.groupby("model").size(), "n deploy": dec.groupby("model").size(),
        "n given": given.groupby("model").size()}).round(3).to_string())

    print("\n--- P(patch correct) by safeguard")
    print(bel.groupby(["model", "safeguard"], observed=True).success_probability.mean().unstack().round(3).to_string())
    print("\n--- deploy rate by safeguard")
    print(dec.groupby(["model", "safeguard"], observed=True).deploy.mean().unstack().round(3).to_string())
    print("\n--- deploy rate when the probability is supplied from outside")
    print(given.groupby(["model", "given_p", "safeguard"], observed=True).deploy.mean().unstack().round(3).to_string())

    rows = []
    for m, g in bel.groupby("model"):
        w = g.pivot_table(index="instance_id", columns="safeguard", values="success_probability", observed=True)
        for s in order[1:]:
            rows.append({"model": NAME.get(m, m), "channel": "belief", "contrast": f"{s}-none",
                         **summarize_diff(w[s] - w["none"], "")})
    for m, g in dec.groupby("model"):
        w = g.pivot_table(index="instance_id", columns="safeguard", values="deploy", observed=True).astype(float)
        for s in order[1:]:
            rows.append({"model": NAME.get(m, m), "channel": "deploy (free belief)", "contrast": f"{s}-none",
                         **summarize_diff(w[s] - w["none"], "")})
    for m, g in given.groupby("model"):
        for gp, gg in g.groupby("given_p"):
            w = gg.pivot_table(index="instance_id", columns="safeguard", values="deploy", observed=True).astype(float)
            rows.append({"model": NAME.get(m, m), "channel": f"deploy (given p={gp:.2f})", "contrast": "killswitch-none",
                         **summarize_diff(w["killswitch"] - w["none"], "")})
    r = pd.DataFrame(rows)
    ST.mkdir(parents=True, exist_ok=True)
    r.to_csv(ST / "deploy2.csv", index=False)
    print("\n--- paired contrasts (kill switch vs none), by channel")
    print(r[r.contrast == "killswitch-none"][["model", "channel", "n", "mean", "ci_low", "ci_high", "p_perm"]]
          .round(3).to_string(index=False))
    # miscalibration against the test-based reference
    bel["err"] = bel.success_probability - bel.normative_probability
    print("\n--- belief error vs the test-pass reference, by CI strength")
    print(bel.groupby(["model", "pass_rate"], observed=True).err.mean().unstack().round(3).to_string())
    return r


def mitigation():
    df = load("mitigation")
    print("\n" + "=" * 80 + "\nMITIGATION ARMS\n" + "=" * 80)
    p = df[df.success_probability.notna()].copy()
    p["arm"] = p.arm
    print("--- stated probability by arm and buffer level")
    print(p.groupby(["model", "arm", "buffer_level"], observed=True).success_probability.mean()
          .unstack().round(3).to_string())
    rows = []
    for (m, arm), g in p.groupby(["model", "arm"]):
        w = g.pivot_table(index="instance_id", columns="buffer_level", values="success_probability")
        rows.append({"model": NAME.get(m, m), "arm": arm, **summarize_diff(w["high"] - w["low"], "")})
    r = pd.DataFrame(rows)
    r.to_csv(ST / "mitigation.csv", index=False)
    print("\n--- buffer effect on the stated probability, by arm")
    print(r[["model", "arm", "n", "mean", "ci_low", "ci_high", "p_perm"]].round(3).to_string(index=False))
    # decision side
    d = df[df.proceed.notna()].copy()
    print("\n--- proceed rate by arm and buffer")
    print(d.groupby(["model", "arm", "buffer_level"], observed=True).proceed.mean().unstack().round(3).to_string())
    dr = []
    for (m, arm), g in d.groupby(["model", "arm"]):
        w = g.pivot_table(index="instance_id", columns="buffer_level", values="proceed", observed=True).astype(float)
        dr.append({"model": NAME.get(m, m), "arm": arm, **summarize_diff(w["high"] - w["low"], "")})
    drr = pd.DataFrame(dr)
    drr.to_csv(ST / "mitigation_decision.csv", index=False)
    print("\n--- buffer effect on the proceed decision, by arm")
    print(drr[["model", "arm", "n", "mean", "ci_low", "ci_high", "p_perm"]].round(3).to_string(index=False))
    return r, drr


def robustness():
    df = load("robustness")
    print("\n" + "=" * 80 + "\nROBUSTNESS OF THE MANIPULATION SURFACE\n" + "=" * 80)
    main = df[df.style.isin(["para", "bullet"])].copy()
    rows = []
    for (m, order, style), g in main.groupby(["model", "order", "style"]):
        w = g.pivot_table(index="instance_id", columns="buffer_level", values="success_probability")
        rows.append({"model": NAME.get(m, m), "order": order, "style": style,
                     **summarize_diff(w["high"] - w["low"], "")})
    r = pd.DataFrame(rows)
    r.to_csv(ST / "robustness.csv", index=False)
    print("--- buffer effect by placement and formatting (qualitative evidence, random allocation)")
    for col, lab in (("style", "style"), ("order", "order")):
        print(f"\n  by {lab}:")
        print(r.pivot_table(index="model", columns=[col], values="mean").round(3).to_string())
        pv = r.pivot_table(index="model", columns=[col], values="p_perm")
        print("  p-values:"); print(pv.round(4).to_string())

    rw = df[df.style == "rewrite"].copy()
    if len(rw):
        rr = []
        for m, g in rw.groupby("model"):
            w = g.pivot_table(index="instance_id", columns="buffer_level", values="success_probability")
            rr.append({"model": NAME.get(m, m), **summarize_diff(w["high"] - w["low"], "")})
        rrw = pd.DataFrame(rr)
        rrw.to_csv(ST / "robustness_rewrite.csv", index=False)
        print("\n--- hand-written semantic rewrite of the buffer paragraph (3 domains)")
        print(rrw[["model", "n", "mean", "ci_low", "ci_high", "p_perm"]].round(3).to_string(index=False))
    return r


if __name__ == "__main__":
    deploy2()
    mitigation()
    robustness()
