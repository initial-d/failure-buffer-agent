"""Offline analysis of the completed round-4 runs plus a backend-heterogeneity check on Study 1."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fba.metrics.leakage import summarize_diff  # noqa: E402
ST = ROOT / "results/statistics"; M = 0.02
pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)


def load(study, models):
    conds = {json.loads(l)["condition_id"]: json.loads(l) for l in open(ROOT / f"data/generated/{study}/conditions.jsonl")}
    rows = []
    for m in models:
        for l in open(ROOT / f"results/raw/{study}/{m}.jsonl"):
            r = json.loads(l)
            if r["status"] == "ok":
                c = conds[r["condition_id"]]
                rows.append({"model": m, "condition_id": r["condition_id"], "instance_id": c["instance_id"],
                             "buffer_level": c.get("buffer_level"), "known_p": c.get("known_p"),
                             "route": r.get("model_version"), **r["parsed_response"]})
    return pd.DataFrame(rows)


out = {}
# ---- known probability, accuracy-scored Yes/No
kp = load("known_p", ["gpt-4o-mini", "qwen3.5-27b", "deepseek-v4.1-flash", "claude-sonnet-5"])
kp["yes"] = (kp.answer == "yes").astype(float)
kp["correct"] = (kp.yes == (kp.known_p > 0.5)).astype(float)
print("=== known probability: accuracy by buffer ===")
print(kp.pivot_table(index="model", columns="buffer_level", values="correct").round(3).to_string())
rows = []
for m, g in kp.groupby("model"):
    w = g.pivot_table(index="instance_id", columns="buffer_level", values="yes")
    s = summarize_diff(w["high"] - w["low"], "")
    rows.append({"model": m, "net_yes_HL": s["mean"], "p": s["p_perm"], "flip": float(((w["high"] - w["low"]) != 0).mean()),
                 "acc_none": g[g.buffer_level == "none"].correct.mean(), "acc_low": g[g.buffer_level == "low"].correct.mean(),
                 "acc_high": g[g.buffer_level == "high"].correct.mean()})
kpr = pd.DataFrame(rows); kpr.to_csv(ST / "known_p.csv", index=False); print(kpr.to_string(index=False))

# ---- strict event wording
st = load("strict", ["gpt-4o-mini", "qwen3.5-27b", "deepseek-v4.1-flash"])
s1 = pd.read_csv(ST / "study1_pilot/bbl.csv")
base = s1[(s1.value == "p") & (s1.metric == "H-L") & (s1.ambiguity == "qualitative") & (s1.indep == "exog")].set_index("model")["mean"]
rows = []
for m, g in st.groupby("model"):
    w = g.pivot_table(index="instance_id", columns="buffer_level", values="success_probability")
    s = summarize_diff(w["high"] - w["low"], "")
    rows.append({"model": m, "strict": s["mean"], "lo": s["ci_low"], "hi": s["ci_high"], "p": s["p_perm"], "original": base[m]})
str_ = pd.DataFrame(rows); str_.to_csv(ST / "strict.csv", index=False)
print("\n=== strict event wording vs original (qualitative, random allocation) ==="); print(str_.to_string(index=False))

# ---- repeated independent calls (route study): same prompts, fresh calls
rt = load("route", ["deepseek-v4.1-flash", "gpt-4o-mini"])
rows = []
for m, g in rt.groupby("model"):
    gg = g[g.buffer_level.isin(["low", "high"])]
    w = gg.pivot_table(index="instance_id", columns="buffer_level", values="success_probability")
    s = summarize_diff(w["high"] - w["low"], "")
    orig = {json.loads(l)["condition_id"]: json.loads(l)["parsed_response"]["success_probability"]
            for l in open(ROOT / f"results/raw/study1_pilot/{m}.jsonl") if json.loads(l)["status"] == "ok"}
    g = g.assign(orig=g.condition_id.map(orig))
    rows.append({"model": m, "rerun": s["mean"], "lo": s["ci_low"], "hi": s["ci_high"], "p": s["p_perm"],
                 "original": base[m], "identical_share": float((g.success_probability == g.orig).mean()),
                 "mean_abs_diff": float((g.success_probability - g.orig).abs().mean()),
                 "n_routes": g.route.nunique()})
rr = pd.DataFrame(rows); rr.to_csv(ST / "rerun.csv", index=False)
print("\n=== independent re-run of the main contrast ==="); print(rr.to_string(index=False))

# ---- channel heterogeneity inside Study 1 (models whose requests were served by several backends)
print("\n=== backend heterogeneity within Study 1 (qualitative, random allocation) ===")
conds = {json.loads(l)["condition_id"]: json.loads(l) for l in open(ROOT / "data/generated/study1_pilot/conditions.jsonl")}
rows = []
for m in ["deepseek-v4.1-flash", "glm-5.3", "kimi-k3", "gpt-5.6-terra"]:
    recs = []
    for l in open(ROOT / f"results/raw/study1_pilot/{m}.jsonl"):
        r = json.loads(l)
        c = conds.get(r["condition_id"])
        if r["status"] == "ok" and c and c["condition_type"] == "exog" and c["ambiguity"] == "qualitative" and c["buffer_level"] in ("low", "high"):
            recs.append({"iid": c["instance_id"], "b": c["buffer_level"], "p": r["parsed_response"]["success_probability"], "route": r["model_version"]})
    d = pd.DataFrame(recs).drop_duplicates(["iid", "b"], keep="last")
    d["high"] = (d.b == "high").astype(float)
    import statsmodels.formula.api as smf
    full = smf.ols("p ~ high * C(route) + C(iid)", d).fit()
    red = smf.ols("p ~ high + C(route) + C(iid)", d).fit()
    from statsmodels.stats.anova import anova_lm
    a = anova_lm(red, full)
    rows.append({"model": m, "n_backends": d.route.nunique(), "n": len(d), "p_interaction": float(a["Pr(>F)"].iloc[1])})
ch = pd.DataFrame(rows); ch.to_csv(ST / "backend_heterogeneity.csv", index=False); print(ch.to_string(index=False))
