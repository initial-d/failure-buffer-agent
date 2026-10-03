"""Mediation of the deploy decision (kill switch vs no safeguard) through the stated belief.

total     = E[deploy | kill] - E[deploy | none]                       (free-belief arm)
direct    = coefficient on kill in  deploy ~ kill + p                 (belief held fixed)
mediated  = total - direct;  share = mediated / total
Linear probability model, restricted to the two safeguards being compared; 95% intervals from a
bootstrap over scenarios (both safeguard cells of a scenario are resampled together).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAME = {"gpt-4o-mini": "GPT-4o-mini", "qwen3.5-27b": "Qwen3.5-27B", "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash"}

conds = pd.DataFrame([json.loads(l) for l in open(ROOT / "data/generated/deploy2/conditions.jsonl")])
recs = []
for f in sorted((ROOT / "results/raw/deploy2").glob("*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["status"] == "ok":
            recs.append({"condition_id": r["condition_id"], "model": f.stem, **r["parsed_response"]})
df = pd.DataFrame(recs).merge(conds.drop(columns=["system", "prompt", "meta"]), on="condition_id")
bel = df[df.question_type == "belief"][["model", "instance_id", "safeguard", "success_probability"]] \
    .rename(columns={"success_probability": "p"})
dec = df[df.question_type == "deploy"][["model", "instance_id", "safeguard", "deploy"]]
j = bel.merge(dec, on=["model", "instance_id", "safeguard"])
j = j[j.safeguard.isin(["none", "killswitch"])].copy()
j["deploy"] = j.deploy.astype(float)
j["kill"] = (j.safeguard == "killswitch").astype(float)


def decompose(g):
    X1 = np.c_[np.ones(len(g)), g.kill]
    X2 = np.c_[np.ones(len(g)), g.kill, g.p]
    y = g.deploy.values
    tot = np.linalg.lstsq(X1, y, rcond=None)[0][1]
    b = np.linalg.lstsq(X2, y, rcond=None)[0]
    return tot, b[1], tot - b[1], b[2]


rng = np.random.default_rng(0)
rows = []
for m, g in j.groupby("model"):
    tot, d, med, slope = decompose(g)
    ids = g.instance_id.unique()
    bs = []
    by = {i: gg for i, gg in g.groupby("instance_id")}
    for _ in range(2000):
        s = pd.concat([by[i] for i in rng.choice(ids, len(ids))])
        bs.append(decompose(s))
    bs = np.array(bs)
    ci = lambda k: (np.quantile(bs[:, k], .025), np.quantile(bs[:, k], .975))
    sh = bs[:, 2] / np.where(np.abs(bs[:, 0]) > 1e-9, bs[:, 0], np.nan)
    rows.append({"model": NAME[m], "n_scen": len(ids), "total": tot, "total_lo": ci(0)[0], "total_hi": ci(0)[1],
                 "direct": d, "direct_lo": ci(1)[0], "direct_hi": ci(1)[1],
                 "mediated": med, "med_lo": ci(2)[0], "med_hi": ci(2)[1],
                 "share": med / tot, "share_lo": np.nanquantile(sh, .025), "share_hi": np.nanquantile(sh, .975),
                 "slope_p": slope,
                 "belief_shift": g[g.kill == 1].p.mean() - g[g.kill == 0].p.mean()})
o = pd.DataFrame(rows)
o.to_csv(ROOT / "results/statistics/channels.csv", index=False)
pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
print(o.to_string(index=False))
