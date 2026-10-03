"""Check that contamination flips a decision mainly near the decision threshold.

For each model, item and safeguard we have the buffer-free belief p0 (stage 1), the decision taken from that
belief (two-stage arm), and the decision taken with the buffer in context (vanilla arm).  The threshold rule
predicts that the two decisions disagree mainly when |p0 - p*(B)| is small.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_gt as A  # noqa: E402

df = A.load()
p0 = df[df.arm == "belief_nobuf"].set_index(["model", "instance_id"]).success_probability
v = df[(df.arm == "vanilla") & df.deploy.notna()].set_index(["model", "instance_id", "safeguard"])
t = df[(df.arm == "twostage") & df.deploy.notna()].set_index(["model", "instance_id", "safeguard"])
j = pd.DataFrame({"van": v.deploy, "two": t.deploy, "pv": v.success_probability}).dropna(subset=["van", "two"]).reset_index()
j["p0"] = [p0.get((m, i), np.nan) for m, i in zip(j.model, j.instance_id)]
j["pstar"] = j.safeguard.map(lambda s: A.LOSS[s] / (1 + A.LOSS[s]))
j["dist"] = (j.p0 - j.pstar).abs()
j["delta"] = (j.pv - j.p0).abs()
j["disagree"] = (j.van != j.two).astype(float)
j["predicted"] = (j.dist <= j.delta).astype(float)       # Prop. 3 crossing condition
bins = [0, .05, .1, .2, .3, .5, 1]
j["bin"] = pd.cut(j.dist, bins, include_lowest=True)
pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
print("=== disagreement between buffer-in-context and separated decisions, by distance to threshold ===")
print(j.groupby("bin", observed=True).disagree.agg(["mean", "size"]).to_string())
print("\n=== by model ===")
print(j.pivot_table(index="model", columns="bin", values="disagree", observed=True).round(2).to_string())
# crossing condition: how well does |p0-p*| <= |delta| predict a disagreement?
tab = pd.crosstab(j.predicted, j.disagree, normalize="index")
print("\n=== disagreement rate when the crossing condition holds (1) or fails (0) ===")
print(pd.crosstab(j.predicted, j.disagree).to_string())
print(tab.round(3).to_string())
out = {"dis_near": float(j[j.dist <= .1].disagree.mean()), "dis_far": float(j[j.dist > .3].disagree.mean()),
       "dis_cross": float(j[j.predicted == 1].disagree.mean()), "dis_nocross": float(j[j.predicted == 0].disagree.mean()),
       "n_cross": int((j.predicted == 1).sum()), "n_nocross": int((j.predicted == 0).sum())}
json.dump(out, open(ROOT / "results/statistics/groundtruth/threshold.json", "w"), indent=1)
print(out)
