"""Representation identification: which part of 'a number is present' removes the effect?

Conditions (all with the same qualitative cues and the same buffer manipulation):
  qual         no number
  equiv_frac   a verbalised fraction ("about two in three")
  irrelev_num  an irrelevant identifying number
  n1 / n2 / n4 one, two, four comparable cases
  pct          "a success rate of 71%"
  freq         "100 cases, 71 of which succeeded"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fba.metrics.leakage import summarize_diff  # noqa: E402

ST = ROOT / "results/statistics"
ORDER = ["qual", "equiv_frac", "irrelev_num", "n1", "n2", "n4", "pct", "freq"]
MARGIN = 0.02


def load():
    conds = pd.DataFrame([json.loads(l) for l in open(ROOT / "data/generated/representation/conditions.jsonl")])
    conds = conds.drop(columns=[c for c in ("system", "prompt", "meta") if c in conds.columns])
    rows = []
    for f in sorted((ROOT / "results/raw/representation").glob("*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r["status"] == "ok":
                rows.append({"condition_id": r["condition_id"], "model": f.stem, **r["parsed_response"]})
    return pd.DataFrame(rows).merge(conds, on="condition_id")


def main():
    df = load()
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    rows = []
    for (m, rep), g in df.groupby(["model", "rep"]):
        w = g.pivot_table(index="instance_id", columns="buffer_level", values="success_probability")
        rows.append({"model": m, "rep": rep, **summarize_diff(w["high"] - w["low"], "")})
    res = pd.DataFrame(rows)
    res["n_overlap"] = res.n
    pd.set_option("display.width", 220)
    print("=== buffer effect by numeric representation ===")
    print(res.pivot(index="model", columns="rep", values="mean")[ORDER].round(4).to_string())
    print("\n=== p-values ===")
    print(res.pivot(index="model", columns="rep", values="p_perm")[ORDER].round(4).to_string())
    print("\n=== equivalence to zero, margin %.2f (True = accept equivalence) ===" % MARGIN)
    res["equivalent"] = (res.ci_low > -MARGIN) & (res.ci_high < MARGIN)
    print(res.pivot(index="model", columns="rep", values="equivalent")[ORDER].to_string())
    res.to_csv(ST / "representation.csv", index=False)

    print("\n=== pooled over models: mean effect and 95% CI by representation ===")
    pool = []
    for rep in ORDER:
        g = res[res.rep == rep]
        d = df[(df.rep == rep)]
        w = d.pivot_table(index=["model", "instance_id"], columns="buffer_level", values="success_probability")
        pool.append({"rep": rep, **summarize_diff(w["high"] - w["low"], "")})
    p = pd.DataFrame(pool)
    p["equivalent"] = (p.ci_low > -MARGIN) & (p.ci_high < MARGIN)
    p["relative_to_qual"] = p["mean"] / p[p.rep == "qual"]["mean"].iloc[0]
    print(p.round(4).to_string(index=False))
    p.to_csv(ST / "representation_pooled.csv", index=False)

    # does the effect track the amount of information (n) or merely the presence of a number?
    print("\n=== presence vs amount: effect for 'a number, any number' vs 'more cases' ===")
    have = res[res.rep.isin(["n1", "n2", "n4", "pct", "freq"])]
    none = res[res.rep.isin(["qual", "equiv_frac", "irrelev_num"])]
    print(f"  with a computable number : mean {have['mean'].mean():+.4f}  (n={len(have)} cells)")
    print(f"  without one              : mean {none['mean'].mean():+.4f}  (n={len(none)} cells)")
    for rep in ["equiv_frac", "irrelev_num", "n1"]:
        sub = res[res.rep == rep]
        print(f"  {rep:12s}: {sub['mean'].mean():+.4f}, equivalent in {int(sub.equivalent.sum())}/{len(sub)} models")


if __name__ == "__main__":
    main()
