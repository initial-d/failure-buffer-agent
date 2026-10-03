"""Implicit belief: log-odds of Yes vs No at the first answer token; paired high-low contrasts."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.metrics.leakage import add_cond_key, summarize_diff  # noqa: E402


def yes_no_logodds(lp):
    top = lp[0]["top_logprobs"]
    acc = {"yes": [], "no": []}
    for t in top:
        k = t["token"].strip().lower()
        if k in acc:
            acc[k].append(t["logprob"])
    floor = min(t["logprob"] for t in top)        # censoring bound if a side is outside top-k
    ly = np.logaddexp.reduce(acc["yes"]) if acc["yes"] else floor
    ln = np.logaddexp.reduce(acc["no"]) if acc["no"] else floor
    return ly - ln, not (acc["yes"] and acc["no"])


def main():
    conds = pd.DataFrame([json.loads(l) for l in open(ROOT / "data/generated/study1_pilot/conditions.jsonl")])
    conds = conds.drop(columns=["slots", "system", "prompt"])
    frames = []
    for f in sorted((ROOT / "results/raw/study1_logprob").glob("*.jsonl")):
        rows = {}
        for l in open(f):
            r = json.loads(l)
            if r["status"] == "ok" and r.get("logprobs"):
                lo, cens = yes_no_logodds(r["logprobs"])
                rows[r["condition_id"]] = {"condition_id": r["condition_id"], "model": f.stem, "lo": lo,
                                           "censored": cens, "yes": r["parsed_response"]["answer"] == "yes"}
        frames.append(pd.DataFrame(rows.values()))
    df = add_cond_key(pd.concat(frames).merge(conds, on="condition_id"))
    df["p"] = 1 / (1 + np.exp(-df.lo))
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    print(df.groupby("model").agg(n=("lo", "size"), censored=("censored", "mean"), yes_rate=("yes", "mean"),
                                  mean_logodds=("lo", "mean")))
    rows = []
    main_df = df[df.condition_type.isin(["main", "exog"])]
    for (m, a, i), g in main_df.groupby(["model", "ambiguity", "indep"]):
        for val, lab in [("lo", "logodds H-L"), ("p", "P(yes) H-L"), ("yes", "answer flips H-L")]:
            w = g.assign(v=g[val].astype(float)).pivot_table(index="instance_id", columns="buffer_level", values="v")
            rows.append({"model": m, "ambiguity": a, "indep": i, **summarize_diff(w["high"] - w["low"], lab)})
    res = pd.DataFrame(rows)
    out = ROOT / "results/statistics/study1_logprob"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "logprob_bbl.csv", index=False)
    res["cell"] = res.apply(
        lambda r: f"{r['mean']:+.3f} [{r.ci_low:+.3f},{r.ci_high:+.3f}]{'*' if r.p_perm < .05 else ' '}", axis=1)
    for lab in res.metric.unique():
        print(f"\n=== {lab} ===")
        print(res[res.metric == lab].pivot(index=["model", "ambiguity"], columns="indep", values="cell").to_string())
    # controls on log-odds
    rows = []
    for (m, a), g in df[df.ambiguity.isin(["sparse", "qualitative"])].groupby(["model", "ambiguity"]):
        piv = g.pivot_table(index="instance_id", columns="cond_key", values="lo")
        nz = [c for c in piv.columns if c.startswith("noise")]
        lmh = ["main_low_imp", "main_medium_imp", "main_high_imp"]
        rows.append({"model": m, "ambiguity": a,
                     "noise_range": (piv[nz].max(1) - piv[nz].min(1)).mean(),
                     "buffer_range_imp": (piv[lmh].max(1) - piv[lmh].min(1)).mean(),
                     "poscontrol-none": (piv["positive_control"] - piv["none"]).mean(),
                     "negbuf-low": (piv["negative_buffer"] - piv["main_low_imp"]).mean(),
                     "negbuf-high": (piv["negative_buffer"] - piv["main_high_imp"]).mean()})
    print("\n=== Controls (log-odds) ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
