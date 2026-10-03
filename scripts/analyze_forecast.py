"""Analysis of the clean categorical-forecast test (accuracy-only scoring)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.metrics.leakage import summarize_diff  # noqa: E402


def main():
    inst = {}
    for l in open(ROOT / "data/generated/study1_pilot/instances.jsonl"):
        r = json.loads(l)
        inst[r["instance_id"]] = r["evidence_stats"]["precise"]["k"]
    frames = []
    for f in sorted((ROOT / "results/raw/forecast_clean").glob("*.jsonl")):
        rows = {}
        for l in open(f):
            r = json.loads(l)
            if r["status"] == "ok":
                iid, amb, tag = r["condition_id"].split("__")
                rows[r["condition_id"]] = {"model": f.stem, "instance_id": iid, "ambiguity": amb, "tag": tag,
                                           "yes": float(r["parsed_response"]["answer"] == "yes")}
        frames.append(pd.DataFrame(rows.values()))
    df = pd.concat(frames)
    df["k"] = df.instance_id.map(inst)
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    print(df.groupby("model").size().to_string())

    print("\n=== Yes rate by condition ===")
    print(df.pivot_table(index=["ambiguity", "model"], columns="tag", values="yes", aggfunc="mean")
          [["none", "noise0", "noise1", "low_imp", "high_imp", "low_exog", "high_exog", "low_exp", "high_exp"]].to_string())

    rows = []
    for (m, a), g in df.groupby(["model", "ambiguity"]):
        w = g.pivot_table(index="instance_id", columns="tag", values="yes")
        for ind in ("imp", "exog", "exp"):
            d = w[f"high_{ind}"] - w[f"low_{ind}"]
            s = summarize_diff(d, f"{ind}: Yes(H)-Yes(L)")
            rows.append({"model": m, "ambiguity": a, "contrast": ind, "net": s["mean"], "ci_low": s["ci_low"],
                         "ci_high": s["ci_high"], "p": s["p_perm"], "flip_any": float((d.dropna() != 0).mean())})
        d = w["noise0"] - w["noise1"]
        rows.append({"model": m, "ambiguity": a, "contrast": "noise", "net": d.mean(), "ci_low": np.nan,
                     "ci_high": np.nan, "p": np.nan, "flip_any": float((d.dropna() != 0).mean())})
    res = pd.DataFrame(rows)
    out = ROOT / "results/statistics/forecast_clean"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "forecast_flips.csv", index=False)
    res["cell"] = res.apply(lambda r: f"{r.net:+.2f}{'*' if r.p < .05 else ' '} (flip {r.flip_any:.2f})", axis=1)
    print("\n=== Net Yes(high)-Yes(low) and share of instances whose answer changes ===")
    print(res.pivot(index=["ambiguity", "model"], columns="contrast", values="cell")[["noise", "imp", "exog", "exp"]].to_string())

    # Precise evidence has a known correct answer: Yes iff k > 50.
    pr = df[(df.ambiguity == "precise") & (df.k != 50)].copy()
    pr["correct"] = (pr.yes == (pr.k > 50).astype(float)).astype(float)
    print("\n=== Precise evidence: accuracy vs base-rate-optimal answer (Yes iff k>50) ===")
    print(pr.pivot_table(index="model", columns="tag", values="correct")
          [["none", "low_imp", "high_imp", "low_exog", "high_exog", "low_exp", "high_exp"]].to_string())
    pr["side"] = np.where(pr.k > 50, "k>50 (Yes correct)", "k<50 (No correct)")
    print("\n=== Precise evidence: Yes rate split by which answer is correct ===")
    print(pr[pr.tag.isin(["none", "low_exog", "high_exog", "low_exp", "high_exp"])]
          .pivot_table(index=["model", "side"], columns="tag", values="yes").to_string())


if __name__ == "__main__":
    main()
