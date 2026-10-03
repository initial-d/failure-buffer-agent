"""Within-type checks for the slack analysis (no API calls)."""
import json
import sys
from pathlib import Path

import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import analyze_slack as s  # noqa: E402

fr = {i: g for i, g in s.build()}
inst = {}
for l in open(ROOT / "data/generated/study1_pilot/instances.jsonl"):
    r = json.loads(l)
    inst[r["instance_id"]] = r["evidence_stats"]["qualitative"]
for ind in ["imp", "exog"]:
    g = fr[ind]
    print(f"\n=== {ind}: mean slack and mean BBL by evidence type (pooled over models) ===")
    print(g.groupby("ambiguity").agg(slack=("slack", "mean"), bbl=("d", "mean"), n=("d", "size")).round(3).to_string())
    for amb in ["qualitative", "sparse"]:
        q = g[g.ambiguity == amb]
        q = q.dropna(subset=["d", "slack"]).reset_index(drop=True)
        m = smf.mixedlm("d ~ slack", q, groups=q["model"]).fit()
        print(f"within {amb:12s} slope on own width: {m.params['slack']:+.3f} (p={m.pvalues['slack']:.2g})")
    q = g[g.ambiguity == "qualitative"].copy()
    q["n_pos"] = q.instance_id.map(lambda i: inst[i]["n_pos"])
    print(f"--- qualitative, {ind}: BBL by number of favourable cues (of 5)")
    print(q.groupby("n_pos").agg(bbl=("d", "mean"), p_none=("p_none", "mean"), n=("d", "size")).round(3).to_string())
