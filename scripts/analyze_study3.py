import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.metrics.leakage import summarize_diff  # noqa: E402

EPS = 0.005


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load(cfg, models=None) -> pd.DataFrame:
    conds = pd.DataFrame([json.loads(l) for l in open(ROOT / "data/generated" / cfg["study"] / "conditions.jsonl")])
    conds = conds.drop(columns=["system", "prompt", "meta", "sequence"])
    frames = []
    for f in sorted((ROOT / "results/raw" / cfg["study"]).glob("*.jsonl")):
        if models and f.stem not in models:
            continue
        ok = {}
        for l in open(f):
            r = json.loads(l)
            if r["status"] == "ok":
                ok[r["condition_id"]] = r["parsed_response"]
        frames.append(pd.DataFrame([{"condition_id": k, "model": f.stem, "p": v["success_probability"],
                                     "width": v["upper_bound"] - v["lower_bound"]} for k, v in ok.items()]))
    df = pd.concat(frames).merge(conds, on="condition_id")
    df["feedback"] = df.feedback.fillna("prior")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/study3.yaml"))
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    df = load(cfg, args.models)
    pd.set_option("display.width", 220, "display.max_rows", 500, "display.float_format", "{:.4f}".format)
    print(df.groupby("model").size())

    idx = ["model", "domain", "instance_id", "ambiguity", "buffer_level"]
    prior = df[df.feedback == "prior"][idx + ["p", "width", "normative_probability"]].rename(
        columns={"p": "q0", "width": "w0", "normative_probability": "n0"})
    post = df[df.feedback != "prior"].merge(prior, on=idx)
    post["upd"] = post.p - post.q0                      # signed update (neg feedback -> negative)
    post["upd_logit"] = logit(post.p) - logit(post.q0)
    post["norm_upd_logit"] = logit(post.normative_probability) - logit(post.n0)
    post["resp"] = post.upd_logit / post.norm_upd_logit  # responsiveness (sparse only)
    post.to_csv(ROOT / "results/processed" / f"{cfg['study']}.csv", index=False)

    rows = []
    keys = ["model", "ambiguity", "feedback"]
    # mean levels
    lev = post.groupby(keys + ["buffer_level"]).agg(q0=("q0", "mean"), q1=("p", "mean"), upd=("upd", "mean"),
                                                    upd_logit=("upd_logit", "mean"), resp=("resp", "mean")).reset_index()
    print("\n=== Mean prior, posterior, update by buffer ===")
    print(lev.to_string(index=False))

    for k, g in post.groupby(keys):
        w = g.pivot_table(index="instance_id", columns="buffer_level", values=["q0", "p", "upd", "upd_logit", "width"])
        sign = -1 if k[2].startswith("neg") else 1
        base = dict(zip(keys, k))
        rows.append({**base, **summarize_diff(w["q0"]["high"] - w["q0"]["low"], "prior H-L")})
        rows.append({**base, **summarize_diff(w["p"]["high"] - w["p"]["low"], "posterior H-L")})
        # update magnitude in the normative direction; BFS>0 => high buffer updates LESS
        rows.append({**base, **summarize_diff(sign * (w["upd"]["low"] - w["upd"]["high"]), "BFS prob (L-H)")})
        rows.append({**base, **summarize_diff(sign * (w["upd_logit"]["low"] - w["upd_logit"]["high"]), "BFS logit (L-H)")})
        rows.append({**base, **summarize_diff(sign * (w["upd_logit"]["none"] - w["upd_logit"]["high"]), "BFS logit (none-H)")})
        rows.append({**base, **summarize_diff(sign * (w["upd_logit"]["none"] - w["upd_logit"]["low"]), "BFS logit (none-L)")})
    res = pd.DataFrame(rows)
    out = ROOT / "results/statistics" / cfg["study"]
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "bfs.csv", index=False)
    print("\n=== Paired contrasts ===")
    print(res.drop(columns=["mean_abs"]).to_string(index=False))


if __name__ == "__main__":
    main()
