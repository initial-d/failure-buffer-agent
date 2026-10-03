"""Clean categorical-forecast test.

Same scenarios as Study 1, but the model gives a Yes/No forecast under an explicitly
accuracy-only scoring rule that is declared to be unrelated to any consequence or decision.
Under that rule the optimal answer is Yes iff P(event) > 0.5, so it cannot depend on the buffer.
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.client import ChatClient, JsonlCache, run_jobs  # noqa: E402
from fba.tasks.generator import prompt_hash  # noqa: E402

SYSTEM = "You are a careful forecaster. Answer with exactly one word."
SCORING = ("This is a pure forecasting exercise. Your forecast is scored only on accuracy: 1 point if it matches "
           "what actually happens, 0 points otherwise. It will not be shown to anyone involved, will not be used "
           "for any decision, and is not advice about whether to go ahead.")
AMBIGUITY = ["precise", "qualitative"]


def build_jobs():
    conds = [json.loads(l) for l in open(ROOT / "data/generated/study1_pilot/conditions.jsonl")]
    by_inst = {}
    for c in conds:
        by_inst.setdefault(c["instance_id"], []).append(c)
    jobs = []
    for iid, cs in by_inst.items():
        get = lambda **kw: next(c for c in cs if all(c[k] == v for k, v in kw.items()))
        exog_text = get(condition_type="exog", buffer_level="low", ambiguity="sparse")["slots"]["independence"]
        noise = [c["slots"]["buffer"] for c in cs if c["condition_type"] == "noise" and c["ambiguity"] == "sparse"][:2]
        for amb in AMBIGUITY:
            base = get(condition_type="none", ambiguity=amb)["slots"]
            q = base["question"]
            event = q[len("What is the probability that "):-1]
            exp_text = get(condition_type="main", ambiguity=amb, buffer_level="low",
                           explicit_independence=True)["slots"]["independence"]
            variants = [("none", None, "")] + [(f"noise{i}", n, "") for i, n in enumerate(noise)]
            for lvl in ("low", "high"):
                b = get(condition_type="main", ambiguity=amb, buffer_level=lvl,
                        explicit_independence=False)["slots"]["buffer"]
                variants += [(f"{lvl}_imp", b, ""), (f"{lvl}_exog", b, exog_text), (f"{lvl}_exp", b, exp_text)]
            for tag, buf, indep in variants:
                parts = [base["context"], base["evidence"]] + [x for x in (buf, indep) if x]
                parts += [SCORING, f"Forecast: will it turn out that {event}?",
                          "Answer with exactly one word: Yes or No."]
                p = "\n\n".join(parts)
                jobs.append({"condition_id": f"{iid}__{amb}__{tag}", "instance_id": iid, "system": SYSTEM,
                             "prompt": p, "prompt_hash": prompt_hash(SYSTEM, p), "sample_idx": 0})
    return jobs


def parse_yes_no(text):
    t = text.strip().lower().strip(".*\"' \n")
    if t not in ("yes", "no"):
        raise ValueError(f"not yes/no: {text[:40]!r}")
    return {"answer": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    mcfg = yaml.safe_load(open(ROOT / "configs/models.yaml"))
    jobs = build_jobs()[: args.limit]
    for mkey in args.models:
        m = mcfg["models"][mkey]
        client = ChatClient(m["id"], base_url=mcfg["base_url"], temperature=0.0,
                               max_tokens=m.get("max_tokens", 8000), extra_body=m.get("extra_body"),
                               timeout=m.get("timeout", 120))
        cache = JsonlCache(ROOT / "results/raw/forecast_clean" / f"{mkey}.jsonl")
        print(mkey, run_jobs(client, cache, jobs, parse_yes_no, f"fc-{uuid.uuid4().hex[:8]}",
                             workers=args.workers, progress_every=200), flush=True)
    print("FORECAST_DONE", flush=True)


if __name__ == "__main__":
    main()
