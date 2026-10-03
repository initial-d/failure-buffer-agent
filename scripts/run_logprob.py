"""Implicit-belief elicitation: binary Yes/No forecast, belief = P(Yes) from first-token logprobs.

Reuses the Study 1 conditions (same context / evidence / buffer / independence slots); only the
question and answer format differ.  No number is ever verbalised, so this measure is immune to
the "verbal probability is not the model's belief" objection.
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

SYSTEM = "You are a careful analyst. Answer with exactly one word."
KEEP = {"main", "exog", "none", "noise", "negative_buffer", "positive_control"}


def to_binary(c):
    q = c["slots"]["question"]
    assert q.startswith("What is the probability that ") and q.endswith("?")
    event = q[len("What is the probability that "):-1]
    parts = [c["slots"]["context"], c["slots"]["evidence"]]
    for k in ("buffer", "independence"):
        if c["slots"].get(k):
            parts.append(c["slots"][k])
    parts.append(f"Make your single best prediction: will it turn out that {event}?")
    parts.append("Answer with exactly one word: Yes or No.")
    return "\n\n".join(parts)


def parse_yes_no(text):
    t = text.strip().lower().strip(".")
    if t not in ("yes", "no"):
        raise ValueError(f"not yes/no: {text[:40]!r}")
    return {"answer": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["qwen3.5-27b", "gpt-4o-mini"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    mcfg = yaml.safe_load(open(ROOT / "configs/models.yaml"))
    conds = [json.loads(l) for l in open(ROOT / "data/generated/study1_pilot/conditions.jsonl")]
    conds = [c for c in conds if c["condition_type"] in KEEP][: args.limit]
    jobs = []
    for c in conds:
        p = to_binary(c)
        jobs.append({"condition_id": c["condition_id"], "instance_id": c["instance_id"], "system": SYSTEM,
                     "prompt": p, "prompt_hash": prompt_hash(SYSTEM, p), "sample_idx": 0})
    for mkey in args.models:
        m = mcfg["models"][mkey]
        extra = {**(m.get("extra_body") or {}), "logprobs": True, "top_logprobs": 5}
        client = ChatClient(m["id"], base_url=mcfg["base_url"], temperature=0.0, max_tokens=5,
                               extra_body=extra, timeout=m.get("timeout", 120))
        cache = JsonlCache(ROOT / "results/raw/study1_logprob" / f"{mkey}.jsonl")
        print(mkey, run_jobs(client, cache, jobs, parse_yes_no, f"s1lp-{uuid.uuid4().hex[:8]}",
                             workers=args.workers, progress_every=500), flush=True)


if __name__ == "__main__":
    main()
