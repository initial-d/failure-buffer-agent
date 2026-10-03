"""Run the new studies. Study A/B/C use the belief schema; Study C's deploy question uses its own.

Budget caps for Study B are imposed by lowering max_tokens per condition (trace is truncated, the JSON
object still comes), which the client supports because max_tokens is part of the request key.
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
from fba.parsing import parse_belief, extract_json  # noqa: E402

ANSWER_TOKENS = 160          # room for the JSON object
BUDGETS = {"direct": None, "b32": 32, "b128": 128, "b512": 512}


def parse_deploy(text):
    d = extract_json(text)
    a = str(d["deploy"]).strip().lower()
    if a not in ("yes", "no", "true", "false"):
        raise ValueError(f"not a decision: {a[:20]!r}")
    return {"deploy": a in ("yes", "true")}


PARSERS = {"deploy": parse_deploy}


def jobs_for(study):
    rows = [json.loads(l) for l in open(ROOT / "data/generated" / study / "conditions.jsonl")]
    jobs = []
    for c in rows:
        qtype = c.get("question_type")
        parse = PARSERS["deploy"] if qtype == "deploy" else parse_belief
        cap = None
        if study == "budget":
            b = BUDGETS[c["budget"]]
            cap = None if b is None else b + ANSWER_TOKENS
        jobs.append({"condition_id": c["condition_id"], "instance_id": c["instance_id"],
                     "system": c["system"], "prompt": c["prompt"], "prompt_hash": c["prompt_hash"],
                     "sample_idx": 0, "max_tokens": cap, "parser": parse})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    mcfg = yaml.safe_load(open(ROOT / "configs/models.yaml"))
    jobs = jobs_for(args.study)[: args.limit]
    for mkey in args.models:
        m = mcfg["models"][mkey]
        # group by token cap: one client per cap (the cap is part of the request key)
        groups = {}
        for j in jobs:
            groups.setdefault((j["max_tokens"], j["parser"]), []).append(j)
        cache = JsonlCache(ROOT / "results/raw" / args.study / f"{mkey}.jsonl")
        for (cap, parser), gj in sorted(groups.items(), key=lambda kv: (kv[0][0] is not None, kv[0][0] or 0)):
            if cap is None:
                continue
            client = ChatClient(m["id"], base_url=mcfg["base_url"], temperature=0.0, max_tokens=cap,
                                   extra_body=m.get("extra_body"), timeout=m.get("timeout", 120))
            st = run_jobs(client, cache, gj, parser, f"{args.study}-{uuid.uuid4().hex[:8]}",
                          workers=args.workers, progress_every=500)
            print(mkey, "cap", cap, "parser", parser.__name__, st, flush=True)
        for (cap, parser), gj in groups.items():
            if cap is not None:
                continue
            client = ChatClient(m["id"], base_url=mcfg["base_url"], temperature=0.0,
                                   max_tokens=m.get("max_tokens", 2000), extra_body=m.get("extra_body"),
                                   timeout=m.get("timeout", 120))
            st = run_jobs(client, cache, gj, parser, f"{args.study}-{uuid.uuid4().hex[:8]}",
                          workers=args.workers, progress_every=500)
            print(mkey, "uncapped", "parser", parser.__name__, st, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
