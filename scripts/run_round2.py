"""Runner for the round-2 studies: mixed output schemas, one tolerant parser."""
import argparse
import json
import sys
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.client import ChatClient, JsonlCache, run_jobs  # noqa: E402
from fba.parsing import _prob, extract_json  # noqa: E402


def parse_mixed(text):
    """Accepts any subset of the round-2 fields and returns what is present."""
    d = extract_json(text)
    out = {}
    for key, alias in (("success_probability", "success_probability"), ("probability", "success_probability")):
        if key in d:
            out["success_probability"] = _prob(d[key])
            break
    for k in ("lower_bound", "lower"):
        if k in d:
            out["lower_bound"] = _prob(d[k])
            break
    for k in ("upper_bound", "upper"):
        if k in d:
            out["upper_bound"] = _prob(d[k])
            break
    for key in ("deploy", "proceed"):
        if key in d:
            v = str(d[key]).strip().lower()
            if v not in ("yes", "no", "true", "false"):
                raise ValueError(f"bad decision value: {v[:20]!r}")
            out[key] = v in ("yes", "true")
    if not out:
        raise ValueError(f"no recognised fields in {text[:60]!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    mcfg = yaml.safe_load(open(ROOT / "configs/models.yaml"))
    rows = [json.loads(l) for l in open(ROOT / "data/generated" / args.study / "conditions.jsonl")]
    jobs = [{"condition_id": c["condition_id"], "instance_id": c["instance_id"], "system": c["system"],
             "prompt": c["prompt"], "prompt_hash": c["prompt_hash"], "sample_idx": 0} for c in rows]
    if args.limit:
        jobs = jobs[: args.limit]
    for mkey in args.models:
        m = mcfg["models"][mkey]
        client = ChatClient(m["id"], base_url=mcfg["base_url"], temperature=0.0,
                               max_tokens=m.get("max_tokens", 1200), extra_body=m.get("extra_body"),
                               timeout=m.get("timeout", 120))
        cache = JsonlCache(ROOT / "results/raw" / args.study / f"{mkey}.jsonl")
        st = run_jobs(client, cache, jobs, parse_mixed, f"{args.study}-{uuid.uuid4().hex[:8]}",
                      workers=args.workers, progress_every=400)
        print(mkey, st, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
