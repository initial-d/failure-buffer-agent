import argparse
import json
import sys
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.client import ChatClient, JsonlCache, run_jobs  # noqa: E402
from fba.parsing import parse_belief  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/study1.yaml"))
    ap.add_argument("--models", nargs="*", help="subset of model keys")
    ap.add_argument("--limit", type=int, default=None, help="only first N conditions (smoke test)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mcfg = yaml.safe_load(open(ROOT / "configs/models.yaml"))

    conds = [json.loads(l) for l in open(ROOT / "data/generated" / cfg["study"] / "conditions.jsonl")]
    if args.limit:
        conds = conds[: args.limit]
    jobs = [{"condition_id": c["condition_id"], "instance_id": c["instance_id"], "system": c["system"],
             "prompt": c["prompt"], "prompt_hash": c["prompt_hash"], "sample_idx": 0} for c in conds]
    run_id = f"{cfg['study']}-{uuid.uuid4().hex[:8]}"

    for mkey in args.models or cfg["models"]:
        m = mcfg["models"][mkey]
        client = ChatClient(m["id"], base_url=mcfg["base_url"], temperature=cfg["temperature"],
                               max_tokens=m.get("max_tokens", 8000), extra_body=m.get("extra_body"),
                               timeout=m.get("timeout", 300))
        cache = JsonlCache(ROOT / "results/raw" / cfg["study"] / f"{mkey}.jsonl")
        stats = run_jobs(client, cache, jobs, parse_belief, run_id, workers=args.workers or cfg["workers"])
        print(mkey, stats, flush=True)


if __name__ == "__main__":
    main()
