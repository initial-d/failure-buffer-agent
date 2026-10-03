"""Generate and run the round-3 ground-truth studies.

  python scripts/run_gt.py stage1        # belief without safeguard, needed by the two-stage arm
  python scripts/run_gt.py main          # vanilla, instruction, randomised-probability arms
  python scripts/run_gt.py stage2        # two-stage decision, built from each model's stage-1 output
"""
import json
import sys
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fba.client import ChatClient, JsonlCache, run_jobs  # noqa: E402
from fba.tasks.groundtruth import select_items, stage1, stage2, stage_main  # noqa: E402
from run_round2 import parse_mixed  # noqa: E402

MODELS = ["gpt-4o-mini", "qwen3.5-27b", "deepseek-v4.1-flash"]
OUT = ROOT / "data/generated/groundtruth"


def run(conds, tag, workers=64):
    mcfg = yaml.safe_load(open(ROOT / "configs/models.yaml"))
    for mkey in MODELS:
        rows = [c for c in conds if c.get("model", mkey) == mkey]
        jobs = [{"condition_id": c["condition_id"], "instance_id": c["instance_id"], "system": c["system"],
                 "prompt": c["prompt"], "prompt_hash": c["prompt_hash"], "sample_idx": 0} for c in rows]
        m = mcfg["models"][mkey]
        client = ChatClient(m["id"], base_url=mcfg["base_url"], temperature=0.0,
                               max_tokens=m.get("max_tokens", 1200), extra_body=m.get("extra_body"),
                               timeout=m.get("timeout", 120))
        cache = JsonlCache(ROOT / "results/raw/groundtruth" / f"{mkey}.jsonl")
        for _ in range(2):
            st = run_jobs(client, cache, jobs, parse_mixed, f"gt-{tag}-{uuid.uuid4().hex[:6]}",
                          workers=workers, progress_every=1000)
        print(tag, mkey, st, flush=True)


def write(conds, name):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w") as fh:
        for c in conds:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")


def own_stage1(mkey):
    p = {}
    f = ROOT / "results/raw/groundtruth" / f"{mkey}.jsonl"
    for l in open(f):
        r = json.loads(l)
        if r["status"] == "ok" and "__belief_nobuf__" in r["condition_id"]:
            p[r["instance_id"]] = r["parsed_response"]["success_probability"]
    return p


if __name__ == "__main__":
    step = sys.argv[1]
    items = select_items(ROOT / "data/mbpp/candidates.jsonl")
    if step == "stage1":
        c = stage1(items); write(c, "stage1"); run(c, "stage1")
    elif step == "main":
        c = stage_main(items); write(c, "main"); run(c, "main")
    elif step == "stage2":
        allc = []
        for mkey in MODELS:
            cs = stage2(items, own_stage1(mkey))
            for x in cs:
                x["model"] = mkey
                x["condition_id"] = x["condition_id"] + f"__{mkey}"
            allc += cs
        write(allc, "stage2"); run(allc, "stage2")
    print("DONE", step)
