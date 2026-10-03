"""Generate and run the supplementary studies (known_p, strict, route)."""
import json
import sys
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fba.client import ChatClient, JsonlCache, run_jobs  # noqa: E402
from fba.parsing import parse_belief  # noqa: E402
from fba.tasks.round4 import build_known_p, build_route, build_strict  # noqa: E402
from run_forecast import parse_yes_no  # noqa: E402

STUDIES = {"known_p": (build_known_p, parse_yes_no, 0), "strict": (build_strict, parse_belief, 0),
           "route": (lambda: build_route(ROOT), parse_belief, 1)}   # sample_idx=1 forces fresh calls


def main(study, models, workers):
    build, parser, sidx = STUDIES[study]
    conds = build()
    out = ROOT / "data/generated" / study
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "conditions.jsonl", "w") as fh:
        for c in conds:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    jobs = [{**{k: c[k] for k in ("condition_id", "instance_id", "system", "prompt", "prompt_hash")},
             "sample_idx": sidx} for c in conds]
    mcfg = yaml.safe_load(open(ROOT / "configs/models.yaml"))
    for mkey in models:
        m = mcfg["models"][mkey]
        client = ChatClient(m["id"], base_url=mcfg["base_url"], temperature=0.0,
                               max_tokens=m.get("max_tokens", 2000), extra_body=m.get("extra_body"),
                               timeout=m.get("timeout", 120))
        cache = JsonlCache(ROOT / "results/raw" / study / f"{mkey}.jsonl")
        for _ in range(2):
            st = run_jobs(client, cache, jobs, parser, f"{study}-{uuid.uuid4().hex[:6]}", workers=workers,
                          progress_every=1000)
        print(study, mkey, len(jobs), st, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[3:], int(sys.argv[2]))
