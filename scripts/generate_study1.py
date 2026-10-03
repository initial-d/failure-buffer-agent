import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.tasks.generator import make_base_instance, build_conditions, prompt_hash  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/study1.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = ROOT / "data/generated" / cfg["study"]
    out_dir.mkdir(parents=True, exist_ok=True)

    n_inst = n_cond = 0
    with open(out_dir / "instances.jsonl", "w") as fi, open(out_dir / "conditions.jsonl", "w") as fc:
        for domain in cfg["domains"]:
            for i in range(cfg["instances_per_domain"]):
                inst = make_base_instance(domain, i, cfg["seed"])
                fi.write(json.dumps(asdict(inst), ensure_ascii=False) + "\n")
                n_inst += 1
                for c in build_conditions(inst, control_ambiguity=tuple(cfg["control_ambiguity"])):
                    rec = asdict(c)
                    rec["prompt_hash"] = prompt_hash(c.system, c.prompt)
                    fc.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_cond += 1
    print(f"wrote {n_inst} instances, {n_cond} conditions -> {out_dir}")


if __name__ == "__main__":
    main()
