"""Generate Study A (sample gradient), B (deliberation budget), C (deployment agent)
and the fresh confirmatory set (new seed, main design)."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fba.tasks.budget import budget_record, build_budget_instance  # noqa: E402
from fba.tasks.deployment import build_deploy_instance, deploy_record  # noqa: E402
from fba.tasks.gradient import build_gradient_instance, gradient_record  # noqa: E402
from fba.tasks.generator import build_conditions, make_base_instance, prompt_hash  # noqa: E402

CFG = {
    "gradient": {"domains": ["finance", "software", "database", "science", "agent_tool"], "n": 10, "seed": 20261010},
    "budget": {"domains": ["software", "agent_tool", "finance"], "n": 20, "seed": 20261011},
    "deploy": {"n": 40, "seed": 20261012},
    "confirmatory": {"domains": ["finance", "software", "database", "science", "agent_tool"], "n": 40, "seed": 20261101},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="+", default=list(CFG))
    args = ap.parse_args()
    for study in args.studies:
        cfg = CFG[study]
        out_dir = ROOT / "data/generated" / study
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out_dir / "conditions.jsonl", "w") as fc:
            if study == "gradient":
                for d in cfg["domains"]:
                    for i in range(cfg["n"]):
                        for c in build_gradient_instance(d, i, cfg["seed"]):
                            fc.write(json.dumps(gradient_record(c), ensure_ascii=False) + "\n"); n += 1
            elif study == "budget":
                for d in cfg["domains"]:
                    for i in range(cfg["n"]):
                        for c in build_budget_instance(d, i, cfg["seed"]):
                            fc.write(json.dumps(budget_record(c), ensure_ascii=False) + "\n"); n += 1
            elif study == "deploy":
                for i in range(cfg["n"]):
                    for c in build_deploy_instance(i, cfg["seed"]):
                        fc.write(json.dumps(deploy_record(c), ensure_ascii=False) + "\n"); n += 1
            elif study == "confirmatory":
                with open(out_dir / "instances.jsonl", "w") as fi:
                    for d in cfg["domains"]:
                        for i in range(cfg["n"]):
                            inst = make_base_instance(d, i, cfg["seed"])
                            fi.write(json.dumps(asdict(inst), ensure_ascii=False) + "\n")
                            for c in build_conditions(inst, control_ambiguity=("sparse", "qualitative")):
                                rec = asdict(c); rec["prompt_hash"] = prompt_hash(c.system, c.prompt)
                                fc.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
        print(f"{study}: {n} conditions -> {out_dir}")


if __name__ == "__main__":
    main()
