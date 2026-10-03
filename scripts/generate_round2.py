"""Generate round-2 studies: deployment v2, mitigation arms, robustness cells."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.tasks.domains import DOMAINS  # noqa: E402
from fba.tasks.generator import (EXOG_TEMPLATE, FORMAT_INSTRUCTION, build_conditions,  # noqa: E402
                                 make_base_instance, _fmt, prompt_hash)
from fba.tasks.round2 import (build_deploy2, build_mitigation, build_rewrite_conditions,  # noqa: E402
                              build_robustness, d2_record, mit_record, rob_record)

SEED = 20261120


def instance_dict(inst):
    spec = DOMAINS[inst.domain]
    f = inst.fields
    bspec = spec["buffers"][inst.buffer_type]
    return {
        "instance_id": inst.instance_id, "domain": inst.domain,
        "context": _fmt(spec["context"], f),
        "evidence_qualitative": inst.evidence["qualitative"],
        "question": f"What is the probability that {_fmt(spec['target'], f)}?",
        "buffer_low": _fmt(bspec["low"], f), "buffer_high": _fmt(bspec["high"], f),
        "buffer_type": inst.buffer_type,
        "exog": EXOG_TEMPLATE.format(entity=spec["entity"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="+", default=["deploy2", "mitigation", "robustness"])
    args = ap.parse_args()

    if "deploy2" in args.studies:
        out = ROOT / "data/generated/deploy2"; out.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out / "conditions.jsonl", "w") as fh:
            for i in range(40):
                for c in build_deploy2(i, SEED):
                    fh.write(json.dumps(d2_record(c), ensure_ascii=False) + "\n"); n += 1
        print(f"deploy2: {n} conditions")

    if "mitigation" in args.studies:
        out = ROOT / "data/generated/mitigation"; out.mkdir(parents=True, exist_ok=True)
        models = ["gpt-4o-mini", "qwen3.5-27b", "deepseek-v4.1-flash"]
        # stage 1: every model's own buffer-free qualitative estimate, taken from the main study
        own = {}
        for m in models:
            cands = list((ROOT / "results/raw/study1_pilot").glob(f"{m}*.jsonl"))
            cands = [c for c in cands if "think" not in c.stem]
            if not cands:
                continue
            vals = {}
            for l in open(cands[0]):
                r = json.loads(l)
                if r["status"] == "ok" and "__qualitative__none" in r["condition_id"]:
                    vals[r["condition_id"].split("__")[0]] = r["parsed_response"]["success_probability"]
            own[m] = vals
        n = 0
        with open(out / "conditions.jsonl", "w") as fh:
            for d in DOMAINS:
                for i in range(20):
                    inst = instance_dict(make_base_instance(d, i, 20261002))
                    for m in models:
                        est = own.get(m, {}).get(inst["instance_id"])
                        if est is None:
                            continue
                        for c in build_mitigation(inst, inst["buffer_low"], inst["buffer_high"], est, m):
                            fh.write(json.dumps(mit_record(c), ensure_ascii=False) + "\n"); n += 1
        print(f"mitigation: {n} conditions")

    if "robustness" in args.studies:
        out = ROOT / "data/generated/robustness"; out.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out / "conditions.jsonl", "w") as fh:
            for d in DOMAINS:
                for i in range(20):
                    inst = instance_dict(make_base_instance(d, i, 20261002))
                    for order in ("after", "before"):
                        for style in ("para", "bullet"):
                            for c in build_robustness(inst, inst["buffer_low"], inst["buffer_high"],
                                                      inst["exog"], order, style):
                                fh.write(json.dumps(rob_record(c), ensure_ascii=False) + "\n"); n += 1
                    for c in build_rewrite_conditions(inst, inst["buffer_type"], inst["exog"]):
                        fh.write(json.dumps(rob_record(c), ensure_ascii=False) + "\n"); n += 1
        print(f"robustness: {n} conditions")


if __name__ == "__main__":
    main()
