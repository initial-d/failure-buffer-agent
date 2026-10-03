import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.tasks.feedback import build_s3_instance, s3_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/study3.yaml"))
    cfg = yaml.safe_load(open(ap.parse_args().config))
    out_dir = ROOT / "data/generated" / cfg["study"]
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_dir / "conditions.jsonl", "w") as fc:
        for d in cfg["domains"]:
            for i in range(cfg["instances_per_domain"]):
                for c in build_s3_instance(d, i, cfg["seed"]):
                    fc.write(json.dumps(s3_record(c), ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} conditions -> {out_dir}")


if __name__ == "__main__":
    main()
