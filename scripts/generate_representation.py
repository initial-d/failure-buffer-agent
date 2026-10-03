import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fba.tasks.domains import DOMAINS  # noqa: E402
from fba.tasks.representation import build_rep_instance, rep_record  # noqa: E402

out = ROOT / "data/generated/representation"; out.mkdir(parents=True, exist_ok=True)
n = 0
with open(out / "conditions.jsonl", "w") as fh:
    for d in DOMAINS:
        for i in range(20):
            for c in build_rep_instance(d, i, 20261003):
                fh.write(json.dumps(rep_record(c), ensure_ascii=False) + "\n"); n += 1
print(f"representation: {n} conditions")
