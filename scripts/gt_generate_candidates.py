"""Ground-truth task: real MBPP problems, model-written candidate patches, hidden tests run locally.

Stage 1 (this script): generate one candidate per problem with a cheap model at temperature 0.8, execute
every test in MBPP's test_list in a subprocess with a timeout, and record pass/fail. The outcome of a
candidate is a fact about code, not about a prompt, so it is identical across buffer conditions.
"""
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.client import ChatClient  # noqa: E402

GEN_SYSTEM = "You are a Python programmer. Reply with a single Python code block and nothing else."
FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def fn_name(test):
    m = re.search(r"assert\s+(?:\w+\()?\s*([A-Za-z_]\w*)\s*\(", test)
    return m.group(1) if m else None


def extract_code(text):
    m = FENCE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def run_tests(code, tests, setup="", timeout=6):
    prog = "\n".join([setup or "", code, "", *tests, "print('ALL_PASS')"])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(prog)
        path = fh.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return "ALL_PASS" in r.stdout
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def run_one(code, test, setup=""):
    return run_tests(code, [test], setup)


def main(n_problems=974, model_id="qwen/qwen-turbo", temperature=0.8):
    probs = [json.loads(l) for l in open(ROOT / "data/mbpp/mbpp.jsonl")][:n_problems]
    client = ChatClient(model_id, temperature=temperature, max_tokens=700, timeout=90)
    out_path = ROOT / "data/mbpp/candidates.jsonl"
    done = {}
    if out_path.exists():
        for l in open(out_path):
            r = json.loads(l)
            done[r["task_id"]] = r

    def work(p):
        if p["task_id"] in done:
            return done[p["task_id"]]
        name = fn_name(p["test_list"][0])
        prompt = (f"{p['text']}\n\nThe function must be called `{name}` and pass tests like:\n"
                  f"{p['test_list'][0]}")
        try:
            resp = client.complete(GEN_SYSTEM, prompt)
        except Exception as e:  # API failure: skip, never recorded as an outcome
            return None
        code = extract_code(resp["content"])
        setup = p.get("test_setup_code", "")
        hidden = p["test_list"] + p.get("challenge_test_list", [])
        rec = {"task_id": p["task_id"], "text": p["text"], "fn": name, "code": code,
               "shown_test": p["test_list"][0], "shown_pass": run_one(code, p["test_list"][0], setup),
               "hidden_pass": run_tests(code, hidden, setup), "n_hidden": len(hidden),
               "generator": model_id, "temperature": temperature}
        return rec

    with ThreadPoolExecutor(32) as ex, open(out_path, "a") as fh:
        for rec in ex.map(work, probs):
            if rec and rec["task_id"] not in done:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done[rec["task_id"]] = rec
    recs = list(done.values())
    sp = [r for r in recs if r["shown_pass"]]
    print(f"{len(recs)} candidates; shown test passes: {len(sp)}; "
          f"of those, hidden tests pass: {sum(r['hidden_pass'] for r in sp)}")


if __name__ == "__main__":
    main()
