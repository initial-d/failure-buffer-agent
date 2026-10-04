# failure-buffer-agent

Code for generating paired-prompt experiments that test whether a language model's stated probability of
success changes with the consequence of failure (rollback, backup, insurance, ...) while the evidence is held
fixed, and for analysing the responses.

## Layout

```
src/fba/tasks/     scenario templates and condition generators
src/fba/client.py  OpenAI-compatible client with an idempotent JSONL cache (resumable runs)
src/fba/parsing.py response parsers
src/fba/metrics/   paired contrasts, bootstrap intervals, permutation tests
scripts/           generate_*  build prompt sets      -> data/generated/
                   run_*       query models           -> results/raw/
                   analyze_*   statistics             -> results/statistics/
tests/             invariants of the paired design (prompts differ only in the manipulated slot)
configs/           model list and study settings
```

## Data

The generated prompt sets and all raw model responses are published as a release asset:
[`failure-buffer-agent-data.tar.gz`](https://github.com/initial-d/failure-buffer-agent/releases/tag/data-v1)
(52 MB; SHA-256 `73687b7bcdaa42ad569e86f5017b92fc8cf8b75b902f257145677461bfe22adc`).
Unpack it in the repository root:

```bash
curl -LO https://github.com/initial-d/failure-buffer-agent/releases/download/data-v1/failure-buffer-agent-data.tar.gz
tar -xzf failure-buffer-agent-data.tar.gz   # creates data/generated/ and results/{raw,processed,statistics}/
```

| Path | Contents |
|---|---|
| `data/generated/<study>/` | prompt sets (one JSONL record per condition) |
| `results/raw/<study>/<model>.jsonl` | one record per model call: request key, prompt hash, model, decoding parameters, raw text, reasoning text where returned, parsed output, finish reason, latency, token usage |
| `results/processed/` | flattened response tables |
| `results/statistics/` | paired contrasts, intervals and tests produced by `scripts/analyze_*` |

When a model was served through more than one backend, `model_version` records an anonymised backend label
(`backend-1/<model>`, `backend-2/<model>`, ...), which is what the backend-heterogeneity analysis groups by.
With the archive unpacked, the `analyze_*` scripts read only these files and issue no API calls.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export FBA_BASE_URL=https://<your-openai-compatible-endpoint>/v1
export FBA_API_KEY=<your key>
.venv/bin/python -m pytest -q tests
```

Model identifiers in `configs/models.yaml` must match the names your endpoint expects.

## Typical workflow

```bash
.venv/bin/python scripts/generate_study1.py
.venv/bin/python scripts/run_study1.py --models gpt-4o-mini
.venv/bin/python scripts/analyze_study1.py --models gpt-4o-mini
```

Every request is keyed by model, prompt and decoding parameters, so an interrupted run resumes without
repeating completed calls; responses that fail to parse are retried, and transport errors are never recorded
as outcomes.

The code-execution task (`scripts/gt_generate_candidates.py`) expects MBPP at `data/mbpp/mbpp.jsonl`:

```bash
mkdir -p data/mbpp && curl -o data/mbpp/mbpp.jsonl \
  https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl
```

Candidate programs are executed locally in subprocesses with a timeout; run this only in an environment where
executing model-written code is acceptable.

## License

MIT
