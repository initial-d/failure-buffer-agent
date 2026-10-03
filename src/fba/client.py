"""OpenAI-compatible chat-completions client with append-only JSONL cache.

* idempotent: a request key = sha256(model, system, prompt, params, sample_idx);
  keys already present with a successfully parsed response are never re-sent.
* retry-safe: transport/API errors are retried with backoff and are never
  recorded as experimental outcomes.
* parse failures are recorded (status="parse_error") and re-requested on the
  next run, up to `max_parse_attempts`.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import requests

DEFAULT_BASE_URL = os.environ.get("FBA_BASE_URL", "")


def load_api_key() -> str:
    key = os.environ.get("FBA_API_KEY")
    if key:
        return key.strip()
    path = os.environ.get("FBA_API_KEY_FILE")
    if not path:
        raise RuntimeError("Set FBA_API_KEY (or FBA_API_KEY_FILE) to an API key for an OpenAI-compatible endpoint.")
    return Path(path).read_text().strip().splitlines()[0]


def request_key(model: str, system: str, prompt: str, params: dict, sample_idx: int) -> str:
    blob = json.dumps([model, system, prompt, params, sample_idx], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class ChatClient:
    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL, api_key: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 8000, timeout: int = 300, max_retries: int = 8,
                 extra_body: Optional[dict] = None):
        self.model = model
        base_url = base_url or DEFAULT_BASE_URL
        if not base_url:
            raise RuntimeError("Set FBA_BASE_URL to an OpenAI-compatible endpoint, e.g. https://host/v1")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key or load_api_key()
        self.params = {"temperature": temperature, "max_tokens": max_tokens, "top_p": 1, **(extra_body or {})}
        self.params = {k: v for k, v in self.params.items() if v is not None}  # e.g. top_p: null drops it
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(self, system: str, prompt: str) -> dict:
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                **self.params}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                r = requests.post(self.url, headers=headers, json=body, timeout=self.timeout)
                data = r.json()
                if r.status_code == 200 and data.get("choices") and "error" not in data:
                    msg = data["choices"][0].get("message") or {}
                    return {
                        "content": msg.get("content") or "",
                        "logprobs": ((data["choices"][0].get("logprobs") or {}).get("content") or [])[:3],
                        "reasoning_content": msg.get("reasoning_content"),
                        "finish_reason": data["choices"][0].get("finish_reason"),
                        "model_version": data.get("model"),
                        "usage": data.get("usage", {}),
                        "latency": round(time.time() - t0, 3),
                        "api_attempts": attempt + 1,
                    }
                last_err = f"HTTP {r.status_code}: {str(data)[:300]}"
            except Exception as e:  # network / JSON decode
                last_err = repr(e)[:300]
            time.sleep(min(30, 2 * 2 ** attempt))
        raise RuntimeError(f"API failed after {self.max_retries} attempts: {last_err}")


class JsonlCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.records: dict[str, dict] = {}
        self.parse_attempts: dict[str, int] = {}
        if self.path.exists():
            with open(self.path) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    k = rec["request_key"]
                    self.parse_attempts[k] = self.parse_attempts.get(k, 0) + 1
                    if rec["status"] == "ok" or k not in self.records:
                        if self.records.get(k, {}).get("status") != "ok":
                            self.records[k] = rec

    def done(self, key: str) -> bool:
        return self.records.get(key, {}).get("status") == "ok"

    def append(self, rec: dict):
        with self.lock:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            k = rec["request_key"]
            self.parse_attempts[k] = self.parse_attempts.get(k, 0) + 1
            if rec["status"] == "ok" or self.records.get(k, {}).get("status") != "ok":
                self.records[k] = rec


def run_jobs(client: ChatClient, cache: JsonlCache, jobs: Iterable[dict],
             parser: Callable[[str], dict], run_id: str, workers: int = 16,
             max_parse_attempts: int = 3, progress_every: int = 50) -> dict:
    """jobs: dicts with keys condition_id, instance_id, system, prompt, prompt_hash, sample_idx."""
    todo = []
    for j in jobs:
        k = request_key(client.model, j["system"], j["prompt"], client.params, j.get("sample_idx", 0))
        if cache.done(k) or cache.parse_attempts.get(k, 0) >= max_parse_attempts:
            continue
        todo.append((k, j))
    stats = {"submitted": len(todo), "ok": 0, "parse_error": 0, "api_error": 0}
    if not todo:
        return stats

    def work(k, j):
        resp = client.complete(j["system"], j["prompt"])
        try:
            parsed, status, err = parser(resp["content"]), "ok", None
        except Exception as e:
            parsed, status, err = None, "parse_error", repr(e)[:300]
        return {
            "request_key": k, "run_id": run_id, "condition_id": j["condition_id"],
            "instance_id": j["instance_id"], "sample_idx": j.get("sample_idx", 0),
            "model": client.model, "model_version": resp["model_version"], 
            **client.params, "prompt_hash": j["prompt_hash"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_response": resp["content"], "logprobs": resp["logprobs"] or None, "reasoning_content": resp["reasoning_content"],
            "finish_reason": resp["finish_reason"], "parsed_response": parsed, "status": status,
            "parse_error": err, "latency": resp["latency"], "token_usage": resp["usage"],
            "api_attempts": resp["api_attempts"],
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, k, j): k for k, j in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
                cache.append(rec)
                stats[rec["status"]] += 1
            except Exception as e:
                stats["api_error"] += 1
                print(f"[{client.model}] API error (not recorded): {e}", flush=True)
            if i % progress_every == 0 or i == len(todo):
                print(f"[{client.model}] {i}/{len(todo)} done in {time.time()-t0:.0f}s  {stats}", flush=True)
    return stats
