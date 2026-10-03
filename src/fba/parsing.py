"""Strict-but-tolerant parser for the Study 1 belief schema."""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found")
        return json.loads(text[start:end + 1])


def _prob(x) -> float:
    v = float(x)
    if 1.0 < v <= 100.0:   # percentage given instead of probability
        v /= 100.0
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"probability out of range: {x}")
    return v


def parse_belief(text: str) -> dict:
    d = extract_json(text)
    p = _prob(d["success_probability"])
    lo = _prob(d["lower_bound"])
    hi = _prob(d["upper_bound"])
    if lo > hi:
        lo, hi = hi, lo
    return {"success_probability": p, "lower_bound": lo, "upper_bound": hi,
            "point_in_interval": lo - 1e-9 <= p <= hi + 1e-9}
