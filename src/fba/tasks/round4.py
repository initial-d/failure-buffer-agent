"""Supplementary studies.

known_p   Categorical forecast with the probability stated as known ("the probability of success for this
          case is known to be 0.62"), removing any dispute about the normative answer: under accuracy-only
          scoring the correct forecast is Yes iff p > 0.5, whatever the buffer.
strict    Main contrast (qualitative evidence, random allocation, low vs high buffer) with the target event
          worded so that it cannot absorb recovery: the event concerns the first attempt itself, and any
          rollback, reimbursement or restoration is stated not to count as success.
route     Main contrast re-issued as independent calls (same prompts as Study 1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .domains import DOMAINS
from .generator import EXOG_TEMPLATE, FORMAT_INSTRUCTION, SYSTEM_PROMPT, _fmt, make_base_instance, prompt_hash

FC_SYSTEM = "You are a careful forecaster. Answer with exactly one word."
SCORING = ("This is a pure forecasting exercise. Your forecast is scored only on accuracy: 1 point if it matches "
           "what actually happens, 0 points otherwise. It will not be shown to anyone involved, will not be used "
           "for any decision, and is not advice about whether to go ahead.")
KNOWN_P = [0.30, 0.38, 0.45, 0.55, 0.62, 0.70]
STRICT_SUFFIX = (" The question concerns this first attempt itself. If it fails, any rollback, reimbursement, "
                 "restoration or later retry does not count as success.")


def _rec(cid, iid, system, prompt, **kw):
    r = {"condition_id": cid, "instance_id": iid, "system": system, "prompt": prompt, **kw}
    r["prompt_hash"] = prompt_hash(system, prompt)
    return r


def build_known_p(seed=20261005, per_domain=20):
    out = []
    for d, spec in DOMAINS.items():
        for i in range(per_domain):
            inst = make_base_instance(d, i, seed)
            f = inst.fields
            rng = np.random.default_rng([seed, i, sum(map(ord, d)), 41])
            p = float(rng.choice(KNOWN_P))
            bspec = spec["buffers"][inst.buffer_type]
            target = _fmt(spec["target"], f)
            known = (f"From an audited forecasting model with an excellent track record, the probability that "
                     f"{target} is known to be {p:.2f}.")
            for lvl in ("none", "low", "high"):
                parts = [_fmt(spec["context"], f), known]
                if lvl != "none":
                    parts += [_fmt(bspec[lvl], f), EXOG_TEMPLATE.format(entity=spec["entity"])]
                parts += [SCORING, f"Forecast: will it turn out that {target}?",
                          "Answer with exactly one word: Yes or No."]
                out.append(_rec(f"{inst.instance_id}__kp__{lvl}", inst.instance_id, FC_SYSTEM, "\n\n".join(parts),
                                known_p=p, buffer_level=lvl, domain=d))
    return out


def build_strict(seed=20261002, per_domain=20):
    out = []
    for d, spec in DOMAINS.items():
        for i in range(per_domain):
            inst = make_base_instance(d, i, seed)
            f = inst.fields
            bspec = spec["buffers"][inst.buffer_type]
            q = f"What is the probability that {_fmt(spec['target'], f)}?" + STRICT_SUFFIX
            for lvl in ("low", "high"):
                parts = [_fmt(spec["context"], f), inst.evidence["qualitative"], _fmt(bspec[lvl], f),
                         EXOG_TEMPLATE.format(entity=spec["entity"]), q, FORMAT_INSTRUCTION]
                out.append(_rec(f"{inst.instance_id}__strict__{lvl}", inst.instance_id, SYSTEM_PROMPT,
                                "\n\n".join(parts), buffer_level=lvl, domain=d))
    return out


def build_route(root: Path):
    """The exact Study 1 prompts for the qualitative random-allocation cells and the no-buffer cell."""
    out = []
    for l in open(root / "data/generated/study1_pilot/conditions.jsonl"):
        c = json.loads(l)
        if c["ambiguity"] == "qualitative" and (c["condition_type"] == "exog" and c["buffer_level"] in ("low", "high")
                                                or c["condition_type"] == "none"):
            out.append(_rec(c["condition_id"], c["instance_id"], c["system"], c["prompt"],
                            buffer_level=c["buffer_level"] or "none", domain=c["domain"]))
    return out
