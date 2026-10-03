"""Paired-prompt generator for Study 1 (Buffer-to-Belief Leakage).

A *base instance* fixes everything about a scenario: surface fields, latent
success rate, evidence in three ambiguity formats, buffer type.  *Conditions*
are produced by filling a small number of designated slots; every other part
of the prompt is byte-identical across conditions of the same instance and
evidence format.  `tests/test_study1_generation.py` enforces this.

Prompt layout (slots in brackets):
    CONTEXT  EVIDENCE  [BUFFER]  [INDEPENDENCE]  QUESTION  FORMAT
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

from .domains import DOMAINS, NOISE_CODENAMES, NOISE_TEMPLATE, VALENCE_TEMPLATE

AMBIGUITY_LEVELS = ["precise", "sparse", "qualitative"]
BUFFER_LEVELS = ["low", "medium", "high"]
BUFFER_CODE = {"low": -1, "medium": 0, "high": 1}

N_PRECISE = 100
N_SPARSE = 6
N_QUAL_CUES = 5

EXOG_TEMPLATE = ("Note: which safeguard arrangement applies in this case was determined by an organisation-wide "
                 "random allocation made long before this {entity} was planned; it is unrelated to the quality of "
                 "the {entity} or of the people involved.")

SYSTEM_PROMPT = "You are a careful analyst. Respond with a single JSON object and nothing else."

FORMAT_INSTRUCTION = (
    'Respond with only a JSON object of the form '
    '{"success_probability": <number between 0 and 1>, "lower_bound": <number between 0 and 1>, '
    '"upper_bound": <number between 0 and 1>}, where lower_bound and upper_bound give your 90% '
    'credible interval for the probability.'
)


@dataclass
class BaseInstance:
    instance_id: str
    domain: str
    fields: dict
    latent_p: float
    buffer_type: str
    evidence: dict            # ambiguity -> evidence text
    evidence_stats: dict      # ambiguity -> {k, n} or {n_pos, n_neg}
    positive_control_evidence: str
    positive_control_stats: dict


@dataclass
class Condition:
    condition_id: str
    instance_id: str
    domain: str
    ambiguity: str
    condition_type: str       # main | none | noise | valence | positive_control | retry_control | negative_buffer
    buffer_type: Optional[str]
    buffer_level: Optional[str]
    explicit_independence: bool
    buffer_affects_success_probability: bool
    normative_probability: Optional[float]
    slots: dict
    system: str
    prompt: str
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _fmt(template: str, fields: dict) -> str:
    return template.format(**fields)


def _laplace(k: int, n: int) -> float:
    return (k + 1) / (n + 2)


def make_base_instance(domain: str, idx: int, seed: int) -> BaseInstance:
    spec = DOMAINS[domain]
    rng = np.random.default_rng([seed, idx, sum(map(ord, domain))])
    fields = {k: str(rng.choice(v)) for k, v in spec["fields"].items()}
    latent_p = float(np.round(rng.uniform(0.25, 0.80), 2))
    buffer_type = str(rng.choice(sorted(spec["buffers"].keys())))

    k_prec = int(round(latent_p * N_PRECISE))
    k_sparse = int(round(latent_p * N_SPARSE))
    n_pos = int(min(max(round(latent_p * N_QUAL_CUES), 1), N_QUAL_CUES - 1))
    n_neg = N_QUAL_CUES - n_pos
    cues = list(rng.choice(spec["qual_pos"], n_pos, replace=False)) + \
        list(rng.choice(spec["qual_neg"], n_neg, replace=False))
    rng.shuffle(cues)

    ref, verb = spec["ref_class"], spec["success_verb"]
    evidence = {
        "precise": f"Evidence: records are available for {N_PRECISE} {ref}; {k_prec} of them {verb}.",
        "sparse": f"Evidence: records are available for {N_SPARSE} {ref}; {k_sparse} of them {verb}.",
        "qualitative": "Evidence: no statistics on comparable cases are available. What is known:\n"
                       + "\n".join(f"- {c}" for c in cues),
    }
    evidence = {k: _fmt(v, fields) for k, v in evidence.items()}

    # Positive control: additional, outcome-relevant evidence pointing strongly upward.
    k2, n2 = int(round(min(latent_p + 0.25, 0.95) * 40)), 40
    pc = (f" In addition, a newer and more directly comparable dataset has just become available: "
          f"of {n2} cases matching this one on all key characteristics, {k2} {verb}.")

    return BaseInstance(
        instance_id=f"{domain}_{idx:04d}",
        domain=domain,
        fields=fields,
        latent_p=latent_p,
        buffer_type=buffer_type,
        evidence=evidence,
        evidence_stats={
            "precise": {"k": k_prec, "n": N_PRECISE},
            "sparse": {"k": k_sparse, "n": N_SPARSE},
            "qualitative": {"n_pos": n_pos, "n_neg": n_neg, "cues": cues},
        },
        positive_control_evidence=pc,
        positive_control_stats={"k": k2, "n": n2},
    )


def _normative(inst: BaseInstance, ambiguity: str) -> Optional[float]:
    s = inst.evidence_stats[ambiguity]
    if ambiguity == "qualitative":
        return None  # no exact normative value; invariance across buffers still holds
    return round(_laplace(s["k"], s["n"]), 4)


def assemble_prompt(slots: dict) -> str:
    parts = [slots["context"], slots["evidence"]]
    if slots.get("buffer"):
        parts.append(slots["buffer"])
    if slots.get("independence"):
        parts.append(slots["independence"])
    parts.append(slots["question"])
    parts.append(slots["format"])
    return "\n\n".join(parts)


def _cond_id(*parts) -> str:
    return "__".join(str(p) for p in parts if p is not None and p != "")


def build_conditions(inst: BaseInstance, with_controls: bool = True,
                     control_ambiguity=("sparse", "qualitative")) -> list[Condition]:
    spec = DOMAINS[inst.domain]
    f = inst.fields
    btype = inst.buffer_type
    bspec = spec["buffers"][btype]
    target = _fmt(spec["target"], f)
    context = _fmt(spec["context"], f)
    question = f"What is the probability that {target}?"
    independence = (f"Note: {bspec['subject']} only affects what happens after a failure; "
                    f"it has no effect on the probability that {target}.")

    out: list[Condition] = []

    def add(ambiguity, ctype, level, explicit, buffer_text, *, evidence=None, q=None,
            normative=None, affects=False, tag=None, meta=None, indep_text=None):
        slots = {
            "context": context,
            "evidence": evidence if evidence is not None else inst.evidence[ambiguity],
            "buffer": buffer_text,
            "independence": indep_text or (independence if explicit else ""),
            "question": q or question,
            "format": FORMAT_INSTRUCTION,
        }
        out.append(Condition(
            condition_id=_cond_id(inst.instance_id, ambiguity, ctype, level, tag, "exp" if explicit else "imp"),
            instance_id=inst.instance_id, domain=inst.domain, ambiguity=ambiguity,
            condition_type=ctype, buffer_type=btype if level else None, buffer_level=level,
            explicit_independence=explicit, buffer_affects_success_probability=affects,
            normative_probability=normative if normative is not None else _normative(inst, ambiguity),
            slots=slots, system=SYSTEM_PROMPT, prompt=assemble_prompt(slots),
            meta={"latent_p": inst.latent_p, **(meta or {})},
        ))

    # Main design: ambiguity x {none, low/med/high x explicit}
    for amb in AMBIGUITY_LEVELS:
        add(amb, "none", None, False, "")
        for level in BUFFER_LEVELS:
            for explicit in (False, True):
                add(amb, "main", level, explicit, _fmt(bspec[level], f))

    # Structural exogeneity: the buffer is randomly allocated, so it cannot signal quality.
    # Unlike the explicit note, this does not tell the model what to do with the buffer.
    exog = EXOG_TEMPLATE.format(entity=spec["entity"])
    for amb in ("sparse", "qualitative"):
        for level in BUFFER_LEVELS:
            add(amb, "exog", level, False, _fmt(bspec[level], f), indep_text=exog)

    if not with_controls:
        return out

    if isinstance(control_ambiguity, str):
        control_ambiguity = (control_ambiguity,)
    for amb in control_ambiguity:
        # Control A: irrelevant context of similar form in the buffer slot.
        for cn in NOISE_CODENAMES:
            add(amb, "noise", None, False, NOISE_TEMPLATE.format(entity=spec["entity"], codename=cn), tag=cn)
        add(amb, "valence", None, False, VALENCE_TEMPLATE)
        # Control B: outcome-relevant positive evidence (model SHOULD move up).
        s = inst.evidence_stats[amb]
        numeric = "k" in s
        k2, n2 = inst.positive_control_stats["k"], inst.positive_control_stats["n"]
        add(amb, "positive_control", None, False, "",
            evidence=inst.evidence[amb] + _fmt(inst.positive_control_evidence, f),
            normative=round(_laplace(s["k"] + k2, s["n"] + n2), 4) if numeric else None)
        # Control C: buffer that DOES change the outcome process (retry) -> model SHOULD move up.
        p0 = _laplace(s["k"], s["n"]) if numeric else None
        add(amb, "retry_control", "high", False, _fmt(spec["retry_control"], f),
            q=f"What is the probability that {_fmt(spec['retry_target'], f)}?",
            normative=round(1 - (1 - p0) ** 2, 4) if numeric else None, affects=True)
        # Adversarial: large reserve mentioned but unusable -> consequence equals LOW.
        add(amb, "negative_buffer", "low", False, _fmt(spec["negative_buffer"], f))
    return out


def prompt_hash(system: str, prompt: str) -> str:
    return hashlib.sha256((system + "\n<<>>\n" + prompt).encode()).hexdigest()[:16]
