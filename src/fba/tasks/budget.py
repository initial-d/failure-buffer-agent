"""Study B: deliberation-budget gradient.

Same item, same evidence, same buffer; the only change is how much room the model is given to think.
Four budgets: "direct" (no reasoning requested), and reasoning capped at 32 / 128 / 512 tokens.
The cap is imposed with max_tokens less the answer budget, so a trace that runs long is truncated
mid-sentence and the model still emits the JSON object.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .domains import DOMAINS
from .generator import EXOG_TEMPLATE, SYSTEM_PROMPT, FORMAT_INSTRUCTION, _fmt, _laplace, prompt_hash

BUDGETS = {"direct": 0, "b32": 32, "b128": 128, "b512": 512}
BUFFER_LEVELS = ["low", "high"]
N_SPARSE = 6
N_QUAL = 5


@dataclass
class BudgetCondition:
    condition_id: str
    instance_id: str
    domain: str
    ambiguity: str
    budget: str
    buffer_level: str
    normative_probability: float
    system: str
    prompt: str
    meta: dict


def build_budget_instance(domain: str, idx: int, seed: int) -> list[BudgetCondition]:
    spec = DOMAINS[domain]
    rng = np.random.default_rng([seed, idx, sum(map(ord, domain)), 11])
    fields = {k: str(rng.choice(v)) for k, v in spec["fields"].items()}
    latent_p = float(np.round(rng.uniform(0.30, 0.75), 2))
    btype = str(rng.choice(sorted(spec["buffers"].keys())))
    bspec = spec["buffers"][btype]
    context = _fmt(spec["context"], fields)
    target = _fmt(spec["target"], fields)
    question = f"What is the probability that {target}?"
    exog = EXOG_TEMPLATE.format(entity=spec["entity"])
    ref, verb = spec["ref_class"], spec["success_verb"]

    k_sparse = int(round(latent_p * N_SPARSE))
    n_pos = int(min(max(round(latent_p * N_QUAL), 1), N_QUAL - 1))
    cues = list(rng.choice(spec["qual_pos"], n_pos, replace=False)) + \
        list(rng.choice(spec["qual_neg"], N_QUAL - n_pos, replace=False))
    rng.shuffle(cues)
    evidence = {
        "sparse": f"Evidence: records are available for {N_SPARSE} {ref}; {k_sparse} of them {verb}.",
        "qualitative": "Evidence: no statistics on comparable cases are available. What is known:\n"
                       + "\n".join(f"- {c}" for c in cues),
    }
    iid = f"budget_{domain}_{idx:04d}"
    out = []
    for amb in ("sparse", "qualitative"):
        for lvl in BUFFER_LEVELS:
            buf = _fmt(bspec[lvl], fields)
            for bname in BUDGETS:
                parts = [context, evidence[amb], buf, exog, question, FORMAT_INSTRUCTION]
                out.append(BudgetCondition(
                    condition_id=f"{iid}__{amb}__{bname}__{lvl}", instance_id=iid, domain=domain,
                    ambiguity=amb, budget=bname, buffer_level=lvl,
                    normative_probability=round(_laplace(k_sparse, N_SPARSE), 4) if amb == "sparse" else None,
                    system=SYSTEM_PROMPT, prompt="\n\n".join(parts),
                    meta={"latent_p": latent_p, "buffer_type": btype}))
    return out


def budget_record(c: BudgetCondition) -> dict:
    r = asdict(c)
    r["prompt_hash"] = prompt_hash(c.system, c.prompt)
    return r
