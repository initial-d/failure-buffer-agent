"""Study A: continuous evidence-sample gradient (epistemic slack made exogenous).

One latent success probability per scenario; the number of comparable cases shown varies over
{0 (qualitative only), 2, 4, 8, 16, 32, 64, 100}.  Every scenario appears at three buffer levels
under the random-allocation framing, so the sample size n is the only thing that changes the amount
of information, and the buffer is the only thing that changes the consequence of failure.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .domains import DOMAINS
from .generator import EXOG_TEMPLATE, SYSTEM_PROMPT, FORMAT_INSTRUCTION, _fmt, _laplace, prompt_hash

N_GRID = [0, 2, 4, 8, 16, 32, 64, 100]
BUFFER_LEVELS = ["low", "high"]


@dataclass
class GradientCondition:
    condition_id: str
    instance_id: str
    domain: str
    n_cases: int
    buffer_level: str
    normative_probability: float
    exog: bool
    system: str
    prompt: str
    meta: dict


def build_gradient_instance(domain: str, idx: int, seed: int) -> list[GradientCondition]:
    spec = DOMAINS[domain]
    rng = np.random.default_rng([seed, idx, sum(map(ord, domain)), 7])
    fields = {k: str(rng.choice(v)) for k, v in spec["fields"].items()}
    latent_p = float(np.round(rng.uniform(0.30, 0.75), 2))
    btype = str(rng.choice(sorted(spec["buffers"].keys())))
    bspec = spec["buffers"][btype]
    context = _fmt(spec["context"], fields)
    target = _fmt(spec["target"], fields)
    question = f"What is the probability that {target}?"
    exog = EXOG_TEMPLATE.format(entity=spec["entity"])
    ref, verb = spec["ref_class"], spec["success_verb"]

    iid = f"grad_{domain}_{idx:04d}"
    out = []
    for n in N_GRID:
        for lvl in BUFFER_LEVELS:
            buf = _fmt(bspec[lvl], fields)
            if n == 0:
                n_pos = int(min(max(round(latent_p * 5), 1), 4))
                cues = list(rng.choice(spec["qual_pos"], n_pos, replace=False)) + \
                    list(rng.choice(spec["qual_neg"], 5 - n_pos, replace=False))
                ev = ("Evidence: no statistics on comparable cases are available. What is known:\n"
                      + "\n".join(f"- {c}" for c in cues))
                norm = None
            else:
                k = int(round(latent_p * n))
                ev = (f"Evidence: records are available for {n} {ref}; {k} of them {verb}.")
                norm = round(_laplace(k, n), 4)
            parts = [context, ev, buf, exog, question, FORMAT_INSTRUCTION]
            out.append(GradientCondition(
                condition_id=f"{iid}__n{n}__{lvl}", instance_id=iid, domain=domain, n_cases=n,
                buffer_level=lvl, normative_probability=norm, exog=True, system=SYSTEM_PROMPT,
                prompt="\n\n".join(parts),
                meta={"latent_p": latent_p, "buffer_type": btype, "fields": fields}))
    return out


def gradient_record(c: GradientCondition) -> dict:
    r = asdict(c)
    r["prompt_hash"] = prompt_hash(c.system, c.prompt)
    return r
