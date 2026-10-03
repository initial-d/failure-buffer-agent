"""Study 3: Buffer-Induced Feedback Suppression (BFS) with paired replay.

Each base instance is a *series* of exchangeable trials of the same item (e.g. one patch rolled
out cluster by cluster).  The agent receives prior evidence, a buffer describing the consequence
of each failed trial, and a FIXED observed outcome sequence that is identical across buffer
conditions (paired replay).  q_before is elicited from a prompt with no observations; q_after
from the same prompt plus observations.  Both are single-turn, so neither can anchor on the other.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

from .domains import DOMAINS
from .generator import SYSTEM_PROMPT, _laplace, prompt_hash

N_SPARSE = 6
N_QUAL_CUES = 5
BUFFER_LEVELS_S3 = ["none", "low", "high"]

# (label, n_success, n_failure)
FEEDBACK = {
    "neg_1of1": (0, 1),
    "neg_3of5": (2, 3),
    "neg_8of10": (2, 8),
    "pos_1of1": (1, 0),
    "pos_4of5": (4, 1),
}

FORMAT_S3 = (
    'Respond with only a JSON object of the form '
    '{"success_probability": <number between 0 and 1>, "lower_bound": <number between 0 and 1>, '
    '"upper_bound": <number between 0 and 1>}, where lower_bound and upper_bound give your 90% '
    'credible interval for the probability.'
)

SERIES = {
    "software": {
        "base": "software",
        "context": "An engineer on the {team} team has written a patch for bug {bugid} in {service}. The bug causes "
                   "{symptom}. The patch is being rolled out one cluster at a time to 12 independent regional "
                   "clusters that run identical code and serve similar traffic.",
        "trial": "cluster", "trials": "clusters",
        "target_next": "the patch resolves the bug without introducing a regression on the next cluster",
        "ok": "the patch worked (bug resolved, no regression)",
        "fail": "the patch failed (the bug persisted or a regression appeared)",
        "buffers": {
            "low": "There is no rollback for these clusters: every cluster on which the patch fails stays broken "
                   "for several days until a new fix is shipped, with customer-visible impact.",
            "high": "The clusters use blue-green deployment: on any cluster where the patch fails, traffic is "
                    "switched back to the previous version within seconds, with no customer-visible impact.",
        },
    },
    "agent_tool": {
        "base": "agent_tool",
        "context": "An autonomous AI agent is working through a series of 12 similar machines in {system}, one machine "
                   "at a time. On each machine it runs {tool} in order to {goal}.",
        "trial": "machine", "trials": "machines",
        "target_next": "running the tool achieves its intended objective on the next machine",
        "ok": "the tool achieved its intended objective",
        "fail": "the tool did not achieve its intended objective",
        "buffers": {
            "low": "The machines have no checkpoints and the tool's changes are irreversible: on every machine where "
                   "the tool fails, the resulting damage must be repaired by hand, which takes days.",
            "high": "Every machine is continuously checkpointed: on any machine where the tool fails, the machine is "
                    "restored automatically within seconds to its exact prior state, with no lasting effect.",
        },
    },
    "finance": {
        "base": "finance",
        "context": "{company}, a company with about {size} employees, is rolling out {product} one market at a time "
                   "across 15 comparable regional markets.",
        "trial": "regional launch", "trials": "regional launches",
        "target_next": "the next regional launch reaches its first-quarter sales target",
        "ok": "the regional launch reached its first-quarter sales target",
        "fail": "the regional launch missed its first-quarter sales target",
        "buffers": {
            "low": "Each regional launch that misses its target causes a large, uninsured loss; the company's remaining "
                   "cash could absorb only one or two more such misses before the company would have to shut down.",
            "high": "Every regional launch is fully insured by the parent company: any region that misses its target "
                    "is reimbursed in full, so misses have no effect on the company's finances or survival.",
        },
    },
}


@dataclass
class S3Condition:
    condition_id: str
    instance_id: str
    domain: str
    ambiguity: str
    buffer_level: str
    feedback: Optional[str]      # None => prior-only (q_before)
    n_success: int
    n_failure: int
    sequence: list
    normative_probability: Optional[float]
    system: str
    prompt: str
    meta: dict = field(default_factory=dict)


def build_s3_instance(domain: str, idx: int, seed: int) -> list[S3Condition]:
    ser = SERIES[domain]
    spec = DOMAINS[ser["base"]]
    rng = np.random.default_rng([seed, idx, sum(map(ord, domain)), 3])
    fields = {k: str(rng.choice(v)) for k, v in spec["fields"].items()}
    latent_p = float(np.round(rng.uniform(0.45, 0.85), 2))
    k_sparse = int(round(latent_p * N_SPARSE))
    n_pos = int(min(max(round(latent_p * N_QUAL_CUES), 1), N_QUAL_CUES - 1))
    cues = list(rng.choice(spec["qual_pos"], n_pos, replace=False)) + \
        list(rng.choice(spec["qual_neg"], N_QUAL_CUES - n_pos, replace=False))
    rng.shuffle(cues)
    ref, verb = spec["ref_class"], spec["success_verb"]
    if domain == "finance":
        ref, verb = "comparable regional launches by similar firms", "reached their first-quarter sales target"
    evidence = {
        "sparse": f"Prior evidence: records are available for {N_SPARSE} {ref}; {k_sparse} of them {verb}.",
        "qualitative": "Prior evidence: no statistics on comparable cases are available. What is known:\n"
                       + "\n".join(f"- {c}" for c in cues),
    }
    # One fixed outcome order per feedback type (paired across buffers and ambiguity levels).
    sequences = {}
    for fb, (s, f) in FEEDBACK.items():
        seq = [1] * s + [0] * f
        rng.shuffle(seq)
        if fb.startswith("neg") and seq[-1] == 1:      # end on the disconfirming observation
            j = seq.index(0)
            seq[j], seq[-1] = seq[-1], seq[j]
        sequences[fb] = seq

    context = ser["context"].format(**fields)
    question = f"What is the probability that {ser['target_next'].format(**fields)}?"
    iid = f"s3_{domain}_{idx:04d}"
    out = []
    for amb in ("sparse", "qualitative"):
        for lvl in BUFFER_LEVELS_S3:
            buf = ser["buffers"][lvl].format(**fields) if lvl != "none" else ""
            for fb in [None] + list(FEEDBACK):
                parts = [context, evidence[amb]]
                if buf:
                    parts.append(buf)
                s = f = 0
                seq = []
                if fb:
                    seq = sequences[fb]
                    s, f = sum(seq), len(seq) - sum(seq)
                    lines = [f"- {ser['trial'].capitalize()} {i + 1}: {ser['ok'] if y else ser['fail']}."
                             for i, y in enumerate(seq)]
                    parts.append(f"Outcomes observed so far ({len(seq)} "
                                 f"{ser['trials'] if len(seq) > 1 else ser['trial']}):\n" + "\n".join(lines))
                parts += [question, FORMAT_S3]
                norm = round(_laplace(k_sparse + s, N_SPARSE + s + f), 4) if amb == "sparse" else None
                out.append(S3Condition(
                    condition_id=f"{iid}__{amb}__{lvl}__{fb or 'prior'}", instance_id=iid, domain=domain,
                    ambiguity=amb, buffer_level=lvl, feedback=fb, n_success=s, n_failure=f, sequence=seq,
                    normative_probability=norm, system=SYSTEM_PROMPT, prompt="\n\n".join(parts),
                    meta={"latent_p": latent_p, "k_sparse": k_sparse, "fields": fields}))
    return out


def s3_record(c: S3Condition) -> dict:
    r = asdict(c)
    r["prompt_hash"] = prompt_hash(c.system, c.prompt)
    return r
