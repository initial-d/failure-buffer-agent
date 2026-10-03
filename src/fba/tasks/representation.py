"""Representation identification: hold the evidence content fixed and vary only how a number appears.

A sample-size gradient confounds four things at once (n=0 vs n>=2 changes the amount of
evidence, the representation, whether anything is computable, and whether a number is present at all).
This experiment separates them at a fixed latent rate:

  qual          five qualitative cues, no number anywhere
  n1            one comparable case (k of 1)
  n2            two comparable cases
  n4            four comparable cases
  freq          "58 of 100 comparable cases"
  pct           "58% of comparable cases"   (same content as freq, different form)
  irrelev_num   cues only, plus an irrelevant identifying number
  equiv_frac    "about 6 in 10"             (verbalised fraction, no count)

All conditions keep the qualitative cues as well, so the only thing that moves is the numeric overlay.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .domains import DOMAINS
from .generator import EXOG_TEMPLATE, SYSTEM_PROMPT, FORMAT_INSTRUCTION, _fmt, _laplace, prompt_hash

REPS = ["qual", "equiv_frac", "irrelev_num", "n1", "n2", "n4", "pct", "freq"]
BUFFER_LEVELS = ["low", "high"]
N_QUAL = 5


@dataclass
class RepCondition:
    condition_id: str
    instance_id: str
    domain: str
    rep: str
    buffer_level: str
    normative_probability: float | None
    system: str
    prompt: str
    meta: dict


def build_rep_instance(domain: str, idx: int, seed: int) -> list[RepCondition]:
    spec = DOMAINS[domain]
    rng = np.random.default_rng([seed, idx, sum(map(ord, domain)), 29])
    fields = {k: str(rng.choice(v)) for k, v in spec["fields"].items()}
    latent_p = float(np.round(rng.uniform(0.30, 0.75), 2))
    btype = str(rng.choice(sorted(spec["buffers"].keys())))
    bspec = spec["buffers"][btype]
    context = _fmt(spec["context"], fields)
    target = _fmt(spec["target"], fields)
    question = f"What is the probability that {target}?"
    exog = EXOG_TEMPLATE.format(entity=spec["entity"])
    ref, verb = spec["ref_class"], spec["success_verb"]

    n_pos = int(min(max(round(latent_p * N_QUAL), 1), N_QUAL - 1))
    cues = list(rng.choice(spec["qual_pos"], n_pos, replace=False)) + \
        list(rng.choice(spec["qual_neg"], N_QUAL - n_pos, replace=False))
    rng.shuffle(cues)
    cue_block = "What is known:\n" + "\n".join(f"- {c}" for c in cues)
    k1 = int(round(latent_p))
    k2 = int(round(latent_p * 2))
    k4 = int(round(latent_p * 4))
    k100 = int(round(latent_p * 100))
    pct_val = int(round(latent_p * 100))

    def numeric_overlay(rep: str) -> tuple[str, float | None]:
        if rep == "qual":
            return "", None
        if rep == "equiv_frac":
            words = {1: "very few", 2: "about one in two", 3: "about three in five", 4: "about two in three"}
            return (f"One experienced colleague's rough impression: {words.get(int(round(latent_p * 6)), 'about half')} "
                    f"of comparable cases succeeded.", None)
        if rep == "irrelev_num":
            return (f"The records carry an internal filing number, {int(rng.integers(10000, 99999))}, "
                    f"assigned by the archive system.", None)
        if rep == "n1":
            # one comparable case only; worded so that no plural is needed
            outcome = "and it turned out well" if k1 else "and it did not turn out well"
            return (f"There is a record of one earlier case of the same kind {outcome}.", None)
        if rep == "n2":
            return f"Records are available for two comparable cases; {k2} of them {verb}.", None
        if rep == "n4":
            return f"Records are available for four comparable cases; {k4} of them {verb}.", None
        if rep == "pct":
            return f"A summary of comparable cases reports a success rate of {pct_val}%.", None
        if rep == "freq":
            return f"Records are available for 100 {ref}; {k100} of them {verb}.", None
        raise ValueError(rep)

    iid = f"rep_{domain}_{idx:04d}"
    out = []
    for rep in REPS:
        overlay, norm = numeric_overlay(rep)
        for lvl in BUFFER_LEVELS:
            buf = _fmt(bspec[lvl], fields)
            parts = [context, "Evidence: " + cue_block]
            if overlay:
                parts.append(overlay)
            parts += [buf, exog, question, FORMAT_INSTRUCTION]
            out.append(RepCondition(
                condition_id=f"{iid}__{rep}__{lvl}", instance_id=iid, domain=domain, rep=rep,
                buffer_level=lvl, normative_probability=norm, system=SYSTEM_PROMPT,
                prompt="\n\n".join(parts),
                meta={"latent_p": latent_p, "buffer_type": btype, "k100": k100, "pct": pct_val}))
    return out


def rep_record(c: RepCondition) -> dict:
    r = asdict(c)
    r["prompt_hash"] = prompt_hash(c.system, c.prompt)
    return r
