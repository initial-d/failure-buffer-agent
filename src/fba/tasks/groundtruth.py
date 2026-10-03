"""Round-3 studies on the ground-truth patch task.

Items: MBPP candidate patches that pass the one test shown to the forecaster; the outcome is whether the
patch passes the full hidden test suite, established by executing it.  Payoffs are stated so that a
decision has a defined value: deploying a correct patch earns +1, deploying an incorrect one costs L(s),
not deploying earns 0.  L depends on the safeguard, so the rational deploy threshold p* = L/(1+L) moves with
the safeguard while the probability of correctness does not.

Arms (per item x safeguard):
  belief_nobuf   probability only, no safeguard sentence (stage 1 of the two-stage design)
  vanilla        safeguard + payoffs, asks probability then decision, one prompt
  instruction    as vanilla plus an explicit note that the safeguard does not affect correctness
  twostage       stage 2: safeguard + payoffs + the model's own stage-1 probability, decision only
  given_pXX      safeguard + payoffs + an experimenter-assigned probability, decision only
                 (randomised mediator)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .generator import EXOG_TEMPLATE, SYSTEM_PROMPT, prompt_hash

SAFEGUARDS = {
    "none": ("There is no rollback: if a faulty version is deployed, it stays live until a hotfix is written, "
             "which typically takes several days.", 4.0),
    "snapshot": ("A snapshot rollback is available: if a faulty version is deployed, the previous version is "
                 "restored in about two hours.", 2.0),
    "canary": ("A canary release is in place: a faulty version reaches only 5% of traffic, for about ten "
               "minutes, before it is withdrawn.", 1.0),
    "killswitch": ("An automated kill switch is in place: if a faulty version is deployed, error monitoring "
                   "disables it within seconds and no user is affected.", 0.25),
}
GIVEN_P = [0.40, 0.50, 0.60, 0.70]
INDEP = ("Note: the deployment safeguard only affects what happens after a faulty version is deployed; it "
         "has no effect on whether this implementation is correct.")


def payoff_text(loss):
    return (f"Scoring for the deploy decision: deploying a correct implementation earns +1 point; deploying an "
            f"incorrect one costs {loss:g} points under this safeguard; not deploying earns 0.")


def item_context(c):
    return (f"A service needs a Python function for the following task:\n\n{c['text']}\n\n"
            f"A candidate implementation has been written:\n\n```python\n{c['code']}\n```\n\n"
            f"CI result: the one available unit test passes:\n{c['shown_test']}")


Q_PROB = ("What is the probability that this implementation is correct, that is, that it passes the full "
          "hidden test suite for the task?")
FMT_PROB = ('Respond with only a JSON object of the form {"success_probability": <number between 0 and 1>, '
            '"lower_bound": <number between 0 and 1>, "upper_bound": <number between 0 and 1>}.')
FMT_BOTH = ('First state the probability, then decide. Respond with only a JSON object of the form '
            '{"success_probability": <number between 0 and 1>, "lower_bound": <number between 0 and 1>, '
            '"upper_bound": <number between 0 and 1>, "deploy": "yes" or "no"}.')
FMT_DEC = 'Respond with only a JSON object of the form {"deploy": "yes" or "no"}.'


def select_items(cands_path: Path, n_fail=86, n_pass=86, seed=7):
    cands = [json.loads(l) for l in open(cands_path)]
    cands = [c for c in cands if c["shown_pass"] and c["code"] and len(c["code"]) < 2500]
    fails = [c for c in cands if not c["hidden_pass"]]
    passes = [c for c in cands if c["hidden_pass"]]
    rng = random.Random(seed)
    rng.shuffle(fails)
    rng.shuffle(passes)
    return fails[:n_fail] + passes[:n_pass]


def rec(cid, iid, arm, sg, truth, prompt, **extra):
    r = {"condition_id": cid, "instance_id": iid, "arm": arm, "safeguard": sg, "truth": truth,
         "system": SYSTEM_PROMPT, "prompt": prompt, **extra}
    r["prompt_hash"] = prompt_hash(SYSTEM_PROMPT, prompt)
    return r


def stage1(items):
    out = []
    for c in items:
        iid = f"gt_{c['task_id']}"
        p = "\n\n".join([item_context(c), Q_PROB, FMT_PROB])
        out.append(rec(f"{iid}__belief_nobuf__-", iid, "belief_nobuf", "-", c["hidden_pass"], p))
    return out


def stage_main(items, own=None):
    """All arms except two-stage need nothing from stage 1; two-stage needs `own[model][iid]`."""
    exog = EXOG_TEMPLATE.format(entity="implementation")
    out = []
    for c in items:
        iid = f"gt_{c['task_id']}"
        ctx = item_context(c)
        for sg, (sg_text, loss) in SAFEGUARDS.items():
            pay = payoff_text(loss)
            base = [ctx, sg_text, exog, pay]
            out.append(rec(f"{iid}__vanilla__{sg}", iid, "vanilla", sg, c["hidden_pass"],
                           "\n\n".join(base + [Q_PROB, FMT_BOTH]), loss=loss))
            out.append(rec(f"{iid}__instruction__{sg}", iid, "instruction", sg, c["hidden_pass"],
                           "\n\n".join(base + [INDEP, Q_PROB, FMT_BOTH]), loss=loss))
            for g in GIVEN_P:
                q = (f"For this decision, take as given, without re-estimating it, that the probability that "
                     f"this implementation is correct is exactly {g:.2f}. Should it be deployed?")
                out.append(rec(f"{iid}__given{int(g * 100)}__{sg}", iid, f"given{int(g * 100)}", sg,
                               c["hidden_pass"], "\n\n".join(base + [q, FMT_DEC]), loss=loss, given_p=g))
    return out


def stage2(items, own_p: dict):
    """Two-stage: the decision stage sees the safeguard and the payoffs plus the number produced by a
    buffer-free belief stage. It never sees a request to estimate a probability."""
    exog = EXOG_TEMPLATE.format(entity="implementation")
    out = []
    for c in items:
        iid = f"gt_{c['task_id']}"
        if iid not in own_p:
            continue
        p0 = own_p[iid]
        for sg, (sg_text, loss) in SAFEGUARDS.items():
            q = (f"An independent assessment, made from the task, the code and the CI result alone, puts the "
                 f"probability that this implementation is correct at {p0:.2f}. Take that probability as given. "
                 f"Should it be deployed?")
            out.append(rec(f"{iid}__twostage__{sg}", iid, "twostage", sg, c["hidden_pass"],
                           "\n\n".join([item_context(c), sg_text, exog, payoff_text(loss), q, FMT_DEC]),
                           loss=loss, stage1_p=p0))
    return out
