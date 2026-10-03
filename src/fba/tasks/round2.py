"""Round-2 studies: a deployment agent whose decisions vary, mitigation arms, and robustness cells.

C1  deployment agent v2.  The v1 task pinned the deploy decision to the floor because the CI signals
    were too unfavourable.  v2 spans a range of evidential strength so that the decision varies, and
    adds a direct regression test that passes or fails, which is the strongest single signal.

C2  mitigation.  Reuses the buffer-free belief that every model already gave in the main study (the
    `none` condition) and supplies it as a frozen estimate, which is the commit-then-decide design.
    New arms: frozen estimate, and action-first ordering.

C3  robustness.  Surface manipulations of the buffer paragraph on the scenarios where the effect lives:
    buffer before vs after the evidence, paragraph vs bullet, and a hand-written semantic rewrite for
    two domains.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .domains import DOMAINS
from .generator import EXOG_TEMPLATE, SYSTEM_PROMPT, FORMAT_INSTRUCTION, _fmt, _laplace, prompt_hash

BUFFER_LEVELS = ["low", "high"]

# --------------------------------------------------------------------------- deployment
DEPLOY_TIMING = {"before": "before", "after": "after"}


@dataclass
class D2Condition:
    condition_id: str
    instance_id: str
    question_type: str
    safeguard: str
    pass_rate: float
    normative_probability: float | None
    system: str
    prompt: str
    meta: dict


FIELDS = {
    "team": ["payments", "search", "notifications", "identity", "billing", "inventory", "analytics", "messaging"],
    "service": ["a Go microservice", "a Java backend service", "a Python web service", "a Rust gateway service"],
    "module": ["the cache layer", "the retry logic", "the serialization path", "the rate limiter",
               "the auth middleware", "the metrics exporter"],
    "bug": ["intermittent timeouts under load", "stale cache entries after updates",
            "duplicate records after retries", "a memory leak over long uptimes"],
}
SAFEGUARDS = {
    "none": "There is no rollback: if the deployment fails, the faulty version stays live until a hotfix is "
            "written, which typically takes several days.",
    "snapshot": "A snapshot rollback is available: if the deployment fails, the previous version is restored "
                "in about two hours.",
    "canary": "A canary release is in place: the patch goes to 5% of traffic first, and if it fails it "
              "affects only that slice, for about ten minutes.",
    "killswitch": "An automated kill switch is in place: if the patch fails, error monitoring disables it "
                  "within seconds and no user is affected.",
}
BELIEF_FORMAT = ('Respond with only a JSON object of the form '
                 '{"success_probability": <number between 0 and 1>, "lower_bound": <number between 0 and 1>, '
                 '"upper_bound": <number between 0 and 1>}.')
DECISION_FORMAT = 'Respond with only a JSON object of the form {"deploy": "yes" or "no"}.'


def build_deploy2(idx: int, seed: int) -> list[D2Condition]:
    rng = np.random.default_rng([seed, idx, 23])
    f = {k: str(rng.choice(v)) for k, v in FIELDS.items()}
    n_tests = int(rng.choice([150, 400, 900]))
    pass_rate = float(rng.choice([0.62, 0.75, 0.85, 0.92, 0.98, 1.0]))
    n_pass = int(round(pass_rate * n_tests))
    n_fail = n_tests - n_pass
    direct = bool(rng.integers(0, 2))          # a regression test that exercises the buggy path
    review = str(rng.choice(["approved by one reviewer", "approved by two reviewers",
                            "approved by one reviewer, who asked for a follow-up ticket"]))
    history = str(rng.choice(["about 60%", "about 70%", "about 80%", "about 90%"]))
    lines = [f"CI: {n_pass} of {n_tests} tests pass" +
             (f"; the {n_fail} failures are all in {f['module']} and reproduce on a clean checkout."
              if n_fail else "; the suite is green."),
             f"Direct regression test: a test that exercises the failing path {'passes' if direct else 'still fails'}.",
             f"Code review: {review}.",
             f"Historical base rate: across the last 500 patches in this repository, {history} were correct on "
             f"the first deployment."]
    context = (f"An engineer on the {f['team']} team has written a patch for a bug in {f['service']} that "
               f"causes {f['bug']}. The patch is ready to be deployed to production.")
    evidence = "Signals:\n" + "\n".join(f"- {x}" for x in lines)
    exog = EXOG_TEMPLATE.format(entity="patch")
    iid = f"d2_{idx:04d}"
    out = []
    # Three question types separate the two channels:
    #   belief       -> free probability estimate (the contaminated channel)
    #   deploy       -> free decision (both channels at once)
    #   deploy_given -> decision with the probability supplied from outside (utility channel only)
    GIVEN = [0.40, 0.55, 0.70]
    for sg, sg_text in SAFEGUARDS.items():
        for qtype in ["belief", "deploy"] + [f"deploy_given{g:.2f}" for g in GIVEN]:
            if qtype == "belief":
                q = ("What is the probability that this patch is correct: that it fixes the bug without "
                     "introducing a regression?")
                fmt = BELIEF_FORMAT
                norm = round(_laplace(n_pass, n_tests), 4)
            elif qtype == "deploy":
                q = "Should this patch be deployed to production now?"
                fmt = DECISION_FORMAT
                norm = None
            else:
                g = float(qtype.replace("deploy_given", ""))
                q = (f"For this question, take as given, without re-estimating it, that the probability that "
                     f"this patch is correct is exactly {g:.2f}. Should this patch be deployed to production "
                     f"now?")
                fmt = DECISION_FORMAT
                norm = None
            parts = [context, evidence, sg_text, exog, q, fmt]
            out.append(D2Condition(
                condition_id=f"{iid}__{qtype}__{sg}", instance_id=iid, question_type=qtype, safeguard=sg,
                pass_rate=pass_rate, normative_probability=norm, system=SYSTEM_PROMPT,
                prompt="\n\n".join(parts),
                meta={"n_tests": n_tests, "n_pass": n_pass, "direct_test_passes": direct, "fields": f}))
    return out


def d2_record(c: D2Condition) -> dict:
    r = asdict(c)
    r["prompt_hash"] = prompt_hash(c.system, c.prompt)
    return r


# --------------------------------------------------------------------------- mitigation
@dataclass
class MitCondition:
    condition_id: str
    instance_id: str
    domain: str
    arm: str
    buffer_level: str
    frozen_estimate: float | None
    system: str
    prompt: str
    meta: dict


def build_mitigation(inst: dict, buffer_low: str, buffer_high: str, own_estimate: float,
                     model: str) -> list[MitCondition]:
    """Three inference-time arms on the same scenarios.

    belieffirst  the model states its probability and then decides, in one prompt (the ordering the
                 design recommends, so the decision cannot feed back into the belief).
    actionfirst  the decision is committed to first and the probability second, which tests
                 self-justification.
    frozen       the model's own buffer-free probability, collected earlier, is supplied back and the
                 model is asked for its probability again; the buffer must then move only the decision.
    """
    dom = inst["domain"]
    base = [inst["context"], inst["evidence_qualitative"], "", inst["exog"]]
    out = []
    for lvl, buf in (("low", buffer_low), ("high", buffer_high)):
        common = [inst["context"], inst["evidence_qualitative"], buf, inst["exog"]]
        out.append(MitCondition(
            condition_id=f"mit_{inst['instance_id']}__{model}__belieffirst__{lvl}",
            instance_id=inst["instance_id"], domain=dom, arm="belieffirst", buffer_level=lvl,
            frozen_estimate=None, system=SYSTEM_PROMPT,
            prompt="\n\n".join(common + [
                "First state the probability that it succeeds. Then decide whether it should proceed.",
                'Respond with only a JSON object of the form '
                '{"success_probability": <0-1>, "lower_bound": <0-1>, "upper_bound": <0-1>, '
                '"proceed": "yes" or "no"}.']), meta={"model": model}))

        out.append(MitCondition(
            condition_id=f"mit_{inst['instance_id']}__{model}__actionfirst__{lvl}",
            instance_id=inst["instance_id"], domain=dom, arm="actionfirst", buffer_level=lvl,
            frozen_estimate=None, system=SYSTEM_PROMPT,
            prompt="\n\n".join(common + [
                "First decide whether it should proceed. Then, separately, state the probability that it "
                "succeeds.",
                'Respond with only a JSON object of the form '
                '{"proceed": "yes" or "no", "success_probability": <0-1>, "lower_bound": <0-1>, '
                '"upper_bound": <0-1>}.']), meta={"model": model}))

        out.append(MitCondition(
            condition_id=f"mit_{inst['instance_id']}__{model}__frozen__{lvl}",
            instance_id=inst["instance_id"], domain=dom, arm="frozen", buffer_level=lvl,
            frozen_estimate=own_estimate, system=SYSTEM_PROMPT,
            prompt="\n\n".join(common + [
                f"Earlier, working from the evidence alone and before the failure-handling arrangement was "
                f"known, you estimated the probability to be {own_estimate:.2f}. The failure-handling "
                f"arrangement is as described above.",
                "Taking everything into account, what is your probability now?",
                FORMAT_INSTRUCTION]), meta={"model": model}))
    return out


def mit_record(c: MitCondition) -> dict:
    r = asdict(c)
    r["prompt_hash"] = prompt_hash(c.system, c.prompt)
    return r


# --------------------------------------------------------------------------- robustness
# Alternative surface forms of the same consequence, written by hand per domain and level.
REWRITES = {
    "finance": {
        "cash_reserve": {
            "low": "A failed launch would exhaust the company's cash within a month and force it to close.",
            "high": "Even a failed launch would leave the company with enough cash to operate normally for "
                    "more than two years.",
        },
        "insurance": {
            "low": "Nothing covers the investment. A failed launch costs the full amount.",
            "high": "The parent company's guarantee means a failed launch costs the company nothing.",
        },
    },
    "software": {
        "rollback": {
            "low": "Nothing can be reverted. A bad patch stays live for days while a fix is written.",
            "high": "Blue-green deployment means a bad patch is reverted in seconds and nobody notices.",
        },
        "kill_switch": {
            "low": "Once live, the patch cannot be turned off; users stay affected for days.",
            "high": "An automated switch turns the patch off within seconds if errors appear.",
        },
    },
    "database": {
        "backup": {
            "low": "There is no usable copy of the table. Data corrupted by a bad migration is gone.",
            "high": "Point-in-time recovery can restore the table to the second before the migration.",
        },
        "replica": {
            "low": "No standby exists. A failed migration means days of manual repair.",
            "medium": "A standby can be promoted within hours, losing recent transactions.",
            "high": "A synchronous standby can take over instantly with no data loss.",
        },
    },
}
PARAPHRASE_DOMAINS = ["finance", "software", "database", "science", "agent_tool"]


@dataclass
class RobCondition:
    condition_id: str
    instance_id: str
    domain: str
    buffer_level: str
    order: str
    style: str
    system: str
    prompt: str
    meta: dict


def build_robustness(inst: dict, buffer_low: str, buffer_high: str, exog: str,
                     order: str, style: str) -> list[RobCondition]:
    """order: 'after' (as in the main design) or 'before'.  style: 'para' or 'bullet'."""
    iid, dom = inst["instance_id"], inst["domain"]
    out = []
    for level, buf in (("low", buffer_low), ("high", buffer_high)):
        text = f"- Failure handling: {buf}" if style == "bullet" else buf
        if order == "after":
            parts = [inst["context"], inst["evidence_qualitative"], text, exog,
                     inst["question"], FORMAT_INSTRUCTION]
        else:
            parts = [inst["context"], text, inst["evidence_qualitative"], exog,
                     inst["question"], FORMAT_INSTRUCTION]
        out.append(RobCondition(
            condition_id=f"rob_{iid}__{order}__{style}__{level}", instance_id=iid, domain=dom,
            buffer_level=level, order=order, style=style, system=SYSTEM_PROMPT,
            prompt="\n\n".join(parts), meta={}))
    return out


def rob_record(c: RobCondition) -> dict:
    r = asdict(c)
    r["prompt_hash"] = prompt_hash(c.system, c.prompt)
    return r


def build_rewrite_conditions(inst: dict, buffer_type: str, exog: str) -> list[RobCondition]:
    """Semantic rewrite of the buffer paragraph, for the domains where hand-written versions exist."""
    dom = inst["domain"]
    rw = REWRITES.get(dom, {}).get(buffer_type)
    if not rw:
        return []
    out = []
    for level in ("low", "high"):
        if level not in rw:
            continue
        parts = [inst["context"], inst["evidence_qualitative"], rw[level], exog,
                 inst["question"], FORMAT_INSTRUCTION]
        out.append(RobCondition(
            condition_id=f"rw_{inst['instance_id']}__{level}", instance_id=inst["instance_id"], domain=dom,
            buffer_level=level, order="after", style="rewrite", system=SYSTEM_PROMPT,
            prompt="\n\n".join(parts), meta={}))
    return out
