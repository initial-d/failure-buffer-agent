"""Study C: software-deployment agent.

The agent reviews one patch (test results, logs, regression signals, history) and must give both a
probability that the patch is correct and a deploy decision.  The buffer is the deployment safeguard,
which changes what a failed deploy costs and nothing else.  Two prompts per item keeps the two
questions from anchoring on each other.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .generator import EXOG_TEMPLATE, SYSTEM_PROMPT, _laplace, prompt_hash

SAFEGUARDS = {
    "none": ("There is no rollback: if the deployment fails, the faulty version stays live until a hotfix "
             "is written, which typically takes several days."),
    "snapshot": ("A snapshot rollback is available: if the deployment fails, the previous version is "
                 "restored in about two hours."),
    "canary": ("A canary release is in place: the patch goes to 5% of traffic first, and if it fails it "
               "affects only that slice, for about ten minutes."),
    "killswitch": ("An automated kill switch is in place: if the patch fails, error monitoring disables it "
                   "within seconds and no user is affected."),
}
DEPLOY_FORMAT = ('Respond with only a JSON object of the form '
                 '{"deploy": "yes" or "no", "reason": "<one short sentence>"}.')
BELIEF_FORMAT = ('Respond with only a JSON object of the form '
                 '{"success_probability": <number between 0 and 1>, "lower_bound": <number between 0 and 1>, '
                 '"upper_bound": <number between 0 and 1>}.')

FIELDS = {
    "team": ["payments", "search", "notifications", "identity", "billing", "inventory", "analytics", "messaging"],
    "service": ["a Go microservice", "a Java backend service", "a Python web service", "a Rust gateway service"],
    "bugid": [f"#{x}" for x in range(1000, 9999, 37)],
    "test_total": ["400", "900", "1500", "2200"],
    "history": ["about 60%", "about 65%", "about 70%", "about 75%"],
}


@dataclass
class DeployCondition:
    condition_id: str
    instance_id: str
    question_type: str          # belief | deploy
    safeguard: str
    normative_probability: float | None
    system: str
    prompt: str
    meta: dict


def build_deploy_instance(idx: int, seed: int) -> list[DeployCondition]:
    rng = np.random.default_rng([seed, idx, 17])
    f = {k: str(rng.choice(v)) for k, v in FIELDS.items()}
    latent = float(np.round(rng.uniform(0.45, 0.85), 2))
    n_tests = int(f["test_total"])
    n_pass = int(round(latent * n_tests))
    failing_module = str(rng.choice(["the cache layer", "the retry logic", "the serialization path",
                                     "the rate limiter", "the auth middleware", "the metrics exporter"]))
    signals = [
        f"CI status: {n_pass} of {n_tests} tests pass; the {n_tests - n_pass} failures are all in "
        f"{failing_module} and reproduce on a clean checkout.",
        f"Code review: one approval; the reviewer flagged {failing_module} as untested.",
        f"Historical base rate: across the last 500 patches in this repository, {f['history']} were correct "
        f"on the first deployment.",
        "Production logs: no new error signature appears in staging after the last similar change.",
    ]
    rng.shuffle(signals[2:])
    context = (f"An engineer on the {f['team']} team has written a patch for bug {f['bugid']} in "
               f"{f['service']}. The patch is ready to be deployed to production.")
    evidence = "Signals:\n" + "\n".join(f"- {s}" for s in signals)
    exog = EXOG_TEMPLATE.format(entity="patch")
    iid = f"deploy_{idx:04d}"
    out = []
    for sg, sg_text in SAFEGUARDS.items():
        for qtype in ("belief", "deploy"):
            if qtype == "belief":
                q = "What is the probability that this patch is correct (it fixes the bug without introducing a regression)?"
                fmt = BELIEF_FORMAT
            else:
                q = "Should this patch be deployed to production now?"
                fmt = DEPLOY_FORMAT
            parts = [context, evidence, sg_text, exog, q, fmt]
            out.append(DeployCondition(
                condition_id=f"{iid}__{qtype}__{sg}", instance_id=iid, question_type=qtype, safeguard=sg,
                normative_probability=round(_laplace(n_pass, n_tests), 4) if qtype == "belief" else None,
                system=SYSTEM_PROMPT, prompt="\n\n".join(parts),
                meta={"latent_p": latent, "n_tests": n_tests, "n_pass": n_pass, "fields": f}))
    return out


def deploy_record(c: DeployCondition) -> dict:
    r = asdict(c)
    r["prompt_hash"] = prompt_hash(c.system, c.prompt)
    return r
