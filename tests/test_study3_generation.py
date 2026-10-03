import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.tasks.feedback import FEEDBACK, SERIES, build_s3_instance  # noqa: E402

CONDS = [build_s3_instance(d, i, 7) for d in SERIES for i in range(10)]


@pytest.mark.parametrize("conds", CONDS, ids=lambda c: c[0].instance_id)
def test_replay_sequences_identical_across_buffers(conds):
    for amb in ("sparse", "qualitative"):
        for fb in FEEDBACK:
            seqs = {tuple(c.sequence) for c in conds if c.ambiguity == amb and c.feedback == fb}
            assert len(seqs) == 1
    # identical sequence across ambiguity levels as well
    for fb in FEEDBACK:
        assert len({tuple(c.sequence) for c in conds if c.feedback == fb}) == 1


@pytest.mark.parametrize("conds", CONDS, ids=lambda c: c[0].instance_id)
def test_prompts_differ_only_in_buffer_paragraph(conds):
    for amb in ("sparse", "qualitative"):
        for fb in [None] + list(FEEDBACK):
            grp = {c.buffer_level: c.prompt.split("\n\n") for c in conds if c.ambiguity == amb and c.feedback == fb}
            none, low, high = grp["none"], grp["low"], grp["high"]
            assert low[:2] == high[:2] == none[:2] and low[3:] == high[3:] == none[2:]
            assert low[2] != high[2]


@pytest.mark.parametrize("conds", CONDS, ids=lambda c: c[0].instance_id)
def test_counts_and_normative(conds):
    for c in conds:
        if c.feedback:
            assert (c.n_success, c.n_failure) == FEEDBACK[c.feedback]
            if c.feedback.startswith("neg"):
                assert c.sequence[-1] == 0
        norms = {}
    for amb in ("sparse",):
        for fb in [None] + list(FEEDBACK):
            assert len({c.normative_probability for c in conds if c.ambiguity == amb and c.feedback == fb}) == 1
