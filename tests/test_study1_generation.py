import difflib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.tasks.domains import DOMAINS  # noqa: E402
from fba.tasks.generator import (AMBIGUITY_LEVELS, BUFFER_LEVELS, build_conditions,  # noqa: E402
                                 make_base_instance)
from fba.parsing import parse_belief  # noqa: E402

SEED = 20261002
INSTANCES = [make_base_instance(d, i, SEED) for d in DOMAINS for i in range(20)]


def _by_key(conds):
    return {(c.ambiguity, c.condition_type, c.buffer_level, c.explicit_independence, c.condition_id): c for c in conds}


@pytest.mark.parametrize("inst", INSTANCES, ids=lambda x: x.instance_id)
def test_paired_prompts_differ_only_in_buffer_slots(inst):
    conds = build_conditions(inst)
    for amb in AMBIGUITY_LEVELS:
        group = [c for c in conds if c.ambiguity == amb and c.condition_type in ("main", "none")]
        ref = group[0]
        for c in group[1:]:
            for slot in ("context", "evidence", "question", "format"):
                assert c.slots[slot] == ref.slots[slot], (c.condition_id, slot)
            # reconstructing the prompt without buffer/independence slots gives identical text
            strip = lambda s: "\n\n".join(p for p in s.prompt.split("\n\n")
                                          if p not in (s.slots["buffer"], s.slots["independence"]))
            assert strip(c) == strip(ref)


@pytest.mark.parametrize("inst", INSTANCES, ids=lambda x: x.instance_id)
def test_normative_probability_identical_across_buffers(inst):
    conds = build_conditions(inst)
    for amb in AMBIGUITY_LEVELS:
        vals = {c.normative_probability for c in conds
                if c.ambiguity == amb and c.condition_type in ("main", "none")}
        assert len(vals) == 1
        flags = {c.buffer_affects_success_probability for c in conds
                 if c.ambiguity == amb and c.condition_type in ("main", "none")}
        assert flags == {False}


@pytest.mark.parametrize("inst", INSTANCES, ids=lambda x: x.instance_id)
def test_buffer_levels_actually_differ_and_type_constant(inst):
    conds = [c for c in build_conditions(inst) if c.condition_type == "main"]
    texts = {c.buffer_level: c.slots["buffer"] for c in conds if not c.explicit_independence}
    assert len(set(texts.values())) == 3
    assert {c.buffer_type for c in conds} == {inst.buffer_type}


def test_controls_have_expected_normative_direction():
    for inst in INSTANCES:
        cs = {c.condition_type: c for c in build_conditions(inst) if c.ambiguity == "sparse"}
        sparse_none = [c for c in build_conditions(inst) if c.condition_type == "none" and c.ambiguity == "sparse"][0]
        assert cs["positive_control"].normative_probability > sparse_none.normative_probability
        assert cs["retry_control"].normative_probability > sparse_none.normative_probability
        assert cs["retry_control"].buffer_affects_success_probability is True
        assert cs["negative_buffer"].normative_probability == sparse_none.normative_probability


def test_buffer_text_never_mentions_probability_of_success_change():
    banned = ["more likely to succeed", "increases the chance", "improves the odds", "less likely to fail"]
    for spec in DOMAINS.values():
        for b in spec["buffers"].values():
            for lvl in BUFFER_LEVELS:
                assert not any(x in b[lvl].lower() for x in banned)


def test_condition_ids_unique():
    ids = [c.condition_id for inst in INSTANCES for c in build_conditions(inst)]
    assert len(ids) == len(set(ids))


def test_parser_handles_fences_and_percentages():
    assert parse_belief('```json\n{"success_probability": 0.6, "lower_bound": 0.4, "upper_bound": 0.8}\n```')["success_probability"] == 0.6
    r = parse_belief('Sure: {"success_probability": 60, "lower_bound": 45, "upper_bound": 70}')
    assert abs(r["success_probability"] - 0.6) < 1e-9 and abs(r["lower_bound"] - 0.45) < 1e-9
    with pytest.raises(Exception):
        parse_belief("no json here")


@pytest.mark.parametrize("inst", INSTANCES, ids=lambda x: x.instance_id)
def test_exog_conditions_share_everything_but_buffer(inst):
    conds = [c for c in build_conditions(inst) if c.condition_type == "exog"]
    assert len(conds) == 6
    for amb in ("sparse", "qualitative"):
        g = [c for c in conds if c.ambiguity == amb]
        assert len({c.slots["independence"] for c in g}) == 1 and g[0].slots["independence"]
        assert len({c.slots["buffer"] for c in g}) == 3
        for slot in ("context", "evidence", "question", "format"):
            assert len({c.slots[slot] for c in g}) == 1
        assert len({c.normative_probability for c in g}) == 1
