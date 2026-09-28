from src.metrics import parse, score, set_f1
from src.make_data import sample
import random


def test_parse_extracts_json_from_chatter():
    assert parse('Sure! {"chief_complaint": "cough", "icd10": [], "medications": []} done') == {"chief_complaint": "cough", "icd10": [], "medications": []}
    assert parse("no json here") is None
    assert parse("{broken") is None


def test_set_f1():
    assert set_f1(["I10"], ["I10"]) == 1.0
    assert set_f1(["I10", "E11.9"], ["I10"]) == 2 * (0.5 * 1) / 1.5
    assert set_f1([], ["I10"]) == 0.0


def test_score_perfect_and_invalid():
    t = [{"chief_complaint": "cough", "icd10": ["J44.9"], "medications": ["tiotropium"]}]
    import json
    assert score([json.dumps(t[0])], t)["exact_match"] == 1.0
    assert score(["garbage"], t)["json_valid"] == 0.0


def test_synthetic_sample_is_consistent():
    s = sample(random.Random(1))
    for m in s["target"]["medications"]:
        assert m in s["note"]
    assert s["target"]["icd10"] == sorted(s["target"]["icd10"])
