import json

from src.postprocess import fix_codes, icd10_from_note

ROWS = [json.loads(l) for l in open("data/test.jsonl")]


def test_lookup_matches_gold_on_test_set():
    assert all(icd10_from_note(r["note"]) == r["target"]["icd10"] for r in ROWS)


def test_fix_codes_keeps_model_fields_and_handles_unparsed():
    pred = {"chief_complaint": "cough", "icd10": ["I50"], "medications": ["metformin"]}
    out = fix_codes(pred, "Known COPD, hypertension; on metformin.")
    assert out == {"chief_complaint": "cough", "icd10": ["I10", "J44.9"], "medications": ["metformin"]}
    assert fix_codes(None, "anything") is None
