"""Scoring that matches how the output would be used downstream: must parse, then field-level correctness."""
from __future__ import annotations

import json
import re


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def set_f1(pred, gold) -> float:
    p, g = set(map(str, pred or [])), set(map(str, gold))
    if not p and not g:
        return 1.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def score(outputs: list[str], targets: list[dict]) -> dict:
    n = len(targets)
    parsed = [parse(o) for o in outputs]
    valid = [p for p in parsed if p is not None]
    cc = sum(1 for p, t in zip(parsed, targets) if p and str(p.get("chief_complaint", "")).strip().lower() == t["chief_complaint"])
    icd = sum(set_f1(p.get("icd10") if p else None, t["icd10"]) for p, t in zip(parsed, targets))
    med = sum(set_f1([str(x).lower() for x in (p.get("medications") or [])] if p else None, t["medications"]) for p, t in zip(parsed, targets))
    exact = sum(1 for p, t in zip(parsed, targets) if p and p.get("chief_complaint") == t["chief_complaint"]
                and sorted(map(str, p.get("icd10") or [])) == t["icd10"] and sorted(str(x).lower() for x in (p.get("medications") or [])) == t["medications"])
    return {"n": n, "json_valid": round(len(valid) / n, 3), "chief_complaint_acc": round(cc / n, 3),
            "icd10_f1": round(icd / n, 3), "medications_f1": round(med / n, 3), "exact_match": round(exact / n, 3)}
