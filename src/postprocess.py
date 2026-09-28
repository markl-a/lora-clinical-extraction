"""Deterministic post-processing for lookup-type fields.

The LoRA run shows the model learns the output *format* well but maps diagnoses to ICD-10 codes poorly.
Code lookup is a table, not a reasoning task, so it should not live in the weights: take the diagnosis terms
that appear in the note and map them with the same table the prompt uses. The model still owns the fields
that need reading (chief complaint, medications).
"""
from __future__ import annotations

import re

ICD10 = {"hypertension": "I10", "hyperlipidemia": "E78.5", "type 2 diabetes": "E11.9", "CKD stage 4": "N18.4",
         "atrial fibrillation": "I48.91", "COPD": "J44.9", "heart failure": "I50.9"}


def icd10_from_note(note: str) -> list[str]:
    found = {code for term, code in ICD10.items() if re.search(rf"\b{re.escape(term)}\b", note, re.I)}
    return sorted(found)


def fix_codes(pred: dict | None, note: str) -> dict | None:
    """Replace the model's icd10 list with the table lookup; leave other fields as the model wrote them."""
    if pred is None:
        return None
    return {**pred, "icd10": icd10_from_note(note)}
