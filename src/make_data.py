"""Synthetic clinical notes -> structured JSON. Deterministic (seeded) so results are reproducible.

The task is deliberately format-fixed: extract chief complaint, ICD-10 codes and active meds.
That is the kind of task where fine-tuning beats prompting on a small model — the knowledge is in the note,
what the model must learn is the output contract.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

COMPLAINTS = ["chest tightness", "shortness of breath", "dizziness", "ankle swelling", "palpitations", "fatigue", "headache", "cough"]
DX = [("I10", "hypertension"), ("E78.5", "hyperlipidemia"), ("E11.9", "type 2 diabetes"), ("N18.4", "CKD stage 4"),
      ("I48.91", "atrial fibrillation"), ("J44.9", "COPD"), ("I50.9", "heart failure")]
MEDS = ["amlodipine", "lisinopril", "metformin", "atorvastatin", "apixaban", "furosemide", "spironolactone", "tiotropium"]
TEMPLATES = [
    "{age}yo {sex} presents with {cc}. PMH: {dx_text}. Currently taking {meds}. BP {bp}.",
    "CC: {cc} x{days} days. Hx of {dx_text}. Home meds: {meds}. Vitals BP {bp}, HR {hr}.",
    "Pt ({sex}, {age}) reports {cc}. Known {dx_text}; on {meds}. Plan: labs and follow-up.",
    "Follow-up visit. Main complaint {cc}. Problem list: {dx_text}. Medications: {meds}.",
]


def sample(rng: random.Random) -> dict:
    dx = rng.sample(DX, rng.randint(1, 3))
    meds = rng.sample(MEDS, rng.randint(1, 3))
    cc = rng.choice(COMPLAINTS)
    note = rng.choice(TEMPLATES).format(
        age=rng.randint(45, 88), sex=rng.choice(["M", "F"]), cc=cc, dx_text=", ".join(t for _, t in dx),
        meds=", ".join(meds), bp=f"{rng.randint(110, 175)}/{rng.randint(60, 100)}", days=rng.randint(1, 14), hr=rng.randint(55, 110))
    target = {"chief_complaint": cc, "icd10": sorted(c for c, _ in dx), "medications": sorted(meds)}
    return {"note": note, "target": target}


def main(n_train: int = 400, n_test: int = 60, seed: int = 7) -> None:
    rng = random.Random(seed)
    out = Path(__file__).resolve().parent.parent / "data"
    for name, n in (("train", n_train), ("test", n_test)):
        with (out / f"{name}.jsonl").open("w") as f:
            for _ in range(n):
                f.write(json.dumps(sample(rng)) + "\n")


if __name__ == "__main__":
    main()
