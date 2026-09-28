"""Re-score the saved LoRA adapter with and without the deterministic ICD-10 post-process.

python -m src.eval_adapter    (run after src.train_eval has written results/adapter)
"""
from __future__ import annotations

import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import parse, score
from .postprocess import fix_codes
from .train_eval import ROOT, generate, load


def main():
    report = json.loads((ROOT / "results" / "report.json").read_text())
    tok = AutoTokenizer.from_pretrained(report["model"])
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(report["model"], torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, ROOT / "results" / "adapter")
    test = load("test")[: report["baseline"]["n"]]
    notes, targets = [r["note"] for r in test], [r["target"] for r in test]
    outs, _ = generate(model, tok, notes)
    fixed = [json.dumps(fix_codes(parse(o), n)) if parse(o) else o for o, n in zip(outs, notes)]
    report["lora_ft_rescored"] = score(outs, targets)
    report["lora_ft_plus_icd_lookup"] = score(fixed, targets)
    report["icd_errors"] = [{"note": n, "pred": (parse(o) or {}).get("icd10"), "gold": t["icd10"]}
                            for o, n, t in zip(outs, notes, targets) if (parse(o) or {}).get("icd10") != t["icd10"]][:8]
    (ROOT / "results" / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("lora", report["lora_ft_rescored"])
    print("lora+lookup", report["lora_ft_plus_icd_lookup"])


if __name__ == "__main__":
    main()
