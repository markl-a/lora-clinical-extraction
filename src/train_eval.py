"""Baseline (instruction-only) vs LoRA fine-tuned, same small model, same test set.

python -m src.train_eval --model HuggingFaceTB/SmolLM2-135M-Instruct --steps 150
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import score
from .prompting import build_prompt

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    return [json.loads(l) for l in (ROOT / "data" / f"{name}.jsonl").read_text().splitlines()]


@torch.no_grad()
def generate(model, tok, notes, max_new_tokens=80):
    model.eval()
    outs, lat = [], []
    for note in notes:
        ids = tok(build_prompt(tok, note), return_tensors="pt")
        t0 = time.perf_counter()
        gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        lat.append(time.perf_counter() - t0)
        outs.append(tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs, sum(lat) / len(lat)


def train(model, tok, rows, steps, lr, bs):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for step in range(steps):
        batch = [rows[(step * bs + i) % len(rows)] for i in range(bs)]
        texts, prompt_lens = [], []
        for r in batch:
            prompt = build_prompt(tok, r["note"])
            answer = json.dumps(r["target"]) + tok.eos_token
            texts.append(prompt + answer)
            prompt_lens.append(len(tok(prompt)["input_ids"]))
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=384)
        labels = enc["input_ids"].clone()
        for i, pl in enumerate(prompt_lens):
            labels[i, :pl] = -100          # loss only on the answer tokens
        labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels).loss
        loss.backward(); opt.step(); opt.zero_grad()
        losses.append(round(loss.item(), 4))
        if step % 25 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=60)
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    test = load("test")[: a.n_eval]
    notes, targets = [r["note"] for r in test], [r["target"] for r in test]

    base_out, base_lat = generate(model, tok, notes)
    base = score(base_out, targets)
    print("baseline", base, flush=True)

    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    t0 = time.perf_counter()
    losses = train(model, tok, load("train"), a.steps, a.lr, a.bs)
    train_s = time.perf_counter() - t0

    tok.padding_side = "left"
    ft_out, ft_lat = generate(model, tok, notes)
    ft = score(ft_out, targets)
    print("lora", ft, flush=True)

    report = {"model": a.model, "lora": {"rank": a.rank, "targets": sorted(cfg.target_modules), "trainable_params": trainable,
              "trainable_pct": round(100 * trainable / total, 3)}, "train": {"steps": a.steps, "batch": a.bs, "lr": a.lr,
              "seconds": round(train_s, 1), "loss_first": losses[0], "loss_last": losses[-1]},
              "baseline": {**base, "sec_per_sample": round(base_lat, 2)}, "lora_ft": {**ft, "sec_per_sample": round(ft_lat, 2)},
              "examples": [{"note": n, "baseline": b, "lora": f} for n, b, f in list(zip(notes, base_out, ft_out))[:3]]}
    (ROOT / "results" / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    model.save_pretrained(ROOT / "results" / "adapter")
    print(json.dumps({k: report[k] for k in ("lora", "train", "baseline", "lora_ft")}, indent=1))


if __name__ == "__main__":
    main()
