# lora-clinical-extraction

**Instruction-only vs LoRA fine-tuning on the same small model, for structured extraction from clinical notes. It measures what fine-tuning actually buys, and where it should stop.**

This is a portfolio demo from September 2026. The notes are **synthetic** and template-generated, so no patient data is involved and the results can be reproduced. It trains on a laptop CPU in about 11 minutes.

Task: note → `{"chief_complaint": str, "icd10": [codes], "medications": [names]}`

## Result (60 held-out notes, `HuggingFaceTB/SmolLM2-135M-Instruct`)

| | JSON valid | Chief complaint | ICD-10 F1 | Medications F1 | Exact match | sec/sample (CPU) |
|---|---|---|---|---|---|---|
| Instruction only (ICD map in the prompt) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.91 |
| LoRA, 150 steps | **1.00** | **1.00** | 0.536 | **1.00** | 0.167 | 2.32 |
| LoRA + deterministic ICD-10 lookup | 1.00 | 1.00 | **1.00** | 1.00 | **1.00** | — |

- LoRA: rank 16 on `q/k/v/o_proj`, **1.84 M trainable params (1.35 % of the model)**, 150 steps × batch 8, lr 2e-4. The loss is taken only on answer tokens.
- Training time: 667 s on CPU. Loss fell from 0.864 to 0.080.
- All numbers come from `results/report.json`. `results/train_log.txt` is the committed run.
- A second run with the same seed reproduced these numbers exactly.

### What this shows

1. **Without fine-tuning, a 135M model cannot follow the output contract.** It echoes the note, wraps the answer in prose, or invents keys (`{"data": [{"key": "M", ...`). Scoring requires parseable JSON, so every field scores 0.
2. **LoRA teaches the format and the reading fields almost for free.** It uses 1.35 % of the parameters and 150 steps. Chief complaint and medications appear verbatim in the note, and the model learns to copy them into the right slot.
3. **LoRA did not teach the lookup, and it failed in the dangerous way.** The ICD codes are well-formed but wrong: `["I48.91","J44.9","N18.4"]` for a patient with hypertension, AF and heart failure. A schema validator would pass them. This happens even though the code table is in the prompt.
4. **So the lookup does not belong in the weights.** `src/postprocess.py` takes the diagnosis terms found in the note and maps them through the same table. The model keeps the fields that need reading, and code handles the table.

### Honest limits

- The data is **template-generated**. Real notes have abbreviations, negation ("no history of AF") and misspellings, so the regex lookup here would not survive them unchanged. On real data I would pair it with a terminology service and a negation check.
- With templated notes, rules alone could solve most of this task. The point here is the *division of labour* the experiment shows (fine-tune for the contract, deterministic code for lookups), not clinical accuracy.
- The base model is tiny on purpose so that it runs on CPU. A 7B instruct model would likely get the format right from the prompt alone. The comparison that matters there is cost and latency per call versus a fine-tuned small model.

## When I would (and would not) fine-tune

| Situation | Choice |
|---|---|
| Knowledge that changes (guidelines, drug lists, prices) | RAG or tools, not fine-tuning |
| Fixed output contract with high volume, where a small or local model is needed for cost or privacy | LoRA on a small model |
| Code tables, unit conversions, arithmetic | Deterministic code |
| The prompt and a strong model already hit the target | Don't fine-tune; keep the eval set and revisit when volume or cost changes |

## Run

```bash
pip install -r requirements.txt
python -m src.make_data          # 400 train / 60 test synthetic notes (seeded)
python -m src.train_eval         # baseline eval → LoRA train → eval → results/report.json + results/adapter/
python -m src.eval_adapter       # re-score saved adapter, with and without the ICD lookup
pytest -q                        # metrics, data consistency, lookup == gold on the test set
```

CI runs the tests only. Training runs locally, and its outputs are committed under `results/`.

## Layout

```
src/make_data.py     synthetic notes → gold JSON (seeded)
src/prompting.py     system instruction incl. the ICD-10 table; chat template
src/metrics.py       parse-then-score: JSON validity, field accuracy, set-F1, exact match
src/train_eval.py    baseline vs LoRA (PEFT), answer-only loss, latency
src/postprocess.py   deterministic ICD-10 lookup
src/eval_adapter.py  re-score saved adapter ± lookup, dump ICD error cases
```

MIT License.
