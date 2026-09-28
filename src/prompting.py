INSTRUCTION = ("Extract JSON with keys chief_complaint (string), icd10 (sorted list of ICD-10 codes) and "
               "medications (sorted list of lowercase drug names). ICD-10 map: hypertension=I10, hyperlipidemia=E78.5, "
               "type 2 diabetes=E11.9, CKD stage 4=N18.4, atrial fibrillation=I48.91, COPD=J44.9, heart failure=I50.9. "
               "Output JSON only.")


def build_prompt(tokenizer, note: str) -> str:
    msgs = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": note}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
