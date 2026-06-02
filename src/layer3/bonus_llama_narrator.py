#!/usr/bin/env python3
"""
Optional bonus: local HuggingFace Llama narrative generator for Layer 3 outputs.

This is NOT required for Layer 3 validation. It is only for turning the generated
CSV/JSON explanations into a natural-language explanation.

Install example:
  python3 -m pip install transformers accelerate torch sentencepiece

You may need a HuggingFace token and access approval for:
  meta-llama/Llama-3.1-8B-Instruct

Run:
  python3 src/layer3/bonus_llama_narrator.py \
    --layer3_run outputs/layer3/run_YYYYMMDD_HHMMSS \
    --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(layer3_run: Path, top_n: int = 10) -> str:
    explanations = read_json(layer3_run / "explanations.json")
    rec_path = layer3_run / "recommendations.csv"
    shap_path = layer3_run / "global_shap_summary.csv"
    goals_path = layer3_run / "goals_met_summary.csv"

    rec = pd.read_csv(rec_path).head(top_n) if rec_path.exists() else pd.DataFrame()
    shap = pd.read_csv(shap_path).head(20) if shap_path.exists() else pd.DataFrame()
    goals = pd.read_csv(goals_path).iloc[0].to_dict() if goals_path.exists() else {}

    return f"""
You are explaining an operating-room block allocation optimization result to a hospital operations user.
Use clear language, no equations unless needed.

Layer 3 coverage summary:
{json.dumps(explanations.get("coverage", {}), indent=2)}

Goals:
{json.dumps(goals, indent=2)}

Top global SHAP drivers:
{shap.to_string(index=False) if len(shap) else "No SHAP table available."}

Top recommendations:
{rec[["provider_id", "severity", "message"]].to_string(index=False) if len(rec) else "No recommendations table available."}

Write:
1. A short executive summary.
2. Why the recommendation is being made.
3. Which forecast drivers matter most.
4. What the user should check before accepting changes.
""".strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer3_run", required=True, type=Path)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=900)
    args = ap.parse_args()

    prompt = build_prompt(args.layer3_run)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception as e:
        raise SystemExit(
            "transformers/torch are not installed. Run:\n"
            "python3 -m pip install transformers accelerate torch sentencepiece\n"
            f"Original error: {e}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    messages = [
        {"role": "system", "content": "You are a careful hospital OR optimization explainability assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = pipe(text, max_new_tokens=args.max_new_tokens, do_sample=False)[0]["generated_text"]

    output_path = args.layer3_run / "llama_narrative.md"
    output_path.write_text(out, encoding="utf-8")
    print(f"Saved narrative to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
