from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
from mlx_lm import load, generate


MODEL_NAME = "mlx-community/Qwen2.5-7B-Instruct-4bit"

ALLOWED_CATEGORIES = {
    "water", "food", "health", "shelter", "wash",
    "protection", "education", "livelihoods", "cash", "energy"
}

SYSTEM_INSTRUCTION = """You are analyzing humanitarian assessment text.

Extract structured information from the three fields:
- needs
- urgent needs
- notes

Use only these need categories:
["water", "food", "health", "shelter", "wash", "protection", "education", "livelihoods", "cash", "energy"]

Return valid JSON only, with this exact schema:
{
  "need_categories": [],
  "urgent_need_categories": [],
  "displacement": false,
  "children_present": false,
  "elderly_present": false,
  "disability_present": false,
  "health_issue": false,
  "access_constraint": false
}

Rules:
- Only use the allowed categories.
- All flags must be booleans.
- Do not invent information not supported by the text.
- If unsure, prefer false for flags.
- Return JSON only. No explanation.
- "urgent_need_categories" must include only needs explicitly mentioned in the urgent needs field, unless the notes explicitly state that a need is urgent or immediate.
- Do not infer urgency from general hardship alone.
- Be conservative: do not add a category unless it is directly supported by the text.
"""


def build_prompt(needs: str, urgent_needs: str, notes: str) -> str:
    user_content = f"""Needs: {needs}
                    Urgent needs: {urgent_needs}
                    Notes: {notes}"""

    # Qwen instruct models use chat templates; mlx-lm exposes the tokenizer,
    # and we can apply the model's native chat template.
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]
    return messages


def safe_parse_json(text: str) -> dict[str, Any]:
    default = {
        "need_categories": [],
        "urgent_need_categories": [],
        "displacement": False,
        "children_present": False,
        "elderly_present": False,
        "disability_present": False,
        "health_issue": False,
        "access_constraint": False,
        "_parse_error": False,
    }

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        default["_parse_error"] = True
        return default

    out = default.copy()

    need_categories = data.get("need_categories", [])
    urgent_need_categories = data.get("urgent_need_categories", [])

    if isinstance(need_categories, list):
        out["need_categories"] = [x for x in need_categories if x in ALLOWED_CATEGORIES]

    if isinstance(urgent_need_categories, list):
        out["urgent_need_categories"] = [x for x in urgent_need_categories if x in ALLOWED_CATEGORIES]

    for key in [
        "displacement",
        "children_present",
        "elderly_present",
        "disability_present",
        "health_issue",
        "access_constraint",
    ]:
        out[key] = bool(data.get(key, False))

    return out


def model_response_to_json_text(raw_text: str) -> str:
    """
    Be defensive: sometimes models wrap JSON in markdown fences or add text.
    This tries to isolate the first JSON object.
    """
    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").strip()
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end + 1]

    return raw_text


def run_structured_coding(
    input_path: Path,
    output_path: Path,
    limit: int | None = 50,
    max_tokens: int = 220,
) -> None:
    df = pd.read_parquet(input_path).copy()

    if limit is not None:
        df = df.head(limit).copy()

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load(MODEL_NAME)

    results: list[dict[str, Any]] = []

    checkpoint_path = output_path.with_name("llm_structured_coding_checkpoint.parquet")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM structured coding"):
        needs = row.get("needs", "") or ""
        urgent_needs = row.get("urgent_needs", "") or ""
        notes = row.get("notes", "") or ""

        messages = build_prompt(needs, urgent_needs, notes)

        # Use the tokenizer's native chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        raw_output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        json_text = model_response_to_json_text(raw_output)
        parsed = safe_parse_json(json_text)

        results.append({
            "assessment_id": row["assessment_id"],
            "llm_need_categories": "|".join(parsed["need_categories"]),
            "llm_urgent_need_categories": "|".join(parsed["urgent_need_categories"]),
            "llm_displacement": parsed["displacement"],
            "llm_children_present": parsed["children_present"],
            "llm_elderly_present": parsed["elderly_present"],
            "llm_disability_present": parsed["disability_present"],
            "llm_health_issue": parsed["health_issue"],
            "llm_access_constraint": parsed["access_constraint"],
            "llm_parse_error": parsed["_parse_error"],
            "llm_raw_response": raw_output,
        })
        
        if len(results) % 25 == 0:
            results_df = pd.DataFrame(results)
            partial = df.iloc[:len(results)].merge(results_df, on="assessment_id", how="left")
            partial.to_parquet(checkpoint_path, index=False)
            print(f"Checkpoint saved at {len(results)} rows -> {checkpoint_path}")

    results_df = pd.DataFrame(results)
    merged = df.merge(results_df, on="assessment_id", how="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    print(f"Saved to: {output_path}")
    print(merged[[
        "assessment_id",
        "llm_need_categories",
        "llm_urgent_need_categories",
        "llm_parse_error",
    ]].head(10).to_string(index=False))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    input_path = root / "data" / "interim" / "assessments_with_gt.parquet"
    output_path = root / "data" / "processed" / "llm_structured_coding.parquet"

    run_structured_coding(
        input_path=input_path,
        output_path=output_path,
        limit=50,      # parti con 50 righe
        max_tokens=220,
    )


import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of rows to process. Use -1 for all rows.")
    parser.add_argument("--max-tokens", type=int, default=220)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    input_path = root / "data" / "interim" / "assessments_with_gt.parquet"
    output_path = root / "data" / "processed" / "llm_structured_coding.parquet"

    limit = None if args.limit == -1 else args.limit

    run_structured_coding(
        input_path=input_path,
        output_path=output_path,
        limit=limit,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()