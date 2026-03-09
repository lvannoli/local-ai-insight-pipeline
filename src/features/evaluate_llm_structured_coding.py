from pathlib import Path
import pandas as pd


FLAG_COLS = [
    ("gt_displacement", "pred_displacement", "llm_displacement"),
    ("gt_children_present", "pred_children_present", "llm_children_present"),
    ("gt_elderly_present", "pred_elderly_present", "llm_elderly_present"),
    ("gt_disability_present", "pred_disability_present", "llm_disability_present"),
    ("gt_health_issue", "pred_health_issue", "llm_health_issue"),
    ("gt_access_constraint", "pred_access_constraint", "llm_access_constraint"),
]


def normalize_pipe_set(value) -> set[str]:
    if pd.isna(value) or value == "":
        return set()
    return set(str(value).split("|"))


def exact_match(series_pred: pd.Series, series_gt: pd.Series) -> float:
    matches = [
        normalize_pipe_set(p) == normalize_pipe_set(g)
        for p, g in zip(series_pred, series_gt)
    ]
    return sum(matches) / len(matches) if matches else 0.0


def set_precision_recall_f1(series_pred: pd.Series, series_gt: pd.Series) -> dict:
    tp = fp = fn = 0

    for pred, gt in zip(series_pred, series_gt):
        pred_set = normalize_pipe_set(pred)
        gt_set = normalize_pipe_set(gt)

        tp += len(pred_set & gt_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(((y_true == True) & (y_pred == True)).sum())
    tn = int(((y_true == False) & (y_pred == False)).sum())
    fp = int(((y_true == False) & (y_pred == True)).sum())
    fn = int(((y_true == True) & (y_pred == False)).sum())

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    root = Path(__file__).resolve().parents[2]

    llm_path = root / "data" / "processed" / "llm_structured_coding.parquet"
    baseline_path = root / "data" / "processed" / "baseline_features.parquet"
    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = pd.read_parquet(llm_path)
    baseline = pd.read_parquet(baseline_path)

    # Keep only rows actually scored by LLM
    eval_ids = llm["assessment_id"].unique()
    baseline = baseline[baseline["assessment_id"].isin(eval_ids)].copy()

    df = llm.merge(
        baseline[
            [
                "assessment_id",
                "pred_need_categories",
                "pred_urgent_need_categories",
                "pred_displacement",
                "pred_children_present",
                "pred_elderly_present",
                "pred_disability_present",
                "pred_health_issue",
                "pred_access_constraint",
            ]
        ],
        on="assessment_id",
        how="left",
        validate="one_to_one",
    )

    print(f"Evaluation rows: {len(df)}")

    # -------------------------
    # Step 1: categories
    # -------------------------
    results = []

    for label, pred_col in [
        ("baseline_need_categories", "pred_need_categories"),
        ("llm_need_categories", "llm_need_categories"),
        ("baseline_urgent_need_categories", "pred_urgent_need_categories"),
        ("llm_urgent_need_categories", "llm_urgent_need_categories"),
    ]:
        gt_col = "gt_need_categories" if "urgent" not in pred_col else "gt_urgent_categories"

        exact = exact_match(df[pred_col], df[gt_col])
        prf = set_precision_recall_f1(df[pred_col], df[gt_col])

        results.append({
            "task": label,
            "exact_match": round(exact, 4),
            **prf,
        })

    categories_df = pd.DataFrame(results)
    print("\n=== CATEGORY METRICS ===")
    print(categories_df.to_string(index=False))

    categories_df.to_csv(out_dir / "structured_coding_category_metrics.csv", index=False)

    # -------------------------
    # Step 2: flags
    # -------------------------
    flag_rows = []

    for gt_col, base_col, llm_col in FLAG_COLS:
        base_metrics = binary_metrics(df[gt_col], df[base_col])
        llm_metrics = binary_metrics(df[gt_col], df[llm_col])

        flag_rows.append({
            "flag": gt_col.replace("gt_", ""),
            "model": "baseline",
            **base_metrics,
        })
        flag_rows.append({
            "flag": gt_col.replace("gt_", ""),
            "model": "llm",
            **llm_metrics,
        })

    flags_df = pd.DataFrame(flag_rows)
    print("\n=== FLAG METRICS ===")
    print(flags_df.to_string(index=False))

    flags_df.to_csv(out_dir / "structured_coding_flag_metrics.csv", index=False)

    # Save comparison table
    compare_out = out_dir / "llm_vs_baseline_structured_coding_sample.csv"
    df.to_csv(compare_out, index=False)
    print(f"\nSaved comparison sample to: {compare_out}")

    # Parse errors
    if "llm_parse_error" in df.columns:
        print("\n=== LLM PARSE ERRORS ===")
        print(df["llm_parse_error"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()