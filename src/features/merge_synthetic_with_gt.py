from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]

    observed_path = root / "data" / "synthetic" / "synthetic_assessments.csv"
    gt_path = root / "data" / "synthetic" / "synthetic_assessments_ground_truth.csv"
    out_dir = root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "assessments_with_gt.parquet"

    df = pd.read_csv(observed_path)
    df_gt = pd.read_csv(gt_path)

    # Normalize empty textual fields
    df["urgent_needs"] = df["urgent_needs"].fillna("")
    df_gt["gt_urgent_categories"] = df_gt["gt_urgent_categories"].fillna("")

    # Merge
    df_full = df.merge(df_gt, on="assessment_id", how="left", validate="one_to_one")

    # Basic checks
    missing_after_merge = df_full.isna().sum()
    print("\nMissing values after merge:")
    print(missing_after_merge[missing_after_merge > 0])

    if df_full["assessment_id"].duplicated().any():
        raise ValueError("Duplicate assessment_id found after merge.")

    if len(df_full) != len(df):
        raise ValueError("Row count changed after merge.")

    # Save
    df_full.to_parquet(out_path, index=False)

    print(f"\nSaved merged file to: {out_path}")
    print(f"Shape: {df_full.shape}")
    print("\nSample:")
    print(df_full.head(5).to_string(index=False))


if __name__ == "__main__":
    main()