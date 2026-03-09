from pathlib import Path
import pandas as pd


def split_pipe(value) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    return str(value).split("|")


def compute_llm_severity_scores(
    llm_need_categories: list[str],
    llm_urgent_need_categories: list[str],
    llm_displacement: bool,
    llm_children_present: bool,
    llm_elderly_present: bool,
    llm_disability_present: bool,
    llm_health_issue: bool,
    llm_access_constraint: bool,
) -> dict:
    human = 1
    living = 1
    services = 1
    vulnerability = 1

    # Human impact
    if "health" in llm_need_categories:
        human += 2
    if "food" in llm_need_categories:
        human += 1
    if "water" in llm_need_categories:
        human += 1
    if llm_health_issue:
        human += 1
    if len(llm_urgent_need_categories) >= 1:
        human += 1

    # Living conditions
    if "shelter" in llm_need_categories:
        living += 2
    if "water" in llm_need_categories:
        living += 1
    if "wash" in llm_need_categories:
        living += 1
    if "energy" in llm_need_categories:
        living += 1
    if llm_displacement:
        living += 1

    # Access to services
    if "health" in llm_need_categories:
        services += 1
    if "education" in llm_need_categories:
        services += 1
    if "water" in llm_need_categories:
        services += 1
    if llm_access_constraint:
        services += 2
    if len(llm_urgent_need_categories) >= 1:
        services += 1

    # Vulnerability
    if llm_displacement:
        vulnerability += 2
    if llm_children_present:
        vulnerability += 1
    if llm_elderly_present:
        vulnerability += 1
    if llm_disability_present:
        vulnerability += 1

    # Softer escalation rules
    critical_factors = 0

    if "health" in llm_need_categories and "water" in llm_need_categories:
        critical_factors += 1
    if "health" in llm_need_categories and "food" in llm_need_categories:
        critical_factors += 1
    if "shelter" in llm_need_categories and llm_displacement:
        critical_factors += 1
    if llm_health_issue and llm_access_constraint:
        critical_factors += 1
    if len(llm_need_categories) >= 3:
        critical_factors += 1
    if len(llm_urgent_need_categories) >= 2:
        critical_factors += 1

    if critical_factors >= 3:
        human += 1
        services += 1

    if critical_factors >= 4:
        living += 1
        vulnerability += 1

    # Clip
    human = min(5, max(1, human))
    living = min(5, max(1, living))
    services = min(5, max(1, services))
    vulnerability = min(5, max(1, vulnerability))

    severity_score = round(
        0.30 * human + 0.30 * living + 0.25 * services + 0.15 * vulnerability,
        2,
    )

    if severity_score < 1.5:
        severity_class = "Minimal"
    elif severity_score < 2.5:
        severity_class = "Stress"
    elif severity_score < 3.5:
        severity_class = "Severe"
    elif severity_score < 4.6:
        severity_class = "Extreme"
    else:
        severity_class = "Catastrophic"

    critical_flag = severity_class in ["Extreme", "Catastrophic"]

    return {
        "llm_human_impact": human,
        "llm_living_conditions": living,
        "llm_services_access": services,
        "llm_vulnerability": vulnerability,
        "llm_severity_score": severity_score,
        "llm_severity_class": severity_class,
        "llm_critical_flag": critical_flag,
    }


def main():
    root = Path(__file__).resolve().parents[2]

    in_path = root / "data" / "processed" / "llm_structured_coding.parquet"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "llm_severity_features.parquet"

    df = pd.read_parquet(in_path).copy()

    score_rows = []

    for _, row in df.iterrows():
        llm_need_categories = split_pipe(row.get("llm_need_categories", ""))
        llm_urgent_need_categories = split_pipe(row.get("llm_urgent_need_categories", ""))

        scores = compute_llm_severity_scores(
            llm_need_categories=llm_need_categories,
            llm_urgent_need_categories=llm_urgent_need_categories,
            llm_displacement=bool(row.get("llm_displacement", False)),
            llm_children_present=bool(row.get("llm_children_present", False)),
            llm_elderly_present=bool(row.get("llm_elderly_present", False)),
            llm_disability_present=bool(row.get("llm_disability_present", False)),
            llm_health_issue=bool(row.get("llm_health_issue", False)),
            llm_access_constraint=bool(row.get("llm_access_constraint", False)),
        )
        score_rows.append(scores)

    scores_df = pd.DataFrame(score_rows)
    df = pd.concat([df, scores_df], axis=1)

    df.to_parquet(out_path, index=False)

    print(f"Saved LLM severity features to: {out_path}")
    print(f"Shape: {df.shape}")

    cols = [
        "assessment_id",
        "llm_need_categories",
        "llm_urgent_need_categories",
        "llm_displacement",
        "llm_health_issue",
        "llm_access_constraint",
        "llm_human_impact",
        "llm_living_conditions",
        "llm_services_access",
        "llm_vulnerability",
        "llm_severity_score",
        "llm_severity_class",
        "gt_severity_class",
    ]
    print("\nSample:")
    print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()