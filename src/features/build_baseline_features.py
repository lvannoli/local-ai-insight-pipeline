from pathlib import Path
import pandas as pd


NEED_KEYWORDS = {
    "water": ["water", "clean water", "drinking water", "safe water"],
    "food": ["food", "food assistance", "food support", "basic food items"],
    "health": ["health", "medicine", "medical", "medical care", "medical support", "clinic", "treatment", "fever"],
    "shelter": ["shelter", "shelter repair", "housing", "shelter materials", "tent repair", "overcrowding", "cover"],
    "wash": ["wash", "hygiene", "hygiene items", "wash items", "sanitation", "soap", "hygiene kits"],
    "protection": ["protection", "protection support", "safety support", "case support", "unsafe", "privacy", "safety"],
    "education": ["education", "school", "school access", "learning materials", "school supplies"],
    "livelihoods": ["livelihoods", "income support", "job access", "work opportunities", "income opportunities", "work"],
    "cash": ["cash", "cash support", "cash assistance", "multipurpose cash"],
    "energy": ["energy", "heating fuel", "fuel", "electricity", "electricity support"],
}

DISPLACEMENT_KEYWORDS = ["displaced", "recently displaced", "returnee", "returnees"]
CHILDREN_KEYWORDS = ["children", "child", "kids"]
ELDERLY_KEYWORDS = ["older adult", "elderly"]
DISABILITY_KEYWORDS = ["disability", "disabled"]
HEALTH_ISSUE_KEYWORDS = ["sick", "fever", "medical follow-up", "medicine", "treatment", "clinic"]
ACCESS_CONSTRAINT_KEYWORDS = [
    "difficult to access",
    "access is limited",
    "road conditions",
    "transport to services is difficult",
    "movement constraints",
    "inaccessible",
    "irregular",
]


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def extract_categories(text: str, keyword_map: dict[str, list[str]]) -> list[str]:
    found = []
    for category, keywords in keyword_map.items():
        if contains_any(text, keywords):
            found.append(category)
    return found


def compute_baseline_scores(
    pred_need_categories: list[str],
    pred_urgent_need_categories: list[str],
    pred_displacement: bool,
    pred_children_present: bool,
    pred_elderly_present: bool,
    pred_disability_present: bool,
    pred_health_issue: bool,
    pred_access_constraint: bool,
) -> dict:
    human = 1
    living = 1
    services = 1
    vulnerability = 1

    # Human impact
    if "health" in pred_need_categories:
        human += 1
    if "food" in pred_need_categories:
        human += 1
    if "water" in pred_need_categories:
        human += 1
    if pred_health_issue:
        human += 1
    if len(pred_urgent_need_categories) >= 1:
        human += 1

    # Living conditions
    if "shelter" in pred_need_categories:
        living += 1
    if "water" in pred_need_categories:
        living += 1
    if "wash" in pred_need_categories:
        living += 1
    if "energy" in pred_need_categories:
        living += 1
    if pred_displacement:
        living += 1

    # Access to services
    if "health" in pred_need_categories:
        services += 1
    if "education" in pred_need_categories:
        services += 1
    if "water" in pred_need_categories:
        services += 1
    if pred_access_constraint:
        services += 1
    if len(pred_urgent_need_categories) >= 1:
        services += 1

    # Vulnerability
    if pred_displacement:
        vulnerability += 1
    if pred_children_present:
        vulnerability += 1
    if pred_elderly_present:
        vulnerability += 1
    if pred_disability_present:
        vulnerability += 1

    human = min(5, max(1, human))
    living = min(5, max(1, living))
    services = min(5, max(1, services))
    vulnerability = min(5, max(1, vulnerability))

    severity_score = round(
        0.30 * human + 0.30 * living + 0.25 * services + 0.15 * vulnerability, 2
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
        "pred_human_impact": human,
        "pred_living_conditions": living,
        "pred_services_access": services,
        "pred_vulnerability": vulnerability,
        "pred_severity_score": severity_score,
        "pred_severity_class": severity_class,
        "pred_critical_flag": critical_flag,
    }


def main():
    root = Path(__file__).resolve().parents[2]

    in_path = root / "data" / "interim" / "assessments_with_gt.parquet"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "baseline_features.parquet"

    df = pd.read_parquet(in_path).copy()

    # Normalize text
    for col in ["needs", "urgent_needs", "notes"]:
        df[col] = df[col].fillna("").str.lower()

    # Combined text fields
    df["combined_text"] = (
        "needs: " + df["needs"] +
        " urgent needs: " + df["urgent_needs"] +
        " notes: " + df["notes"]
    )

    # Structured extraction
    df["pred_need_categories"] = df["combined_text"].apply(
        lambda x: "|".join(extract_categories(x, NEED_KEYWORDS))
    )
    df["pred_urgent_need_categories"] = df["urgent_needs"].apply(
        lambda x: "|".join(extract_categories(x, NEED_KEYWORDS))
    )

    df["pred_displacement"] = df["combined_text"].apply(lambda x: contains_any(x, DISPLACEMENT_KEYWORDS))
    df["pred_children_present"] = df["combined_text"].apply(lambda x: contains_any(x, CHILDREN_KEYWORDS))
    df["pred_elderly_present"] = df["combined_text"].apply(lambda x: contains_any(x, ELDERLY_KEYWORDS))
    df["pred_disability_present"] = df["combined_text"].apply(lambda x: contains_any(x, DISABILITY_KEYWORDS))
    df["pred_health_issue"] = df["combined_text"].apply(lambda x: contains_any(x, HEALTH_ISSUE_KEYWORDS))
    df["pred_access_constraint"] = df["combined_text"].apply(lambda x: contains_any(x, ACCESS_CONSTRAINT_KEYWORDS))

    # Severity scoring
    score_rows = []
    for _, row in df.iterrows():
        pred_need_categories = row["pred_need_categories"].split("|") if row["pred_need_categories"] else []
        pred_urgent_need_categories = (
            row["pred_urgent_need_categories"].split("|") if row["pred_urgent_need_categories"] else []
        )

        scores = compute_baseline_scores(
            pred_need_categories=pred_need_categories,
            pred_urgent_need_categories=pred_urgent_need_categories,
            pred_displacement=bool(row["pred_displacement"]),
            pred_children_present=bool(row["pred_children_present"]),
            pred_elderly_present=bool(row["pred_elderly_present"]),
            pred_disability_present=bool(row["pred_disability_present"]),
            pred_health_issue=bool(row["pred_health_issue"]),
            pred_access_constraint=bool(row["pred_access_constraint"]),
        )
        score_rows.append(scores)

    scores_df = pd.DataFrame(score_rows)
    df = pd.concat([df, scores_df], axis=1)

    # Save
    df.to_parquet(out_path, index=False)

    print(f"Saved baseline features to: {out_path}")
    print(f"Shape: {df.shape}")
    print("\nSample:")
    cols = [
        "assessment_id",
        "needs",
        "urgent_needs",
        "notes",
        "pred_need_categories",
        "pred_urgent_need_categories",
        "pred_severity_class",
        "gt_severity_class",
    ]
    print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()