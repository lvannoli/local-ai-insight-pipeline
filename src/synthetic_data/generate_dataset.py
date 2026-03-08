import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Config
# =========================
N_RECORDS = 800
N_WEEKS = 8
START_DATE = datetime(2026, 1, 5)
SEED = 42
OUTDIR = Path("data/synthetic")
OUTDIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# =========================
# Lebanon-like geography
# =========================
# Synthetic admin structure inspired by Lebanon regions, but still simplified
ADMIN_STRUCTURE = {
    "North": {
        "Akkar": [
            ("Halba", 34.542, 36.079),
            ("Bebnine", 34.515, 36.029),
            ("Qoubaiyat", 34.571, 36.270),
        ],
        "Tripoli": [
            ("Bab al-Tabbaneh", 34.438, 35.834),
            ("Mina", 34.451, 35.813),
            ("Qalamoun", 34.386, 35.736),
        ],
        "Miniyeh-Danniyeh": [
            ("Miniyeh", 34.449, 35.839),
            ("Bakhoun", 34.443, 36.006),
            ("Deir Ammar", 34.444, 35.980),
        ],
    },
    "Bekaa": {
        "Baalbek": [
            ("Baalbek", 34.005, 36.218),
            ("Brital", 33.958, 36.320),
            ("Arsal", 34.250, 36.420),
        ],
        "Zahle": [
            ("Zahle", 33.846, 35.902),
            ("Bar Elias", 33.851, 35.987),
            ("Saadnayel", 33.820, 35.900),
        ],
        "West Bekaa": [
            ("Joub Jannine", 33.621, 35.784),
            ("Qab Elias", 33.824, 35.824),
            ("Saghbine", 33.604, 35.726),
        ],
    },
    "South": {
        "Saida": [
            ("Saida", 33.563, 35.368),
            ("Ain al-Hilweh", 33.546, 35.394),
            ("Ghaziyeh", 33.517, 35.368),
        ],
        "Tyre": [
            ("Tyre", 33.270, 35.203),
            ("Burj El Shimali", 33.275, 35.239),
            ("Qana", 33.208, 35.299),
        ],
        "Nabatieh": [
            ("Nabatieh", 33.378, 35.483),
            ("Bint Jbeil", 33.119, 35.433),
            ("Kfarkela", 33.284, 35.591),
        ],
    },
}

POP_GROUPS = ["IDPs", "returnees", "host community", "refugees", "mixed population"]
POP_WEIGHTS = [0.30, 0.12, 0.22, 0.26, 0.10]

PARTNERS = ["Partner A", "Partner B", "Partner C"]
ENUMERATORS = [f"ENUM_{i:02d}" for i in range(1, 13)]

NEED_CATEGORIES = [
    "water",
    "food",
    "health",
    "shelter",
    "wash",
    "protection",
    "education",
    "livelihoods",
    "cash",
    "energy",
]

NEED_SYNONYMS = {
    "water": ["water", "clean water", "drinking water", "safe water"],
    "food": ["food", "food assistance", "food support", "basic food items"],
    "health": ["health", "medicine", "medical care", "medical support", "clinic access", "treatment"],
    "shelter": ["shelter", "shelter repair", "housing", "shelter materials", "tent repair"],
    "wash": ["wash", "hygiene items", "wash items", "sanitation support", "soap and hygiene kits"],
    "protection": ["protection", "protection support", "safety support", "case support"],
    "education": ["education", "school access", "learning materials", "school supplies"],
    "livelihoods": ["livelihoods", "income support", "job access", "work opportunities"],
    "cash": ["cash", "cash support", "multipurpose cash", "cash assistance"],
    "energy": ["energy", "heating fuel", "fuel", "electricity support"],
}

# =========================
# Region patterns
# =========================
BASE_NEED_WEIGHTS = {
    "water": 1.0,
    "food": 1.0,
    "health": 1.0,
    "shelter": 1.0,
    "wash": 1.0,
    "protection": 0.8,
    "education": 0.6,
    "livelihoods": 0.8,
    "cash": 0.9,
    "energy": 0.7,
}

REGION_ADJUSTMENTS = {
    "Akkar": {"water": 1.5, "wash": 1.4, "food": 1.2},
    "Tripoli": {"protection": 1.4, "health": 1.3, "cash": 1.2},
    "Miniyeh-Danniyeh": {"water": 1.2, "food": 1.2, "livelihoods": 1.2},
    "Baalbek": {"shelter": 1.4, "energy": 1.3, "health": 1.2},
    "Zahle": {"cash": 1.3, "food": 1.1, "health": 1.1},
    "West Bekaa": {"water": 1.3, "health": 1.2, "protection": 1.1},
    "Saida": {"protection": 1.4, "health": 1.2, "wash": 1.1},
    "Tyre": {"shelter": 1.3, "food": 1.2, "health": 1.2},
    "Nabatieh": {"energy": 1.4, "shelter": 1.3, "water": 1.1},
}

# Time patterns across 8 weeks
WEEK_ADJUSTMENTS = {
    1: {"food": 1.2, "cash": 1.2},
    2: {"food": 1.2, "cash": 1.1},
    3: {"water": 1.2, "wash": 1.2},
    4: {"water": 1.3, "wash": 1.2},
    5: {"health": 1.2, "water": 1.2},
    6: {"shelter": 1.2, "energy": 1.2},
    7: {"shelter": 1.3, "health": 1.2},
    8: {"health": 1.3, "energy": 1.3},
}

# =========================
# Helper functions
# =========================
def weighted_sample(items, weights, k):
    chosen = []
    pool_items = list(items)
    pool_weights = list(weights)
    for _ in range(min(k, len(pool_items))):
        idx = random.choices(range(len(pool_items)), weights=pool_weights, k=1)[0]
        chosen.append(pool_items.pop(idx))
        pool_weights.pop(idx)
    return chosen

def jitter_coord(value, scale=0.01):
    return value + np.random.normal(0, scale)

def compute_need_weights(admin2, week, pop_group):
    weights = BASE_NEED_WEIGHTS.copy()

    for k, v in REGION_ADJUSTMENTS.get(admin2, {}).items():
        weights[k] *= v

    for k, v in WEEK_ADJUSTMENTS.get(week, {}).items():
        weights[k] *= v

    # Population group tendencies
    if pop_group == "IDPs":
        for k in ["shelter", "food", "health", "water"]:
            weights[k] *= 1.25
    elif pop_group == "refugees":
        for k in ["protection", "health", "cash"]:
            weights[k] *= 1.25
    elif pop_group == "host community":
        for k in ["livelihoods", "cash", "food"]:
            weights[k] *= 1.20
    elif pop_group == "returnees":
        for k in ["shelter", "energy", "livelihoods"]:
            weights[k] *= 1.20
    elif pop_group == "mixed population":
        for k in ["water", "wash", "health"]:
            weights[k] *= 1.10

    return weights

def choose_needs(admin2, week, pop_group):
    weights_dict = compute_need_weights(admin2, week, pop_group)
    cats = list(weights_dict.keys())
    weights = list(weights_dict.values())

    n_needs = random.choices([1, 2, 3], weights=[0.20, 0.55, 0.25], k=1)[0]
    selected_needs = weighted_sample(cats, weights, n_needs)

    n_urgent = random.choices([0, 1, 2], weights=[0.15, 0.60, 0.25], k=1)[0]
    urgent_pool = selected_needs.copy()
    if len(urgent_pool) == 0:
        urgent_pool = cats
    selected_urgent = random.sample(urgent_pool, k=min(n_urgent, len(urgent_pool)))

    return selected_needs, selected_urgent

def generate_flags(pop_group, selected_needs):
    displacement = pop_group in ["IDPs", "returnees"] and random.random() < 0.75
    children_present = random.random() < 0.55
    elderly_present = random.random() < 0.20
    disability_present = random.random() < 0.12
    health_issue = ("health" in selected_needs and random.random() < 0.70) or random.random() < 0.10
    access_constraint = ("water" in selected_needs or "health" in selected_needs) and random.random() < 0.45
    return {
        "displacement": displacement,
        "children_present": children_present,
        "elderly_present": elderly_present,
        "disability_present": disability_present,
        "health_issue": health_issue,
        "access_constraint": access_constraint,
    }

def score_dimensions(selected_needs, selected_urgent, flags, pop_group):
    human = 1
    living = 1
    services = 1
    vulnerability = 1

    # Human impact
    if "health" in selected_needs:
        human += 1
    if "food" in selected_needs:
        human += 1
    if flags["health_issue"]:
        human += 1
    if len(selected_urgent) >= 1:
        human += 1

    # Living conditions
    if "shelter" in selected_needs:
        living += 1
    if "water" in selected_needs:
        living += 1
    if "wash" in selected_needs:
        living += 1
    if "energy" in selected_needs:
        living += 1

    # Access to services
    if "health" in selected_needs:
        services += 1
    if "education" in selected_needs:
        services += 1
    if "water" in selected_needs:
        services += 1
    if flags["access_constraint"]:
        services += 1

    # Vulnerability
    if flags["displacement"]:
        vulnerability += 1
    if flags["children_present"]:
        vulnerability += 1
    if flags["elderly_present"] or flags["disability_present"]:
        vulnerability += 1
    if pop_group in ["IDPs", "refugees"]:
        vulnerability += 1

    # Small noise for realism
    human = min(5, max(1, human + random.choice([0, 0, 0, 1, -1])))
    living = min(5, max(1, living + random.choice([0, 0, 0, 1, -1])))
    services = min(5, max(1, services + random.choice([0, 0, 0, 1, -1])))
    vulnerability = min(5, max(1, vulnerability + random.choice([0, 0, 0, 1, -1])))

    severity_score = round(
        0.30 * human + 0.30 * living + 0.25 * services + 0.15 * vulnerability, 2
    )

    if severity_score < 1.5:
        severity_class = "Minimal"
    elif severity_score < 2.5:
        severity_class = "Stress"
    elif severity_score < 3.5:
        severity_class = "Severe"
    elif severity_score < 4.5:
        severity_class = "Extreme"
    else:
        severity_class = "Catastrophic"

    return {
        "human_impact": human,
        "living_conditions": living,
        "services_access": services,
        "vulnerability": vulnerability,
        "severity_score": severity_score,
        "severity_class": severity_class,
    }

def noisy_phrase(category):
    return random.choice(NEED_SYNONYMS[category])

def maybe_abbreviate(text):
    replacements = {
        "household": "HH",
        "children": "kids",
        "medical support": "med support",
        "displaced": "disp.",
        "hygiene items": "hygiene kits",
    }
    if random.random() < 0.15:
        for k, v in replacements.items():
            text = text.replace(k, v)
    return text

def generate_needs_text(selected_needs):
    phrases = [noisy_phrase(c) for c in selected_needs]
    return ", ".join(phrases)

def generate_urgent_text(selected_urgent):
    if not selected_urgent:
        return ""
    phrases = [noisy_phrase(c) for c in selected_urgent]
    return ", ".join(phrases)

def generate_notes(selected_needs, selected_urgent, flags, pop_group, admin1, admin2):
    templates = []

    if flags["displacement"]:
        templates.append("Recently displaced households were reported.")
    if pop_group == "refugees":
        templates.append("Refugee households were interviewed in the area.")
    if pop_group == "host community":
        templates.append("Host community families reported increased pressure on resources.")
    if pop_group == "returnees":
        templates.append("Returnee families are still struggling to re-establish living conditions.")

    if "water" in selected_needs:
        templates.append(random.choice([
            "Access to safe drinking water is insufficient.",
            "Families report shortages of clean water.",
            "Water access is irregular and quality is poor.",
        ]))

    if "food" in selected_needs:
        templates.append(random.choice([
            "Food stocks are running low.",
            "Households report difficulty accessing enough food.",
            "Basic food items are not sufficient for all family members.",
        ]))

    if "health" in selected_needs:
        templates.append(random.choice([
            "The nearest clinic is difficult to access.",
            "Medical care is needed but not easily available.",
            "Respondents reported lack of medicine and treatment.",
        ]))

    if "shelter" in selected_needs:
        templates.append(random.choice([
            "Shelter conditions are poor and some structures are damaged.",
            "Families need repair materials for their shelter.",
            "Overcrowding and inadequate cover were reported.",
        ]))

    if "wash" in selected_needs:
        templates.append(random.choice([
            "Hygiene items are not sufficient.",
            "Sanitation conditions are below acceptable standards.",
            "Families reported lack of soap and basic hygiene kits.",
        ]))

    if "protection" in selected_needs:
        templates.append(random.choice([
            "Some respondents expressed concerns about safety and privacy.",
            "Women and girls reported feeling unsafe in some areas.",
            "Protection concerns were raised during the interview.",
        ]))

    if "education" in selected_needs:
        templates.append(random.choice([
            "Children face barriers to school access.",
            "School materials are missing and attendance is affected.",
            "Education access remains limited for part of the population.",
        ]))

    if "livelihoods" in selected_needs:
        templates.append(random.choice([
            "Income opportunities are very limited.",
            "Households report reduced access to work.",
            "Livelihood options remain insufficient.",
        ]))

    if "cash" in selected_needs:
        templates.append(random.choice([
            "Households would prefer cash support to cover basic needs.",
            "Families reported inability to purchase essentials.",
            "Cash constraints are affecting access to food and services.",
        ]))

    if "energy" in selected_needs:
        templates.append(random.choice([
            "Fuel and electricity shortages were mentioned.",
            "Heating fuel is not sufficient for current needs.",
            "Energy access remains unstable.",
        ]))

    if flags["children_present"] and random.random() < 0.5:
        templates.append("Children are present in the household.")
    if flags["elderly_present"]:
        templates.append("At least one older adult is present.")
    if flags["disability_present"]:
        templates.append("A household member has a disability.")
    if flags["health_issue"]:
        templates.append(random.choice([
            "At least one family member was reported sick.",
            "Children with fever were mentioned.",
            "One household member requires medical follow-up.",
        ]))
    if flags["access_constraint"]:
        templates.append(random.choice([
            "Road conditions limit access to services.",
            "Transport to services is difficult.",
            "Movement constraints affect access to assistance.",
        ]))

    if selected_urgent:
        templates.append(f"Most urgent needs mentioned: {', '.join(noisy_phrase(c) for c in selected_urgent)}.")

    random.shuffle(templates)
    note = " ".join(templates[: random.randint(3, 6)])
    return maybe_abbreviate(note)

# =========================
# Generate records
# =========================
records = []
gt_records = []

all_admin1 = list(ADMIN_STRUCTURE.keys())

for i in range(N_RECORDS):
    week = random.randint(1, N_WEEKS)
    date = START_DATE + timedelta(days=random.randint((week - 1) * 7, week * 7 - 1))

    admin1 = random.choice(all_admin1)
    admin2 = random.choice(list(ADMIN_STRUCTURE[admin1].keys()))
    location_name, base_lat, base_lon = random.choice(ADMIN_STRUCTURE[admin1][admin2])

    lat = round(jitter_coord(base_lat, 0.015), 6)
    lon = round(jitter_coord(base_lon, 0.015), 6)

    pop_group = random.choices(POP_GROUPS, weights=POP_WEIGHTS, k=1)[0]
    partner = random.choice(PARTNERS)
    enumerator = random.choice(ENUMERATORS)

    selected_needs, selected_urgent = choose_needs(admin2, week, pop_group)
    flags = generate_flags(pop_group, selected_needs)
    sev = score_dimensions(selected_needs, selected_urgent, flags, pop_group)

    needs_text = generate_needs_text(selected_needs)
    urgent_text = generate_urgent_text(selected_urgent)
    notes = generate_notes(selected_needs, selected_urgent, flags, pop_group, admin1, admin2)

    assessment_id = f"A{i+1:04d}"

    records.append({
        "assessment_id": assessment_id,
        "date": date.strftime("%Y-%m-%d"),
        "week": week,
        "admin1": admin1,
        "admin2": admin2,
        "location_name": location_name,
        "lat": lat,
        "lon": lon,
        "population_group": pop_group,
        "partner": partner,
        "enumerator_id": enumerator,
        "needs": needs_text,
        "urgent_needs": urgent_text,
        "notes": notes,
    })

    gt_records.append({
        "assessment_id": assessment_id,
        "gt_need_categories": "|".join(selected_needs),
        "gt_urgent_categories": "|".join(selected_urgent),
        "gt_displacement": flags["displacement"],
        "gt_children_present": flags["children_present"],
        "gt_elderly_present": flags["elderly_present"],
        "gt_disability_present": flags["disability_present"],
        "gt_health_issue": flags["health_issue"],
        "gt_access_constraint": flags["access_constraint"],
        "gt_human_impact": sev["human_impact"],
        "gt_living_conditions": sev["living_conditions"],
        "gt_services_access": sev["services_access"],
        "gt_vulnerability": sev["vulnerability"],
        "gt_severity_score": sev["severity_score"],
        "gt_severity_class": sev["severity_class"],
    })

# =========================
# Save
# =========================
df = pd.DataFrame(records)
df_gt = pd.DataFrame(gt_records)

df.to_csv(OUTDIR / "synthetic_assessments.csv", index=False)
df_gt.to_csv(OUTDIR / "synthetic_assessments_ground_truth.csv", index=False)

print("Saved:")
print(OUTDIR / "synthetic_assessments.csv")
print(OUTDIR / "synthetic_assessments_ground_truth.csv")

print("\nSample:")
print(df.head(10).to_string(index=False))