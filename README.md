# Local AI Insight Pipeline

Prototype for local analysis of humanitarian free-text fields (`needs`, `urgent_needs`, `notes`) using a fully offline LLM pipeline.

The current goal is to produce:
- structured coding
- severity scoring
- intermediate analytical outputs for dashboards

The project is designed for privacy-preserving workflows where sensitive text data must remain on the local machine.

At this stage, the prototype is tested on synthetic data inspired by Lebanon-based humanitarian assessments.

## What The Project Does

The pipeline currently supports two main tasks.

### 1. Structured Coding

Input text fields:
- `needs`
- `urgent_needs`
- `notes`

Extracted outputs:
- need categories
- urgent need categories
- displacement
- children present
- elderly present
- disability present
- health issue
- access constraint

### 2. Severity Estimation

The pipeline uses extracted features to compute an INFORM-like severity class:
- `Minimal`
- `Stress`
- `Severe`
- `Extreme`
- `Catastrophic`

The project currently compares:
- a rule-based baseline
- a local LLM-based pipeline

## Current Architecture

```text
Synthetic assessment data
    ↓
Data audit / validation
    ↓
Rule-based baseline
    ↓
Local LLM structured coding
    ↓
Severity estimation
    ↓
Evaluation against ground truth
```

## Local Model Setup

The current prototype uses:
- `Qwen2.5 7B Instruct`
- quantized to 4-bit
- running locally via `MLX` on Apple Silicon

This allows local execution on a laptop without sending data to external APIs.

## Tech Stack

- Python
- Pandas
- PyArrow
- Jupyter
- Matplotlib
- DuckDB
- Streamlit
- MLX / MLX-LM
- Qwen2.5 7B Instruct (4-bit)

## Repository Structure

```text
data/
  synthetic/   # synthetic generated data
  interim/     # merged working tables
  processed/   # baseline, LLM outputs, severity outputs

notebooks/
  01_explore_synthetic_data.ipynb

src/
  synthetic_data/   # synthetic data generation
  features/         # baseline, merge, evaluation, severity
  llm/              # prompts and local LLM structured coding
  dashboard/        # future dashboard app
```

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Using `pyproject.toml`:

```bash
pip install -e .
```

## End-To-End Workflow

Below is the recommended order to run the project.

### Step 1. Generate synthetic data

This creates the synthetic dataset and the associated ground truth.

```bash
python -m src.synthetic_data.generate_dataset
```

Outputs:
- `data/synthetic/synthetic_assessments.csv`
- `data/synthetic/synthetic_assessments_ground_truth.csv`

### Step 2. Explore and validate synthetic data

Use the notebook:
- `notebooks/01_explore_synthetic_data.ipynb`

Recommended checks:
- missing values
- category distributions
- severity distribution
- text plausibility
- geographic and temporal patterns

### Step 3. Merge observed data and ground truth

Creates a single working table for development and evaluation.

```bash
python -m src.features.merge_synthetic_with_gt
```

Output:
- `data/interim/assessments_with_gt.parquet`

### Step 4. Build the rule-based baseline

Extracts categories and binary signals using keyword rules, then computes a baseline severity score.

```bash
python -m src.features.build_baseline_features
```

Output:
- `data/processed/baseline_features.parquet`

### Step 5. Run local LLM structured coding

Extracts structured information from:
- `needs`
- `urgent_needs`
- `notes`

Run on a subset first:

```bash
python -m src.llm.xml_structured_coding --limit 50
```

Run on full dataset:

```bash
python -m src.llm.xml_structured_coding --limit -1
```

Output:
- `data/processed/llm_structured_coding.parquet`

Note: the file is currently named `xml_structured_coding.py`, but it is using MLX (not XML).

### Step 6. Evaluate structured coding

Compares LLM outputs and baseline outputs against synthetic ground truth.

```bash
python -m src.features.evaluate_llm_structured_coding
```

Outputs include evaluation tables under:
- `outputs/tables/`

### Step 7. Build severity from LLM features

Computes severity dimensions and final severity class from LLM structured outputs.

```bash
python -m src.features.build_llm_severity_features
```

Output:
- `data/processed/llm_severity_features.parquet`

### Step 8. Evaluate severity

Severity can be evaluated in a notebook using:
- class distribution
- confusion matrix
- accuracy

Typical checks:

```python
df["llm_severity_class"].value_counts()
pd.crosstab(df["gt_severity_class"], df["llm_severity_class"], margins=True)
(df["gt_severity_class"] == df["llm_severity_class"]).mean()
```
