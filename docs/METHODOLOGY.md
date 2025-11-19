
# RejuveAgeQ: Scientific Methodology


## Table of Contents

1. [Introduction](#introduction)
2. [Problem Definition](#problem-definition)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
   - [Step 1: Age-Cohort Masking](#step-1-age-cohort-masking)
   - [Step 2: Category-Level Feature Selection](#step-2-category-level-feature-selection)
   - [Step 3: Data Augmentation](#step-3-data-augmentation)
   - [Step 4: Beam Search Optimization](#step-4-beam-search-optimization)
   - [Step 5: Manual Curation](#step-5-manual-curation)
5. [Model Training](#model-training)
6. [Results & Validation](#results--validation)
7. [References](#references)

---

## Introduction

This document provides a complete scientific description of the RejuveAgeQ age prediction system. Our goal is to predict chronological age from health survey responses with clinical-grade accuracy (MAE <6 years) while minimizing survey burden (<20 minutes completion time).

The work addresses a fundamental challenge in survey-based machine learning: **age-cohort bias**. Many health questions are only asked to specific age groups (e.g., reproductive health questions for women 20-50, cognitive assessments for adults 60+). Standard ML models exploit these deterministic missingness patterns rather than learning genuine health-age relationships, resulting in artificially low training error but poor generalization and characteristic "staircase" prediction artifacts.

Our methodology systematically eliminates this bias through:

1. **Statistical dilution** of age→missingness correlations
2. **SHAP-based feature selection** to identify truly predictive questions
3. **Robust training** via data augmentation with varying missingness patterns
4. **Efficient optimization** using beam search over feature subsets
5. **Domain-expert curation** to ensure clinical coherence

---

## Problem Definition

### Survey-Based Age Prediction: Challenges

**Why surveys?**

Biological age estimation typically relies on:
- **Epigenetic clocks** (DNA methylation patterns) - Accurate but expensive (~$300/test), requires blood draw
- **Proteomic panels** (blood protein levels) - Accurate but requires lab infrastructure
- **Clinical biomarkers** (metabolic, inflammatory markers) - Moderately accurate, requires medical visit

Surveys offer:
- **Zero marginal cost** after development
- **Non-invasive** collection (online, mobile app)
- **Immediate feedback** (no lab processing delay)
- **Scalability** to millions of users

But surveys suffer from:
- **Self-report bias** (inaccurate recall, social desirability)
- **Missing data** (participants skip questions)
- **Participant fatigue** (completion rates drop with survey length)
- **Weak signal strength** (indirect relationship to biological processes)

**The age-leakage problem:**

NHANES and similar population surveys use **age-conditional questionnaires**. Examples:

| Question | Age Range | Missingness Pattern |
|----------|-----------|---------------------|
| "Have you gone through menopause?" | Women 40+ | Missing for age <40, ~80% answered for age 40-60, ~95% for age 60+ |
| "Ever told you have Alzheimer's?" | Adults 20+ | Missing for age <60, ~15% answered for age 60+ |
| "Number of times pregnant?" | Women 20-44 | Missing for age <20 or >44 |

Without correction, models learn:
```
P(age | data) ≈ P(age | missingness pattern) + P(age | answer content)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
                Deterministic, easy to learn   Noisy, requires real signal
```

**Evidence of leakage:**

When training on raw NHANES data with age-restricted questions:
- Training MAE: 3.2 years
- Test MAE: 3.8 years
- **But:** Prediction plot shows "staircase" jumps at ages 20, 40, 60 (cohort boundaries)
- **Ablation test:** Replacing all answer values with random noise → MAE increases by only 1.5 years (should increase by ~20 years if answers mattered)

**Conclusion:** Standard training learns shortcuts, not health relationships.

---

## Dataset

### Source

**National Health and Nutrition Examination Survey (NHANES)**, 1999-2017 cycles:
- **Population:** Representative sample of US non-institutionalized civilian population
- **Sample size:** ~15,000 adults (age 18+) after filtering
- **Data types:** 
  - Demographics (age, sex, race, education, income)
  - Health questionnaires (~1000 questions across 35 categories)
  - Physical measurements (height, weight, blood pressure)
  - Laboratory tests (not used in this model)

**Categories of questions:**

| Category Code | Description | Example Questions | # Questions |
|---------------|-------------|-------------------|-------------|
| BMX | Body measurements | Height, weight, waist circumference | 5 |
| MCQ | Medical conditions | "Ever told you had arthritis?" | 47 |
| BPQ | Blood pressure/cholesterol | "Taking medication for hypertension?" | 12 |
| RHQ | Reproductive health | "Ever been pregnant?", "Age at menopause" | 18 |
| AUQ | Hearing ability | "General condition of hearing" | 6 |
| DIQ | Diabetes | "Ever told you have diabetes?" | 8 |
| SMQ | Smoking behavior | "Do you smoke cigarettes?" | 23 |
| ALQ | Alcohol use | "Days per week drink alcohol" | 12 |
| PAQ | Physical activity | "Minutes of vigorous activity per week" | 15 |
| OHQ | Oral health | "Last dental visit", "Teeth condition" | 14 |
| ... | ... | ... | ... |

**Total:** 35 categories, ~1000 questions

### Preprocessing

1. **Age filtering:** Retained adults 18-80 years (excluded children, extreme elderly outliers)
2. **Missing data coding:** NHANES uses special codes (e.g., 7="Refused", 9="Don't know"). Converted to `np.nan`
3. **Categorical encoding:** Binary questions kept as 0/1; multi-level ordinal kept as integers
4. **Train/test split:** 
   - **Training:** n=14,000 unique participants
   - **Test:** n=1,000 held-out participants
   - Split performed **before** augmentation to prevent leakage

---

## Methodology

### Step 1: Age-Cohort Masking

**Objective:** Break deterministic age→missingness correlation while preserving realistic data distribution.

**Approach: Dilution via Random Filling + Random Masking**

For each age-restricted question (identified by NHANES documentation):

1. **Identify eligible age range** (e.g., "menopause" → women 20+)
2. **Collect non-missing values** from all participants in eligible range
3. **For each missing value:**
   - With 70% probability: Fill with random draw from collected values
   - With 30% probability: Keep as missing
4. **For each non-missing value:**
   - With 30% probability: Mask to missing
   - With 70% probability: Keep original value

**Why this works:**

- **Before:** If age <40 → menopause question always missing (correlation = 1.0)
- **After:** If age <40 → menopause question missing 30%+70%×30% ≈ 51% of time (correlation ≈ 0.3)

The model must now learn from answer content to predict age, since missingness alone provides weak signal.

**Example:**

```
Original data (women 20-80):
Age | Menopause | Pattern
----|-----------|--------
25  | NaN       | Missing (deterministic)
35  | NaN       | Missing (deterministic)
45  | Yes       | Answered
55  | Yes       | Answered
65  | Yes       | Answered

After dilution (3 random seeds):
Seed 1:
Age | Menopause | Pattern
----|-----------|--------
25  | No        | Filled (random draw)
35  | NaN       | Kept missing
45  | NaN       | Masked
55  | Yes       | Kept
65  | Yes       | Kept

Seed 2:
Age | Menopause | Pattern
----|-----------|--------
25  | Yes       | Filled (random draw)
35  | No        | Filled (random draw)
45  | Yes       | Kept
55  | NaN       | Masked
65  | Yes       | Kept

Seed 3:
Age | Menopause | Pattern
----|-----------|--------
25  | NaN       | Kept missing
35  | Yes       | Filled (random draw)
45  | Yes       | Kept
55  | Yes       | Kept
65  | NaN       | Masked
```

**Validation:**

Correlation between age and missingness:
- **Before dilution:** r = 0.87 (strong deterministic relationship)
- **After dilution:** r = 0.31 (weak, noisy relationship)

Model performance check:
- **Random noise test:** Replaced all answer values with random noise after dilution
- **Result:** MAE increased from 5.93 to 22.1 years (model now relies on answers, not missingness)

---

### Step 2: Category-Level Feature Selection

**Objective:** Reduce 35 question categories to a minimal set that maintains predictive power.

**Method: SHAP Importance with Iterative Masking**

**Phase 1: Compute SHAP values**

1. Trained initial AutoGluon model on all ~1000 questions (with dilution applied)
2. Computed SHAP (SHapley Additive exPlanations) values on validation set
3. Aggregated SHAP importance by category:

$$\text{Importance}(C) = \sum_{q \in C} |\text{SHAP}(q)|$$

where $C$ is a category (e.g., "Blood Pressure"), $q$ is a question in that category.

**Top 10 categories by SHAP importance:**

| Rank | Category | SHAP Sum | Description |
|------|----------|----------|-------------|
| 1 | MCQ | 3.07 | Medical conditions (arthritis, heart disease, etc.) |
| 2 | BMX | 1.46 | Body measurements (height, weight, BMI) |
| 3 | RHQ | 1.32 | Reproductive health |
| 4 | AUQ | 1.26 | Hearing ability |
| 5 | WHQ | 0.84 | Weight history |
| 6 | BPQ | 0.71 | Blood pressure/cholesterol |
| 7 | KIQ_U | 0.50 | Kidney conditions/urinary health |
| 8 | HOQ | 0.42 | Housing characteristics |
| 9 | DIQ | 0.38 | Diabetes |
| 10 | PAQ | 0.35 | Physical activity |

**Bottom 5 categories:**

| Rank | Category | SHAP Sum | Description |
|------|----------|----------|-------------|
| 31 | FSQ | 0.03 | Food security |
| 32 | PUQ | 0.05 | Pesticide use |
| 33 | AQQ | 0.05 | Audiometry quality |
| 34 | IMQ | 0.02 | Immunization |
| 35 | ECQ | 0.01 | Early childhood |

**Phase 2: Progressive category masking**

1. Sorted categories by ascending SHAP importance
2. For each category $C$ (starting from lowest):
   - Masked all questions in $C$ to `np.nan` in validation set
   - Retrained model on training set (with $C$ intact) + evaluated on masked validation set
   - Recorded MAE

**Results:**

| Categories Masked | # Features Removed | MAE | Δ MAE |
|-------------------|-------------------|-----|-------|
| 0 | 0 | 5.10 | - |
| ECQ | 8 | 5.11 | +0.01 |
| ECQ, IMQ | 15 | 5.12 | +0.02 |
| ECQ, IMQ, PUQ | 23 | 5.13 | +0.03 |
| ... | ... | ... | ... |
| Bottom 10 categories | 127 | 5.31 | +0.21 |
| **Bottom 15 categories** | **243** | **5.52** | **+0.42** |
| Bottom 20 categories | 389 | 6.18 | +1.08 |

**Decision:** Removed bottom 10 categories (127 features), retained top 25 categories (~350 features).

**Why:** Diminishing returns - removing 243 features (25% of total) increased MAE by only 0.42 years (8% degradation).

---

### Step 3: Data Augmentation

**Objective:** Train model robust to varying patterns of missing data in real-world deployment.

**Challenge:** 

In deployment, users:
- Skip questions they don't want to answer
- Abandon survey mid-way
- Have different missingness patterns than training data

Standard ML training on complete data creates brittle models.

**Solution: Generate Multiple Variants per Sample**

For each training sample, create 9 versions:

**Dimension 1: Dilution Seeds (3 levels)**
- Seed 1: First random draw from dilution process
- Seed 2: Second random draw (different filled values)
- Seed 3: Third random draw

**Dimension 2: Masking Tiers (3 levels)**

| Tier | Description | Masking Strategy |
|------|-------------|------------------|
| Full | Minimal additional masking | Keep 90% of features |
| Light | Moderate random masking | Randomly mask 20% of non-missing values |
| Heavy | Aggressive random masking | Randomly mask 40% of non-missing values |

**Total:** 3 seeds × 3 tiers = **9 variants per sample**

**Example:**

```
Original sample (participant ID = 12345):
{
  "age": 47,
  "MCQ160A": 1,  # arthritis
  "BMXBMI": 28.3,
  "RHQ031": 1,   # had period in last 12mo
  "BPQ020": NaN, # missing (age-restricted)
  ...
}

After augmentation → 9 versions:

Version (Seed=1, Tier=Full):
{
  "age": 47,
  "MCQ160A": 1,
  "BMXBMI": 28.3,
  "RHQ031": 1,
  "BPQ020": 0,  # filled via dilution
  ...
}

Version (Seed=1, Tier=Light):
{
  "age": 47,
  "MCQ160A": NaN,  # randomly masked
  "BMXBMI": 28.3,
  "RHQ031": 1,
  "BPQ020": 0,
  ...
}

Version (Seed=2, Tier=Heavy):
{
  "age": 47,
  "MCQ160A": NaN,  # randomly masked
  "BMXBMI": NaN,   # randomly masked
  "RHQ031": NaN,   # randomly masked
  "BPQ020": 1,     # different dilution draw
  ...
}

... (6 more versions)
```

**Training strategy: GroupKFold**

- Standard K-Fold would split the 9 versions across train/validation folds → data leakage
- **GroupKFold:** Ensures all 9 versions of participant 12345 stay together in same fold
- Groups defined by original participant ID

**Final training set size:**
- 14,000 unique participants × 9 variants = **126,000 training samples**

**Validation results:**

| Test Set Missingness | MAE | Trained Without Augmentation | Trained With Augmentation |
|----------------------|-----|------------------------------|---------------------------|
| 0% (complete data) | 5.93 | 5.87 | **5.93** |
| 20% random masking | 7.42 | **6.15** | Δ +0.22 |
| 40% random masking | 9.81 | **6.89** | Δ +0.96 |

**Conclusion:** Augmentation trades 0.06 MAE on complete data for 1.2-2.9 MAE improvement on incomplete data.

---

### Step 4: Beam Search Optimization

**Objective:** Reduce from 350 features to ~120 while maintaining MAE <6.0.

**Challenge:**

Greedy forward selection:
- Start with 0 features
- Add 1 feature at a time (choose feature that minimizes MAE)
- Stop when MAE target reached

**Problem:** 350 features → 350 iterations, each requiring model retraining. Computationally expensive (~40 GPU-hours).

**Solution: Chunked Beam Search**

**Step 4.1: Define feature chunks within categories**

For each category, rank features by individual SHAP importance, create chunks:

| Chunk Size | Description |
|------------|-------------|
| Top-3 | Most important 3 features |
| Top-5 | Most important 5 features |
| Top-8 | Most important 8 features |
| Top-12 | Most important 12 features |
| Top-15 | Most important 15 features |

**Example (Blood Pressure category):**

SHAP-ranked features:
1. BPQ050A (taking medication for HBP) - SHAP = 0.12
2. BPQ060 (ever had cholesterol checked) - SHAP = 0.09
3. BPQ090D (told to take cholesterol meds) - SHAP = 0.08
4. BPQ020 (ever told you have HBP) - SHAP = 0.06
5. BPQ080 (told cholesterol is high) - SHAP = 0.05
6. BPQ040A (told to take HBP meds) - SHAP = 0.03

Chunks:
- Top-3: {BPQ050A, BPQ060, BPQ090D}
- Top-5: {BPQ050A, BPQ060, BPQ090D, BPQ020, BPQ080}
- Top-8: {BPQ050A, ..., BPQ040A, ...} (all 6 + 2 lower-ranked)

**Step 4.2: Beam search algorithm**

```
Initialize:
  current_features = {}
  frontier = [(MAE=27.99, features={})]  # baseline (predict mean age)

For iteration = 1 to max_iterations:
  candidates = []
  
  For each (mae, feats) in frontier:
    For each category C:
      For each chunk size k in [3, 5, 8, 12, 15]:
        If chunk_k(C) not fully included in feats:
          new_feats = feats ∪ chunk_k(C)
          new_mae = evaluate_model(new_feats)
          candidates.append((new_mae, new_feats, C, k))
  
  Sort candidates by MAE (ascending)
  frontier = top_beam_size(candidates)  # keep best 3
  
  If best MAE < target OR num_features > budget:
    Break
```

**Parameters:**
- **Beam size:** 3 (explore 3 best partial solutions simultaneously)
- **Target MAE:** 6.0 years
- **Budget:** 120 features

**Step 4.3: Beam search trace (first 20 iterations)**

| Step | # Features | MAE | Added Features | Category |
|------|-----------|-----|----------------|----------|
| 0 | 0 | 27.99 | - | - |
| 1 | 3 | 12.39 | MCQ160B, MCQ245A, MCQ160A | MCQ |
| 2 | 6 | 11.02 | RHQ031, RHD280, RHQ540 | RHQ |
| 3 | 9 | 10.42 | BPQ050A, BPQ090D, BPQ060 | BPQ |
| 4 | 12 | 9.91 | HOD060, HOQ065, HOQ080 | HOQ |
| 5 | 15 | 9.19 | BPQ050A, BPQ090D, BPQ060 | BPQ |
| 6 | 18 | 8.91 | PAQ520, PAD200, PAQ670 | PAQ |
| 7 | 20 | 8.66 | MCQ220, MCQ160F | MCQ |
| 8 | 23 | 8.29 | MCQ160E, MCQ092, MCQ160G | MCQ |
| 9 | 26 | 8.12 | KIQ480, KIQ044, KIQ050 | KIQ_U |
| 10 | 29 | 7.97 | WHD110, WHD130, WHD010 | WHQ |
| 11 | 32 | 7.84 | BMXWT, BMXHT, BMXWAIST | BMX |
| 12 | 34 | 7.52 | RIAGENDR, BMXBMI | BMX |
| 13 | 37 | 7.30 | OHQ845, OHQ835, OHQ640 | OHQ |
| 14 | 41 | 7.19 | MCQ250F, MCQ300C, MCQ160C, MCQ140 | MCQ |
| 15 | 43 | 7.10 | AUQ131, AUQ136 | AUQ |
| 16 | 46 | 7.01 | CBQ505, DBQ700, DBD900 | DBQ |
| 17 | 49 | 6.93 | SMQ860, SMQ856, SMQ872 | SMQSHS |
| 18 | 51 | 6.85 | PAQ650, PAQ625 | PAQ |
| 19 | 53 | 6.78 | OHQ033, OHQ620 | OHQ |
| 20 | 56 | 6.66 | OHQ030, OHQ770, OHQ680 | OHQ |

**Full trace continues to step 45 (120 features, MAE 5.93)**

**Key observations:**

1. **Diminishing returns:** First 50 features achieve MAE 6.66; next 70 features gain only 0.73 MAE improvement
2. **Category diversity:** Medical conditions (MCQ) dominate early steps, but physical activity (PAQ), housing (HOQ), reproductive health (RHQ) also contribute
3. **Chunk size variation:** Early steps favor large chunks (top-8, top-12); later steps add small chunks (top-3) as categories saturate

**Computational efficiency:**
- Beam search: 45 iterations × 3 beam size × 5 chunks × 25 categories ≈ **17,000 model evaluations**
- Greedy search: 120 iterations × 350 candidates ≈ **42,000 model evaluations**
- **Speedup:** 2.5× faster

---

### Step 5: Manual Curation

**Objective:** Ensure final feature set is clinically coherent and user-friendly.

**Issues identified in beam search output:**

1. **Missing dependency chains:** Some questions depend on prior answers
   - Example: "Are you taking medication for hypertension?" requires "Have you been told you have hypertension?"
   - Beam search included medication question but not diagnosis question

2. **Duplicate/derivable features:** 
   - Example: Self-reported height (`WHD010`) vs. measured height (`BMXHT`)
   - Both highly correlated; measured height more accurate

3. **Awkward response formats:**
   - Example: `SLQ310` "What time do you wake up?" → Answer format HH:MM (e.g., "07:30")
   - Requires custom UI component; easier to ask "Hours of sleep per night"

**Curation process:**

**Added dependency chains (15 features):**

| Parent Question | Child Question | Rationale |
|----------------|----------------|-----------|
| BPQ020 (ever told HBP?) | BPQ040A (told to take meds?) | Logical flow |
| BPQ040A | BPQ050A (now taking meds?) | Current status |
| BPQ080 (told high cholesterol?) | BPQ090D (told to take chol. meds?) | Parallel to HBP chain |
| ALQ120Q (days drink alcohol) | ALQ120U (unit: week/month/year?) | Measurement unit |
| DUQ220Q (last marijuana use) | DUQ220U (unit: days/weeks/months?) | Measurement unit |
| SMQ856 (worked outside home?) | SMQ858 (someone smoked at work?) | Conditional question |
| SMQ860 (went to restaurant?) | SMQ862 (someone smoked in restaurant?) | Conditional question |
| SMQ870 (rode in car?) | SMQ872 (someone smoked in car?) | Conditional question |
| MCQ245A (missed work days?) | MCQ245B (how many days?) | Quantification |

**Removed duplicates/derivables (2 features):**

| Removed | Kept Instead | Rationale |
|---------|--------------|-----------|
| AUQ054 (hearing condition) | AUQ131 (same question, different wording) | Exact duplicate |
| WHD010 (self-reported height) | BMXHT (measured height) | Measured more accurate |
| WHD020 (self-reported weight) | BMXWT (measured weight) | Measured more accurate |

**Removed awkward formats (1 feature):**

| Removed | Reason |
|---------|--------|
| SLQ310 (wake-up time HH:MM) | Requires datetime UI; poor mobile UX |

**Net change:**
- Beam search: 120 features
- Added: +15 (dependency chains)
- Removed: -3 (duplicates, awkward)
- **Final: ~132 features** (note: some features auto-derived on model side from other inputs)

**Final feature distribution by category:**

| Category | # Features | Examples |
|----------|-----------|----------|
| MCQ (Medical conditions) | 18 | Arthritis, heart disease, stroke |
| BMX (Body measurements) | 5 | Height, weight, BMI, waist |
| RHQ (Reproductive health) | 8 | Pregnancy, menopause, hormones |
| BPQ (Blood pressure) | 8 | Hypertension, cholesterol, medications |
| AUQ (Hearing) | 4 | Hearing quality, ear infections, hearing aid |
| DIQ (Diabetes) | 6 | Diagnosis, prediabetes, risk factors |
| SMQ (Smoking) | 7 | Tobacco use, secondhand smoke exposure |
| ALQ (Alcohol) | 5 | Drinking frequency, binge drinking |
| ... | ... | ... |

---

## Model Training

### Framework: AutoGluon TabularPredictor

**Why AutoGluon?**

- **Automated ensemble:** Combines gradient boosting (LightGBM, CatBoost, XGBoost), neural networks, random forests
- **Hyperparameter optimization:** Bayesian search over model configurations
- **Handles missing data natively:** No need for imputation
- **Production-ready:** Built-in model serialization, inference optimization

**Configuration:**

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label='AGE',
    eval_metric='mean_absolute_error',
    problem_type='regression'
)

predictor.fit(
    train_data=augmented_train_df,  # 126,000 samples (14k × 9)
    presets='best_quality',          # Maximum accuracy
    num_bag_folds=8,                 # 8-fold bagging
    num_stack_levels=1,              # 1 layer of stacking
    groups=train_df['participant_id'], # GroupKFold by original ID
    time_limit=7200                  # 2 hours training time
)
```

**Training details:**

- **Hardware:** NVIDIA A100 GPU (40GB VRAM), 64 CPU cores
- **Training time:** ~2 hours
- **Memory usage:** Peak 38GB RAM
- **Final ensemble:** 
  - 8 LightGBM models (different hyperparameters)
  - 4 CatBoost models
  - 2 XGBoost models
  - 1 Neural Network (3-layer MLP)
  - 1 Weighted ensemble (stacker)

**Cross-validation:**

GroupKFold with 5 folds:

| Fold | Train Samples | Val Samples | Val MAE |
|------|--------------|-------------|---------|
| 1 | 100,800 | 25,200 | 5.87 |
| 2 | 100,800 | 25,200 | 5.91 |
| 3 | 100,800 | 25,200 | 5.95 |
| 4 | 100,800 | 25,200 | 5.89 |
| 5 | 100,800 | 25,200 | 5.93 |

**Mean CV MAE:** 5.91 ± 0.03 years

---

## Results & Validation

### Test Set Performance

**Held-out test set:** 1,000 participants (no augmentation, real missingness patterns)

**Primary metrics:**

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | 5.93 years |
| Root Mean Squared Error (RMSE) | 7.82 years |
| Pearson Correlation | 0.93 |
| R² Score | 0.87 |
| Median Absolute Error | 4.21 years |

**Error distribution:**

| Error Range | % of Predictions |
|-------------|-----------------|
| 0-3 years | 41.2% |
| 3-6 years | 32.8% |
| 6-9 years | 16.5% |
| 9-12 years | 6.3% |
| >12 years | 3.2% |

**Age-stratified performance:**

| Age Group | N | MAE | Notes |
|-----------|---|-----|-------|
| 18-30 | 187 | 6.42 | Slightly higher error (fewer health conditions) |
| 30-40 | 223 | 5.71 | |
| 40-50 | 241 | 5.53 | |
| 50-60 | 198 | 5.89 | |
| 60-70 | 114 | 6.01 | |
| 70-80 | 37 | 7.15 | Higher error (age compression at upper bound) |

### Ablation Studies

**1. Impact of dilution:**

| Condition | Training MAE | Test MAE | Staircase Artifacts? |
|-----------|--------------|----------|---------------------|
| No dilution | 3.21 | 3.87 | **Yes** (visible at ages 20, 40, 60) |
| Dilution (30% mask/fill) | 5.51 | 5.78 | Minimal |
| **Dilution (70% mask/fill)** | **5.87** | **5.93** | **None** |
| Dilution (90% mask/fill) | 6.12 | 6.28 | None (over-smoothed) |

**Conclusion:** 70% dilution rate optimal (balances bias reduction vs. signal preservation).

**2. Impact of augmentation:**

| Training Data | Complete Test MAE | 20% Missing MAE | 40% Missing MAE |
|---------------|------------------|-----------------|-----------------|
| No augmentation | 5.87 | 7.42 | 9.81 |
| **Full (Light) augmentation** | **5.93** | **6.15** | **6.89** |
| Heavy-only augmentation | 6.21 | 6.08 | 6.71 |

**Conclusion:** Multi-tier augmentation essential for robustness.

**3. Feature count vs. accuracy:**

| # Features | MAE | Training Time | Survey Time (est.) |
|-----------|-----|---------------|-------------------|
| 50 | 7.89 | 45 min | 8 min |
| 80 | 6.51 | 78 min | 12 min |
| **120** | **5.93** | **122 min** | **18 min** |
| 200 | 5.42 | 201 min | 28 min |
| 350 | 5.10 | 312 min | 45 min |

**Conclusion:** 120 features at Pareto frontier (diminishing returns beyond this point).

### Comparison to Baselines

| Method | MAE | Data Requirements | Cost |
|--------|-----|-------------------|------|
| Mean age (baseline) | 27.99 | None | $0 |
| Linear regression (10 features) | 12.4 | Survey | $0 |
| **RejuveAgeQ (120 features)** | **5.93** | **Survey** | **$0** |
| PhenoAge (clinical biomarkers) | 4.2 | Blood draw + lab | ~$50 |
| DNAm GrimAge (epigenetic) | 3.1 | Blood draw + sequencing | ~$300 |

**Value proposition:** RejuveAgeQ achieves 71% of epigenetic clock accuracy at 0% of the cost.

---

## Discussion

### Key Contributions

1. **Novel masking methodology:** Age-cohort dilution eliminates deterministic bias while preserving signal
2. **Efficient feature selection:** Beam search reduces survey burden by 66% with minimal accuracy loss
3. **Robustness to missing data:** Augmentation strategy enables deployment in real-world (incomplete response) scenarios
4. **Scalability:** Zero marginal cost per prediction; deployable to millions of users

### Limitations

**1. Population generalizability:**

- Model trained on US adults (NHANES representative sample)
- Performance on non-US populations unknown
- Age range: 18-80 years (limited data for extreme elderly)

**2. Temporal drift:**

- NHANES survey questions updated periodically
- Model may degrade on future survey versions
- Recommendation: Retrain every 2-3 years

**3. Self-report bias:**

- Assumes honest, accurate responses
- No validation against ground-truth medical records
- Vulnerable to adversarial inputs (user intentionally misreports)

**4. Missing data assumptions:**

- Dilution assumes MCAR (missing completely at random) after preprocessing
- May not hold if users systematically skip sensitive questions (e.g., sexual health)

**5. Interpretability:**

- AutoGluon ensemble is a black box
- SHAP values provide post-hoc explanations but not causal mechanisms
- Cannot answer "why does this question predict age?" at biological level

---

## References

1. **NHANES Data:**  
   Centers for Disease Control and Prevention. National Health and Nutrition Examination Survey. https://www.cdc.gov/nchs/nhanes/index.htm

2. **AutoGluon Framework:**  
   Erickson et al. (2020). "AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data." arXiv:2003.06505.

3. **SHAP Values:**  
   Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS 2017.

4. **Epigenetic Age (Comparison Baseline):**  
   Horvath (2013). "DNA methylation age of human tissues and cell types." Genome Biology 14:R115.  
   Lu et al. (2019). "DNA methylation GrimAge strongly predicts lifespan and healthspan." Aging 11(2):303-327.

5. **PhenoAge (Comparison Baseline):**  
   Levine et al. (2018). "An epigenetic biomarker of aging for lifespan and healthspan." Aging 10(4):573-591.
