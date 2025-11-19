# RejuveAgeQ: Survey-Based Biological Age Estimation

## Overview

RejuveAgeQ is a machine learning system that predicts chronological age from health survey responses. Unlike blood-based biomarkers or clinical measurements, survey data is **inexpensive to collect, non-invasive, and scalable** — but traditionally considered a weak signal due to self-report bias and missingness patterns. This project demonstrates that carefully curated survey questions, when processed with appropriate ML techniques, can achieve **clinical-grade age prediction** (MAE <6 years).

The model is trained on the National Health and Nutrition Examination Survey (NHANES), a comprehensive US population health dataset spanning 1999-2017. It forms a component of Rejuve's broader initiative to build dynamic models of human biological functions using AI and heterogeneous health data.

**Model Performance:**
- **Mean Absolute Error:** 5.93 years
- **Pearson Correlation:** 0.93
- **Feature Count:** 120 questions (vs. 350+ candidates)
- **Survey Completion Time:** ~15-20 minutes (estimated)

---

## Why Survey-Based Age Prediction Matters

### The Challenge

- **Biomarker tests** (epigenetic clocks, proteomics) are accurate but expensive and require lab infrastructure
- **Comprehensive surveys** (NHANES has 1000+ questions) provide rich data but suffer from:
  - Participant fatigue → incomplete responses
  - Age-cohort biases (e.g., menopause questions only asked to older women)
  - Self-report noise
- **Short surveys** are practical but lack predictive power

### Our Contribution

We developed a methodology to:

1. **Identify the minimal set** of high-value questions (120 vs. 350+)
2. **Eliminate age-leakage artifacts** from cohort-specific questions
3. **Train models robust to missing data** through strategic augmentation
4. **Achieve clinical accuracy** (MAE 5.93 years) with a survey completable in 15-20 minutes

---

## Methodology Overview

Our approach addresses three core problems:

### 1. Age-Cohort Masking

**Problem:** Questions like "Have you gone through menopause?" are only asked to women >40. Models exploit this deterministic pattern rather than learning true health signals.

**Solution:** Randomly "dilute" the age→missingness correlation by:
- Filling 70% of age-restricted missing values with realistic draws
- Masking 30% of existing values back to missing

**Why:** Forces the model to learn from answer content, not just presence/absence of data.

### 2. SHAP-Based Feature Selection

**Problem:** Not all questions contribute equally. Some categories (e.g., kidney health) are highly predictive; others (e.g., food safety practices) add noise.

**Solution:**
- Computed SHAP importance for all 35 question categories
- Iteratively removed low-value categories until MAE degraded
- Applied beam search within high-value categories to select optimal question subsets

**Why:** Reduces survey burden without sacrificing accuracy.

### 3. Data Augmentation with Masking Tiers

**Problem:** Real-world users skip questions unpredictably. Training on complete data creates brittle models.

**Solution:**
- Created 9 variants per training sample:
  - 3 random dilution seeds
  - 3 masking levels (full survey, light masking, heavy masking)
- Used GroupKFold to prevent leakage across variants

**Why:** Model performs reliably even when 20-40% of questions are unanswered.

### 4. Beam Search Optimization

**Problem:** Greedy feature selection (add one question at a time) is computationally expensive for 350+ candidates.

**Solution:**
- Grouped questions into "chunks" within categories (top-3, top-5, top-8 by SHAP)
- Beam search to explore multiple candidate sets simultaneously
- Stopped at 120 features when MAE reached 5.93 (vs. 5.10 for full 350-feature set)

**Why:** Efficient exploration of the feature space; 66% fewer questions for 16% increase in error.

### 5. Manual Curation

**Problem:** Statistical importance doesn't guarantee logical coherence.

**Solution:**
- Added dependency chains (e.g., "Do you have hypertension?" → "Do you take medication?")
- Removed duplicates (e.g., self-reported height when measured height available)
- Removed awkward formats (e.g., wake-up time as HH:MM)

**Why:** Survey makes clinical sense to domain experts; improves user experience.

---

## Architecture

- **ML Framework:** [AutoGluon TabularPredictor](https://auto.gluon.ai/stable/index.html)
  - Ensemble of gradient boosting machines, neural networks, and linear models
  - 8-fold bagging for robustness
- **Training Data:** NHANES 1999-2017, adults 18+ (n≈15,000 unique participants, 45,000 after augmentation)
- **Inference:** Flask API serving predictions in <200ms

---

## Performance Benchmarks

### Error vs. Feature Count

| Features | MAE (years) | Relative to Baseline |
|----------|-------------|----------------------|
| 0 (predict mean age) | 27.99 | - |
| 120 (production) | 5.93 | **-79%** |
| 350 (full candidate set) | 5.10 | -82% |

### Robustness to Missing Data

| % Missing | MAE | Notes |
|-----------|-----|-------|
| 0% | 5.93 | All questions answered |
| 20% | 6.15 | Light masking tier |
| 40% | 6.89 | Heavy masking tier |

---

## Getting Started

For API integration and technical details, see:
- **[API Documentation](docs/API.md)** - Endpoints, request/response formats
- **[Methodology Deep-Dive](docs/METHODOLOGY.md)** - Complete scientific approach and validation studies


---

## Limitations

1. **Population specificity:** Trained on US adults; performance on other demographics unknown
2. **Temporal drift:** NHANES protocols evolve; model may require retraining on newer data cycles
3. **Self-report bias:** Assumes honest responses; no adversarial robustness testing conducted
4. **Missing data assumptions:** Dilution strategy assumes MCAR (missing completely at random) after preprocessing

---

