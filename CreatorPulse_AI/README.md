# Data-Driven Social Engagement Initiative
### *CreatorPulse AI • Explainable Social Media Intelligence & Content Optimization Platform*

[![Python Version](https://img.shields.io/badge/python-3.13.7-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%201.66-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%201.9-F7931E.svg)](https://scikit-learn.org/)
[![Testing](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Dataset](https://img.shields.io/badge/dataset-synthetic%20sample%20(50%20posts)-lightgrey.svg)]()

---

## 1. Project Overview

The **Data-Driven Social Engagement Initiative** (with optional product branding **CreatorPulse AI**) is an evidence-aware analytics and machine learning system designed to inspect social media content performance, understand audience sentiment, predict engagement categories, simulate content adjustments, and plan testable creator experiments.

Rather than treating machine learning outputs as black-box guarantees, this project prioritizes **data correctness, uncertainty communication, and explainability**. It is built on a transparent synthetic sample of 50 posts and 150 audience comments, designed for academic demonstration.

---

## 2. Key Objectives & Methodological Guardrails

- **Descriptive Performance Analysis:** Aggregate verified interactions (likes, shares, saves, comments) against views without mixing denominators.
- **Explainable Engagement Prediction:** Map classifier predictions directly to meaningful human-readable labels (`High Engagement Potential (>= 11.17%)` vs `Standard / Moderate Engagement (< 11.17%)`), accompanied by tree-ensemble confidence and local feature contributions.
- **What-If Content Simulation:** Allow creators to compare planned content scenarios side-by-side against a baseline without claiming causal effects.
- **Content DNA Profiling:** Provide multi-dimensional heatmaps and segment summaries across format, topic, hook type, and posting hour, while explicitly suppressing or qualifying low-sample groups ($N < 3$).
- **Statistical Anomaly Detection:** Flag genuine high-engagement spikes and low dips using robust Interquartile Range ($\text{IQR}$) thresholds.
- **Evidence-Linked Strategy Planner:** Convert data observations into testable hypotheses with defined metrics, sample goals, and an audit-friendly local log.

---

## 3. Technologies Used

- **Language:** Python 3.13.7
- **Data Manipulation:** Pandas 3.1, NumPy 2.5
- **Machine Learning & NLP:** Scikit-Learn 1.9, Joblib 1.6
- **Data Visualization:** Plotly Express & Plotly Graph Objects 7.1, Matplotlib 3.11
- **Web Application:** Streamlit 1.66
- **Automated Testing:** Pytest 9.2

---

## 4. Project Directory Structure

```text
CreatorPulse_AI/
├── app.py                         # Interactive multi-page Streamlit application
├── README.md                      # Comprehensive academic documentation and setup guide
├── requirements.txt               # Pinned dependencies
├── .env                           # Environment and project configuration
│
├── data/
│   ├── raw/
│   │   ├── content_performance.csv  # 50 rows × 14 cols (Synthetic content performance)
│   │   └── audience_comments.csv    # 150 rows × 5 cols (Synthetic audience feedback)
│   └── processed/
│       ├── cleaned_content.csv             # Deduplicated, typed post dataset
│       ├── content_engagement_metrics.csv  # Engagement rates, ranks, total interactions
│       ├── analyzed_comments.csv           # Normalized comment feedback
│       ├── ab_testing_results.csv          # Format aggregates and comparisons
│       └── experiments.json                # Local persistent log of creator experiments
│
├── models/
│   ├── virality_model.pkl         # Trained RandomForest Pipeline + metadata artifact
│   └── sentiment_model.pkl        # TF-IDF + LogisticRegression NLP model
│
├── reports/
│   ├── project_findings.md        # Academic summary report of original findings
│   ├── trend_forecast.csv         # Daily engagement, rolling average, and linear projection
│   └── engagement_trend.png       # Matplotlib visualization of historical trend
│
├── src/
│   ├── data_cleaning.py           # Data validation, type normalization, and deduplication
│   ├── engagement_metrics.py      # Metric calculations, IQR anomaly detection, Content DNA
│   ├── virality_prediction.py     # Random Forest training, evaluation, feature contribution
│   ├── sentiment_analysis.py      # Comment NLP classifier training and evaluation
│   ├── ab_testing.py              # Observational format A/B comparison framework
│   ├── recommendations.py         # Evidence-linked recommendations with sample counts
│   └── trend_forecasting.py       # Time series smoothing, trend projection & MAE baselines
│
├── tests/
│   └── test_metrics.py            # Automated test suite (15 unit tests)
│
└── notebooks/                     # Exploratory research notebooks
```

---

## 5. Dataset Profile & Schema

The initiative uses synthetic datasets representing a creator's publishing history over a 3-month window (September 1, 2026 to December 8, 2026):

### Content Performance Dataset (`50 posts × 14 features`)
- **Identifiers & Context:** `post_id` (P001–P050), `post_date` (YYYY-MM-DD), `posting_hour` (int, 7–22).
- **Categorical Attributes:** `topic` (8 unique: Comedy, Technology, Gaming, Friendship, Education, Food, Motivation, Lifestyle), `format` (Reel, Short, Image, Carousel), `hook_type` (5 unique), `caption_style` (5 unique).
- **Engagement Counts:** `views` (sum: 652,018), `likes` (sum: 47,779), `shares` (sum: 9,040), `saves` (sum: 7,439), `comments` (sum: 4,502).
- **Secondary Metrics:** `retention_rate` (float, %), `follower_change` (int).

### Audience Comments Dataset (`150 comments × 5 features`)
- **Identifiers:** `comment_id` (C001–C150), `post_id` (48 posts represented, 0 orphan comments), `comment_date`.
- **Text & Classification:** `comment_text`, `sentiment_label` (`relatable`: 84, `neutral`: 47, `unrelated`: 19).

---

## 6. Metric Formulas & Verified Results

| Metric | Formula | Verified Result | Denominator / Context |
|---|---|---|---|
| **Total Verified Views** | $\sum \text{views}$ | **652,018** | Across all 50 sample posts |
| **Total Interactions** | $\sum (\text{likes} + \text{shares} + \text{saves} + \text{comments})$ | **68,760** | Likes: 47,779, Shares: 9,040, Saves: 7,439, Comments: 4,502 |
| **Engagement Rate (Post)** | $\frac{\text{likes} + \text{shares} + \text{saves} + \text{comments}}{\text{views}} \times 100$ | Range: 5.72% – 15.67% | Per post rate, rounded to 2 decimal places |
| **Macro Average Engagement** | $\frac{1}{N} \sum_{i=1}^{N} \text{engagement\_rate}_i$ | **10.59%** | Unweighted mean of post rates ($N=50$) |
| **Weighted Average Engagement** | $\frac{\sum \text{total\_interactions}}{\sum \text{views}} \times 100$ | **10.55%** | 68,760 interactions / 652,018 views |
| **Format A/B Top Difference** | $\text{Mean}_{\text{Short}} - \text{Mean}_{\text{Image}}$ | **0.55 percentage points** | Short: 11.74% ($N=11$) vs Image: 11.18% ($N=9$). Purely descriptive sample difference across all data. |
| **Format Recommendation** | $\arg\max \text{Mean}(\text{format})$ | **Short** (11.74%) | Sample support: $N = 11$ posts |
| **Topic Recommendation** | $\arg\max \text{Mean}(\text{topic})$ | **Lifestyle** (12.41%) | **Caveat:** Supported by only $N = 2$ posts (4% of dataset); flagged as low sample exploratory hypothesis |
| **Posting Hour Recommendation**| $\arg\max \text{Mean}(\text{hour})$ | **18:00** (13.33%) | **Caveat:** Supported by only $N = 4$ posts; requires local timezone validation |

---

## 7. Machine Learning Pipeline & Model Transparency

### 1. Virality Prediction Model
- **Task:** Binary classification estimating whether a planned post will meet or exceed the historical dataset median engagement rate ($11.17\%$).
- **Pipeline:** `ColumnTransformer(OneHotEncoder(categories) + passthrough(posting_hour))` + `RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)`.
- **Target Mapping:**
  - Class `0`: **Standard / Moderate Engagement** ($< 11.17\%$)
  - Class `1`: **High Engagement Potential** ($\ge 11.17\%$)
- **Evaluation:**
  - Single held-out test split ($N=10$): **$60.0\%$** accuracy.
  - 5-Fold Stratified Cross-Validation: **$58.0\%$** mean accuracy. *(Reveals realistic generalization constraints on a 50-row synthetic sample; avoiding exaggerated claims).*
- **Local Feature Contributions:** Evaluates local probability shift when perturbing each input feature against the baseline population distribution, answering *"Why did the model assign this confidence?"*

### 2. Audience Sentiment Classifier
- **Pipeline:** `TfidfVectorizer(ngram_range=(1,2))` + `LogisticRegression(class_weight='balanced', random_state=42)`.
- **Target Classes:** `['neutral', 'relatable', 'unrelated']`.
- **Sample & Deduplication:** The raw dataset contains 150 comments with repeated template texts. To prevent train-to-test leakage, `src/sentiment_analysis.py` trains and evaluates on deduplicated unique comments ($N=23$).
- **5-Fold Stratified Cross-Validation Accuracy:** **$65.00\%$** (per-fold: 60.0%, 40.0%, 100.0%, 50.0%, 75.0%).
- **Held-out Split Accuracy:** **$66.67\%$** (17 train / 6 test).
- **Academic Leakage Warning:** Evaluating across all 150 comments without deduplicating repeated comment strings inflates measured accuracy to ~98% because identical phrases appear in both training and testing folds. True cross-validated generalization on unique comment text is 65.00%.

---

## 8. Dashboard Modules (Streamlit)

The interactive dashboard includes 9 comprehensive sections:

1. **Overview & Executive KPIs:** Macro vs weighted engagement, verified views and interactions, engagement timeline scatter plot, and top-performing posts table.
2. **Engagement Analysis & A/B Testing:** Format comparison bar chart responding dynamically to the active sidebar topic and format filters with active scope badges ($N$). Gracefully handles single-format or low-sample selections without indexing errors, accompanied by an expandable complete dataset baseline benchmark.
3. **Audience Sentiment Intelligence:** Comment category distribution pie chart and searchable feedback table.
4. **Virality Prediction & Explainable AI:** Input form with human-readable class mapping, ensemble confidence breakdown, and local feature contribution table.
5. **What-If Content Simulator:** Side-by-side Baseline vs. Scenario comparison cards with delta confidence metrics and highlighted parameter changes.
6. **Content DNA Analyzer:** Multi-dimensional matrix heatmap (Topic × Format) and attribute distribution with dynamic minimum sample size ($N \ge 3$) qualification.
7. **Engagement Anomaly Detection:** Robust Interquartile Range ($\text{IQR}$) bounds isolating genuine viral spikes from baseline fluctuations.
8. **Trend Forecasting & Baselines:** Interactive Plotly time series tracking daily engagement, 3-point rolling average, linear trend projection, and Mean Absolute Error ($\text{MAE}$) baseline comparison against naive predictors.
9. **Strategy & Experiment Planner:** Evidence-linked recommendation cards and a structured experiment planner saved to local persistent JSON (`data/processed/experiments.json`).

---

## 9. Installation & Reproduction Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13.7)
- Windows, macOS, or Linux

### Setup Instructions

1. **Clone or enter the project directory:**
   ```bash
   cd CreatorPulse_AI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run pipeline modules:**
   ```bash
   python src/data_cleaning.py
   python src/engagement_metrics.py
   python src/virality_prediction.py
   python src/sentiment_analysis.py
   python src/ab_testing.py
   python src/recommendations.py
   python src/trend_forecasting.py
   ```

5. **Run automated unit tests:**
   ```bash
   python -m pytest tests/test_metrics.py
   ```
   *Expected output: 15 passed.*

6. **Launch the interactive Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```
   Open the local URL displayed in the terminal (typically `http://localhost:8501`).

---

## 10. Acceptance Verification Summary

| Requirement | Implementation Verification | Status |
|---|---|---|
| **Human-readable prediction mapping** | Outputs mapped to `"High Engagement Potential (>= 11.17%)"` or `"Standard / Moderate Engagement (< 11.17%)"` via `model.classes_` and `meta['target_names']`. Never displays raw integer classes. | **Verified** |
| **Calibrated & honest confidence display** | Displays ensemble tree vote breakdown, verifies $\sum P = 1.0$, and explicitly labels values as uncalibrated random forest votes on $N=50$. | **Verified** |
| **Interactive trend chart mapping** | Replaced static PNG with an interactive Plotly time series directly bound to dates and engagement % with matching title, axes, and legend. | **Verified** |
| **Metric & denominator transparency** | Exposes total views (652,018), macro-average (10.59%), weighted average (10.55%), and small-sample warnings ($N=2$ for Lifestyle, $N=4$ for 18:00). | **Verified** |
| **Format A/B Comparison Filter & Robustness** | Responds dynamically to sidebar topic filters with active scope display; handles single-format and empty subsets gracefully with informative notices instead of errors. | **Verified** |
| **Sentiment Classifier Evaluation** | Evaluated via reproducible 5-fold Stratified Cross-Validation on deduplicated comments ($N=23$), achieving measured 65.00% CV accuracy (and 66.67% on held-out test split). Avoids test leakage inflation. | **Verified** |
| **Explainable AI (XAI)** | Local feature contribution analysis displays direction and probability shift for each input parameter. | **Verified** |
| **What-If Content Simulator** | Side-by-side Baseline vs Scenario comparison with delta metrics and parameter highlights. | **Verified** |
| **Content DNA Analyzer** | Topic × Format heatmap matrix with supporting post counts and minimum sample size suppression rule. | **Verified** |
| **Statistical Anomaly Detection** | IQR upper ($14.65\%$) and lower ($5.85\%$) bounds isolating genuine viral outliers. | **Verified** |
| **Strategy & Experiment Planner** | Evidence-linked cards with candidate status, experiment suggestions, and local JSON persistence. | **Verified** |
| **Automated Test Suite** | 15 unit tests covering schemas, metrics, models, single-format robustness, probability sums, XAI, and caveats passing cleanly. | **Verified** |

---

## 11. Limitations & Ethical Disclosures

1. **Synthetic Data Constraint:** This initiative uses synthetic benchmark datasets (50 posts, 150 comments). Patterns, correlations, and distributions may reflect generative artifacts rather than real-world creator dynamics.
2. **Correlation vs. Causation:** Observed differences (e.g. Short format averaging 0.55 percentage points higher than Image) are descriptive associations. Changing format or topic does not guarantee an increase in reach or engagement.
3. **Model Generalization:** The virality classifier achieves ~58% cross-validation accuracy on this 50-row sample; the sentiment classifier achieves 65.00% cross-validation accuracy on 23 unique comments. Predictions must be treated as exploratory hypothesis generators, never as production decision engines or virality guarantees.
4. **Sentiment Nuance:** The text classifier operates on a restricted vocabulary (`relatable`, `neutral`, `unrelated`) and may misinterpret sarcasm, regional dialects, or informal vernacular.
5. **Timing & Audience Context:** Suggested posting times (e.g. 18:00) depend on audience demographics, local timezones, and platform delivery schedules.
