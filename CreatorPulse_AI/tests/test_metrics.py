import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Datasets and model files
DATA_FILE = BASE_DIR / "data" / "processed" / "content_engagement_metrics.csv"
MODEL_FILE = BASE_DIR / "models" / "virality_model.pkl"

from src.engagement_metrics import detect_anomalies, compute_content_dna
from src.virality_prediction import load_virality_model, compute_feature_contributions
from src.recommendations import generate_recommendations
from src.ab_testing import compute_format_comparison
from src.sentiment_analysis import train_sentiment_model


@pytest.fixture
def content_data():
    assert DATA_FILE.exists(), "Engagement metrics CSV file not found."
    return pd.read_csv(DATA_FILE)


def test_dataset_not_empty(content_data):
    assert not content_data.empty


def test_required_columns_exist(content_data):
    required_columns = [
        "post_id", "post_date", "topic", "format", "hook_type",
        "caption_style", "posting_hour", "views", "likes", "shares",
        "saves", "comments", "total_interactions", "engagement_rate",
        "engagement_rank"
    ]
    for column in required_columns:
        assert column in content_data.columns, f"Missing column {column}"


def test_post_ids_are_unique(content_data):
    assert content_data["post_id"].is_unique


def test_engagement_rate_is_valid(content_data):
    rates = pd.to_numeric(content_data["engagement_rate"], errors="coerce")
    assert rates.notna().all()
    assert (rates >= 0).all()


def test_total_posts(content_data):
    assert len(content_data) == 50


def test_anomaly_detection_iqr(content_data):
    df_anom, meta = detect_anomalies(content_data)
    assert not df_anom.empty
    assert "is_anomaly" in df_anom.columns
    assert "anomaly_type" in df_anom.columns
    assert meta["iqr"] > 0
    assert meta["upper_bound"] > meta["q3"]


def test_content_dna_dimensions(content_data):
    dna = compute_content_dna(content_data, min_sample=3)
    assert "dimensions" in dna
    assert "topic_format_matrix" in dna
    assert "format" in dna["dimensions"]
    assert "topic" in dna["dimensions"]
    format_df = dna["dimensions"]["format"]
    assert "is_sufficient_sample" in format_df.columns
    assert "avg_engagement" in format_df.columns


def test_virality_model_class_mapping():
    model, meta = load_virality_model()
    assert model is not None, "Virality model failed to load."
    assert "target_names" in meta, "Target names mapping missing."
    assert 0 in meta["target_names"], "Class 0 missing in target names."
    assert 1 in meta["target_names"], "Class 1 missing in target names."
    assert "High Engagement" in meta["target_names"][1]
    assert "Standard / Moderate" in meta["target_names"][0]


def test_virality_model_probability_sum():
    model, meta = load_virality_model()
    test_input = pd.DataFrame([{
        "topic": "Comedy",
        "format": "Short",
        "hook_type": "Question",
        "caption_style": "Funny",
        "posting_hour": 18
    }])
    probs = model.predict_proba(test_input)[0]
    assert len(probs) == 2
    assert np.isclose(probs.sum(), 1.0, atol=1e-5), "Probabilities must sum to 1"


def test_feature_contributions_output():
    model, meta = load_virality_model()
    test_input = pd.DataFrame([{
        "topic": "Education",
        "format": "Short",
        "hook_type": "Question",
        "caption_style": "Story",
        "posting_hour": 19
    }])
    contribs = compute_feature_contributions(model, test_input)
    assert isinstance(contribs, dict)
    assert "format" in contribs
    assert "delta_probability" in contribs["format"]


def test_recommendations_support_and_caveats(content_data):
    rec_data = generate_recommendations(content_data)
    assert rec_data is not None
    assert "recommendations" in rec_data
    for rec in rec_data["recommendations"]:
        assert "sample_count" in rec
        assert "status" in rec
        assert "caveat" in rec
        assert "suggested_experiment" in rec


# === NEW FOCUSED TESTS FOR CONFIRMED ISSUES ===

def test_ab_testing_multi_format(content_data):
    """Verify that multi-format comparison calculates correct descriptive difference."""
    res = compute_format_comparison(content_data)
    assert res["status"] == "multi_format"
    assert res["best_format"]["format"] == "Short"
    assert res["second_best"]["format"] == "Image"
    assert np.isclose(res["difference"], 0.5548, atol=0.01)


def test_ab_testing_single_format_robustness(content_data):
    """Verify that filtering to a single format does not crash with IndexError."""
    single_fmt_df = content_data[content_data["format"] == "Short"]
    res = compute_format_comparison(single_fmt_df)
    assert res["status"] == "single_format"
    assert res["best_format"]["format"] == "Short"
    assert res["second_best"] is None
    assert res["difference"] is None
    assert "Only one format ('Short'" in res["message"]


def test_ab_testing_empty_robustness():
    """Verify that empty subsets return status='empty' gracefully."""
    empty_df = pd.DataFrame(columns=["format", "engagement_rate"])
    res = compute_format_comparison(empty_df)
    assert res["status"] == "empty"
    assert res["data"].empty
    assert res["difference"] is None


def test_sentiment_model_stratified_cv():
    """Verify that stratified 5-fold CV evaluates deduplicated comments to exact measured 65.00%."""
    res = train_sentiment_model()
    assert "cv_mean" in res
    assert np.isclose(res["cv_mean"], 0.65, atol=1e-4), f"Expected 65.00% CV, got {res['cv_mean']}"
    assert res["total_dedup_samples"] == 23
    assert len(res["cv_scores"]) == 5
