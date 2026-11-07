import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "raw" / "content_performance.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "virality_model.pkl"

CATEGORICAL_FEATURES = [
    "topic",
    "format",
    "hook_type",
    "caption_style"
]
NUMERICAL_FEATURES = ["posting_hour"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def train_virality_model():
    print("Loading content performance dataset...")

    df = pd.read_csv(DATA_FILE)

    required = (
        ALL_FEATURES
        + ["views", "likes", "shares", "saves", "comments"]
    )

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Calculate historical engagement to create training labels
    engagement_columns = ["likes", "shares", "saves", "comments"]

    for column in ["views"] + engagement_columns + NUMERICAL_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required).copy()
    df = df[df["views"] > 0].copy()

    df["engagement_rate"] = (
        df[engagement_columns].sum(axis=1) / df["views"] * 100
    )

    # Define high engagement as at or above the dataset median
    threshold = float(df["engagement_rate"].median())
    df["performance"] = (
        df["engagement_rate"] >= threshold
    ).astype(int)

    X = df[ALL_FEATURES]
    y = df["performance"]

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError(
            "Not enough examples in each class to train the model."
        )

    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Convert text categories into numeric features
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            ("numbers", "passthrough", NUMERICAL_FEATURES)
        ]
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    print("Training machine learning model...")
    model.fit(X_train, y_train)

    # Cross-validation assessment across full sample to measure realistic generalization
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    cv_mean = float(cv_scores.mean())

    # Evaluate using held-out test data
    predictions = model.predict(X_test)
    test_acc = float(accuracy_score(y_test, predictions))

    print("\nModel evaluation")
    print("----------------")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print(f"High-engagement threshold (median): {threshold:.2f}%")
    print(f"Held-out test split accuracy: {test_acc:.2%}")
    print(f"5-Fold Stratified CV Mean Accuracy: {cv_mean:.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Standard/Moderate", "High Engagement"],
            zero_division=0
        )
    )

    # Save trained model and comprehensive metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    target_names = {
        0: f"Standard / Moderate Engagement (< {threshold:.2f}%)",
        1: f"High Engagement Potential (>= {threshold:.2f}%)"
    }

    # Baseline class distribution in training data
    class_balance = {
        int(k): int(v) for k, v in y.value_counts().to_dict().items()
    }

    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "features": ALL_FEATURES,
            "feature_types": {
                "categorical": CATEGORICAL_FEATURES,
                "numerical": NUMERICAL_FEATURES
            },
            "classes": [int(c) for c in model.classes_],
            "target_names": target_names,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_accuracy": test_acc,
            "cv_mean_accuracy": cv_mean,
            "class_balance": class_balance,
            "mean_engagement_rate": float(df["engagement_rate"].mean()),
            "median_engagement_rate": threshold,
            "version": "1.1.0-corrected"
        },
        MODEL_FILE
    )

    print("Model and metadata saved to:", MODEL_FILE)


def load_virality_model():
    """Load model artifact and return model and metadata dictionary."""
    if not MODEL_FILE.exists():
        return None, {}
    artifact = joblib.load(MODEL_FILE)
    if isinstance(artifact, dict):
        model = artifact.get("model")
        metadata = artifact
    else:
        model = artifact
        metadata = {
            "model": model,
            "features": ALL_FEATURES,
            "threshold": 11.17,
            "target_names": {
                0: "Standard / Moderate Engagement (< 11.17%)",
                1: "High Engagement Potential (>= 11.17%)"
            },
            "classes": list(getattr(model, "classes_", [0, 1]))
        }
    return model, metadata


def compute_feature_contributions(model, input_df, baseline_df=None):
    """
    Compute local feature contributions (associative impact) of each input
    relative to the baseline training distribution.
    Uses leave-one-feature-marginalized approach to explain probability shift.
    """
    if not hasattr(model, "predict_proba"):
        return {}

    classes = list(model.classes_)
    high_idx = 1 if 1 in classes else (classes.index(1) if 1 in classes else -1)
    if high_idx == -1 and len(classes) > 1:
        high_idx = 1

    base_prob = float(model.predict_proba(input_df)[0][high_idx])

    # If baseline dataset is available, use representative modes/medians
    if baseline_df is None and DATA_FILE.exists():
        baseline_df = pd.read_csv(DATA_FILE)

    contributions = {}
    if baseline_df is not None and not baseline_df.empty:
        # Default baseline values
        defaults = {
            "topic": baseline_df["topic"].mode()[0] if "topic" in baseline_df else "Lifestyle",
            "format": baseline_df["format"].mode()[0] if "format" in baseline_df else "Reel",
            "hook_type": baseline_df["hook_type"].mode()[0] if "hook_type" in baseline_df else "Question",
            "caption_style": baseline_df["caption_style"].mode()[0] if "caption_style" in baseline_df else "Casual",
            "posting_hour": int(baseline_df["posting_hour"].median()) if "posting_hour" in baseline_df else 18
        }
        for feat in ALL_FEATURES:
            if feat in input_df.columns:
                # Perturb to baseline value
                perturbed = input_df.copy()
                perturbed[feat] = defaults.get(feat, perturbed[feat].iloc[0])
                perturbed_prob = float(model.predict_proba(perturbed)[0][high_idx])
                # Delta: how much does the current choice add compared to default?
                delta = base_prob - perturbed_prob
                contributions[feat] = {
                    "value": input_df[feat].iloc[0],
                    "baseline_value": defaults.get(feat),
                    "delta_probability": delta,
                    "direction": "positive" if delta > 0.005 else ("negative" if delta < -0.005 else "neutral")
                }
    return contributions


if __name__ == "__main__":
    try:
        train_virality_model()
    except FileNotFoundError:
        print("Dataset not found. Check data/raw/content_performance.csv")
    except Exception as error:
        print("Error:", error)
