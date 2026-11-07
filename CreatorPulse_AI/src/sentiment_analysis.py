import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "raw" / "audience_comments.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "sentiment_model.pkl"


def train_sentiment_model():
    print("Loading audience comments...")

    df = pd.read_csv(DATA_FILE)

    # Check required columns
    required = ["comment_text", "sentiment_label"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Clean text and labels
    df = df.dropna(subset=required).copy()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df["sentiment_label"] = (
        df["sentiment_label"].astype(str).str.strip().str.lower()
    )
    df = df[df["comment_text"] != ""]
    
    # Deduplicate comments to prevent test leakage in tiny synthetic datasets
    df_train = df.drop_duplicates(subset=["comment_text", "sentiment_label"]).copy()

    total_raw_comments = len(df)
    total_dedup_comments = len(df_train)

    print(f"Total raw comments in dataset: {total_raw_comments}")
    print(f"Unique deduplicated comments available for training: {total_dedup_comments}")
    print("\nDeduplicated label counts:")
    print(df_train["sentiment_label"].value_counts())

    if df_train["sentiment_label"].nunique() < 2:
        raise ValueError("At least two different labels are required.")

    if df_train["sentiment_label"].value_counts().min() < 2:
        raise ValueError("Each label needs at least two examples.")

    X = df_train["comment_text"]
    y = df_train["sentiment_label"]

    # TF-IDF converts text into numerical features
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    # 1. Reproducible Stratified 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    cv_mean = float(cv_scores.mean())

    print("\n===== 5-FOLD STRATIFIED CROSS-VALIDATION =====")
    print(f"Folds: {[round(s, 4) for s in cv_scores]}")
    print(f"Mean CV Accuracy: {cv_mean:.2%}")

    # 2. Held-out test split evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\nTraining sentiment model on held-out split...")
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    test_acc = float(accuracy_score(y_test, predictions))

    print("\n===== HELD-OUT TEST SPLIT EVALUATION =====")
    print(f"Training comments: {len(X_train)} | Testing comments: {len(X_test)}")
    print(f"Accuracy: {test_acc:.2%}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save trained model and evaluation metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model pipeline
    joblib.dump(model, MODEL_FILE)
    print("Model saved to:", MODEL_FILE)

    # Example predictions for sample inputs
    new_comments = [
        "I can relate to this so much",
        "Thanks for sharing",
        "What phone do you use?"
    ]

    print("\nExample Predictions")
    print("-------------------")
    for comment, label in zip(new_comments, model.predict(new_comments)):
        print(f"Comment: '{comment}' -> Predicted category: {label}")

    return {
        "model": model,
        "cv_scores": cv_scores.tolist(),
        "cv_mean": cv_mean,
        "test_acc": test_acc,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_dedup_samples": total_dedup_comments
    }


if __name__ == "__main__":
    try:
        train_sentiment_model()
    except FileNotFoundError:
        print("ERROR: audience_comments.csv was not found.")
        print("Check that the file is inside data/raw.")
    except Exception as error:
        print("ERROR:", error)
