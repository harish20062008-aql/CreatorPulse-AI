
import pandas as pd
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Input and output paths
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CONTENT_FILE = RAW_DIR / "content_performance.csv"
COMMENTS_FILE = RAW_DIR / "audience_comments.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_content_data():
    print("\n--- Cleaning Content Performance Data ---")

    df = pd.read_csv(CONTENT_FILE)

    print("Original shape:", df.shape)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove duplicate post IDs, keeping the first record
    df = df.drop_duplicates(subset=["post_id"], keep="first")

    # Convert numeric columns
    numeric_columns = [
        "posting_hour", "views", "likes", "shares",
        "saves", "comments", "retention_rate",
        "follower_change"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Convert dates
    df["post_date"] = pd.to_datetime(
        df["post_date"], errors="coerce"
    )

    # Remove rows missing essential information
    df = df.dropna(subset=["post_id", "post_date"])

    # Save cleaned data
    output_file = PROCESSED_DIR / "cleaned_content.csv"
    df.to_csv(output_file, index=False)

    print("Cleaned shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Saved:", output_file)


def clean_audience_comments():
    print("\n--- Cleaning Audience Comments Data ---")

    df = pd.read_csv(COMMENTS_FILE)

    print("Original shape:", df.shape)

    # Remove duplicate rows and comment IDs
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["comment_id"], keep="first")

    # Remove comments with missing essential information
    df = df.dropna(subset=["comment_id", "comment_text"])

    # Remove extra spaces from text
    df["comment_text"] = df["comment_text"].astype(str).str.strip()

    # Remove empty comments
    df = df[df["comment_text"] != ""]

    # Normalize labels
    df["sentiment_label"] = (
        df["sentiment_label"].astype("string").str.strip().str.lower()
    )

    # Convert dates
    df["comment_date"] = pd.to_datetime(
        df["comment_date"], errors="coerce"
    )

    # Save cleaned data
    output_file = PROCESSED_DIR / "analyzed_comments.csv"
    df.to_csv(output_file, index=False)

    print("Cleaned shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Sentiment counts:\n", df["sentiment_label"].value_counts())
    print("Saved:", output_file)


if __name__ == "__main__":
    try:
        clean_content_data()
        clean_audience_comments()
        print("\nData cleaning completed successfully!")

    except FileNotFoundError as error:
        print("\nERROR: A CSV file could not be found.")
        print(error)
        print("Check that both CSV files are inside data/raw.")

    except Exception as error:
        print("\nAn error occurred:", error)
