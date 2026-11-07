import pandas as pd
import numpy as np
from pathlib import Path

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "cleaned_content.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "content_engagement_metrics.csv"


def calculate_engagement():
    print("Loading cleaned content dataset...")

    df = pd.read_csv(INPUT_FILE)

    # Check required columns
    required_columns = [
        "post_id", "views", "likes", "shares", "saves", "comments"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Convert engagement columns to numbers
    numeric_columns = ["views", "likes", "shares", "saves", "comments"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Fill missing interaction counts with zero
    interaction_columns = ["likes", "shares", "saves", "comments"]
    df[interaction_columns] = df[interaction_columns].fillna(0)

    # Avoid invalid negative counts
    for column in numeric_columns:
        df.loc[df[column] < 0, column] = pd.NA

    # Remove rows with invalid views or missing post IDs
    df = df.dropna(subset=["post_id", "views"])
    df = df[df["views"] > 0].copy()

    # Calculate total interactions: Likes + Shares + Saves + Comments
    df["total_interactions"] = (
        df["likes"] + df["shares"] + df["saves"] + df["comments"]
    )

    # Calculate engagement rate as a percentage of views: (total_interactions / views) * 100
    df["engagement_rate"] = (
        df["total_interactions"] / df["views"] * 100
    ).round(2)

    # Rank posts: rank 1 means the highest engagement rate
    df["engagement_rank"] = (
        df["engagement_rate"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Sort best-performing posts first
    df = df.sort_values(
        by="engagement_rate", ascending=False
    )

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset analysis completed!")
    print("Total posts analyzed:", len(df))
    macro_avg = df["engagement_rate"].mean()
    weighted_avg = (df["total_interactions"].sum() / df["views"].sum()) * 100
    print(f"Macro-average engagement rate: {macro_avg:.2f}%")
    print(f"Weighted engagement rate (sum interactions / sum views): {weighted_avg:.2f}%")

    print("\nTop 10 Best-Performing Posts:")
    display_columns = [
        "post_id", "topic", "format", "views",
        "total_interactions", "engagement_rate", "engagement_rank"
    ]

    available_columns = [
        col for col in display_columns if col in df.columns
    ]
    print(df[available_columns].head(10).to_string(index=False))

    print("\nResults saved to:")
    print(OUTPUT_FILE)
    return df


def detect_anomalies(df=None, method="iqr"):
    """
    Detect statistical engagement anomalies using robust Interquartile Range (IQR).
    Separates genuine high-engagement spikes from data inconsistencies.
    """
    if df is None:
        if not OUTPUT_FILE.exists():
            return pd.DataFrame()
        df = pd.read_csv(OUTPUT_FILE)

    if df.empty or "engagement_rate" not in df.columns:
        return pd.DataFrame()

    df_out = df.copy()
    rates = df_out["engagement_rate"].dropna()

    q1 = rates.quantile(0.25)
    q3 = rates.quantile(0.75)
    iqr = q3 - q1

    upper_bound = q3 + 1.5 * iqr
    lower_bound = max(0.0, q1 - 1.5 * iqr)

    # Flag outliers
    is_high_spike = df_out["engagement_rate"] > upper_bound
    is_low_dip = df_out["engagement_rate"] < lower_bound

    df_out["is_anomaly"] = is_high_spike | is_low_dip
    df_out["anomaly_type"] = "Normal"
    df_out.loc[is_high_spike, "anomaly_type"] = "High Spike"
    df_out.loc[is_low_dip, "anomaly_type"] = "Low Dip"

    df_out["anomaly_details"] = "Within expected distribution"
    df_out.loc[is_high_spike, "anomaly_details"] = (
        f"Engagement exceeds upper IQR threshold ({upper_bound:.2f}%)"
    )
    df_out.loc[is_low_dip, "anomaly_details"] = (
        f"Engagement falls below lower IQR threshold ({lower_bound:.2f}%)"
    )

    metadata = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "spike_count": int(is_high_spike.sum()),
        "dip_count": int(is_low_dip.sum())
    }

    return df_out, metadata


def compute_content_dna(df=None, min_sample=3):
    """
    Compute multi-dimensional Content DNA profiles across supported features:
    format, topic, hook_type, caption_style, posting_hour.
    Suppresses or qualifies tiny groups below min_sample.
    """
    if df is None:
        if not OUTPUT_FILE.exists():
            return {}
        df = pd.read_csv(OUTPUT_FILE)

    if df.empty:
        return {}

    dimensions = ["format", "topic", "hook_type", "caption_style", "posting_hour"]
    dna_summary = {}

    for dim in dimensions:
        if dim not in df.columns:
            continue

        grouped = (
            df.groupby(dim)["engagement_rate"]
            .agg(
                posts="count",
                avg_engagement="mean",
                std_engagement="std",
                min_engagement="min",
                max_engagement="max"
            )
            .reset_index()
        )

        # Standard error of mean: std / sqrt(n)
        grouped["std_engagement"] = grouped["std_engagement"].fillna(0)
        grouped["sem"] = grouped["std_engagement"] / np.sqrt(grouped["posts"])
        grouped["is_sufficient_sample"] = grouped["posts"] >= min_sample
        grouped["status"] = np.where(
            grouped["is_sufficient_sample"],
            "Sufficient Sample",
            f"Low Sample (N<{min_sample})"
        )
        grouped = grouped.sort_values(by="avg_engagement", ascending=False)
        dna_summary[dim] = grouped

    # 2D Matrix: Topic x Format
    matrix_mean = pd.pivot_table(
        df,
        values="engagement_rate",
        index="topic",
        columns="format",
        aggfunc="mean"
    )

    matrix_count = pd.pivot_table(
        df,
        values="engagement_rate",
        index="topic",
        columns="format",
        aggfunc="count"
    ).fillna(0).astype(int)

    return {
        "dimensions": dna_summary,
        "topic_format_matrix": matrix_mean,
        "topic_format_counts": matrix_count
    }


if __name__ == "__main__":
    try:
        calculate_engagement()
    except FileNotFoundError:
        print("ERROR: cleaned_content.csv was not found.")
        print("Run this command first: python src/data_cleaning.py")
    except Exception as error:
        print("ERROR:", error)
