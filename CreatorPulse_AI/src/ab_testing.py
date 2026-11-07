import pandas as pd
from pathlib import Path

# Project folder paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "content_engagement_metrics.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "ab_testing_results.csv"


def compute_format_comparison(df=None):
    """
    Compute descriptive format performance metrics safely.
    Handles empty subsets and single-format data without IndexError.
    """
    if df is None:
        if not DATA_FILE.exists():
            return {
                "status": "empty",
                "message": "Dataset not found.",
                "data": pd.DataFrame(),
                "best_format": None,
                "difference": None
            }
        df = pd.read_csv(DATA_FILE)

    required_columns = {"format", "engagement_rate"}
    if not required_columns.issubset(df.columns):
        return {
            "status": "empty",
            "message": "Required columns 'format' and 'engagement_rate' are missing.",
            "data": pd.DataFrame(),
            "best_format": None,
            "difference": None
        }

    clean_df = df.copy()
    clean_df["engagement_rate"] = pd.to_numeric(clean_df["engagement_rate"], errors="coerce")
    clean_df = clean_df.dropna(subset=["format", "engagement_rate"])

    if clean_df.empty:
        return {
            "status": "empty",
            "message": "No valid data available for comparison under the current selection.",
            "data": pd.DataFrame(),
            "best_format": None,
            "difference": None
        }

    # Aggregate by format
    results = (
        clean_df.groupby("format")
        .agg(
            posts=("engagement_rate", "count"),
            average_engagement=("engagement_rate", "mean"),
            highest_engagement=("engagement_rate", "max"),
            lowest_engagement=("engagement_rate", "min"),
        )
        .reset_index()
        .sort_values(by="average_engagement", ascending=False)
    )

    num_formats = len(results)

    if num_formats == 0:
        return {
            "status": "empty",
            "message": "No format records found.",
            "data": pd.DataFrame(),
            "best_format": None,
            "difference": None
        }

    best_format = results.iloc[0].to_dict()

    if num_formats == 1:
        fmt_name = best_format["format"]
        posts_count = int(best_format["posts"])
        avg_eng = float(best_format["average_engagement"])
        return {
            "status": "single_format",
            "message": (
                f"Only one format ('{fmt_name}', N={posts_count}) is available in this selection "
                f"(average engagement: {avg_eng:.2f}%). An A/B comparison requires at least two "
                f"distinct formats to measure a performance difference."
            ),
            "data": results,
            "best_format": best_format,
            "second_best": None,
            "difference": None
        }

    # Two or more formats available
    second_best = results.iloc[1].to_dict()
    difference = float(best_format["average_engagement"] - second_best["average_engagement"])

    return {
        "status": "multi_format",
        "message": (
            f"Highest observed average is {best_format['format']} ({best_format['average_engagement']:.2f}%, N={best_format['posts']}) "
            f"compared to {second_best['format']} ({second_best['average_engagement']:.2f}%, N={second_best['posts']}), "
            f"a descriptive difference of {difference:.2f} percentage points."
        ),
        "data": results,
        "best_format": best_format,
        "second_best": second_best,
        "difference": difference
    }


def run_ab_testing():
    res = compute_format_comparison()
    if res["status"] == "empty":
        print(res["message"])
        return

    results = res["data"]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\n===== A/B TESTING FRAMEWORK =====")
    print("\nContent Format Comparison:")
    print(results.round(2).to_string(index=False))

    print("\n===== RECOMMENDATION =====")
    best_format = res["best_format"]
    print(f"Best-performing format in this dataset: {best_format['format']}")
    print(f"Average engagement rate: {best_format['average_engagement']:.2f}%")

    if res["difference"] is not None:
        print(f"Difference from the second-best format: {res['difference']:.2f} percentage points")
    else:
        print(res["message"])

    print("\nResults saved to:", OUTPUT_FILE)
    print("\nNote: This is a descriptive comparison, not proof that the format caused better engagement.")


if __name__ == "__main__":
    run_ab_testing()
