import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "content_engagement_metrics.csv"

# Minimum post count for reliable statistical recommendation
MIN_SAMPLE_THRESHOLD = 5


def generate_recommendations(df=None):
    """
    Generate evidence-linked recommendations with sample counts,
    explicit caveats for low-sample segments, and proposed experiments.
    """
    if df is None:
        if not DATA_FILE.exists():
            return None
        df = pd.read_csv(DATA_FILE)

    required = {"topic", "format", "posting_hour", "engagement_rate"}
    if not required.issubset(df.columns):
        return None

    df["engagement_rate"] = pd.to_numeric(df["engagement_rate"], errors="coerce")
    df["posting_hour"] = pd.to_numeric(df["posting_hour"], errors="coerce")
    df = df.dropna(subset=["topic", "format", "posting_hour", "engagement_rate"]).copy()

    if df.empty:
        return None

    total_posts = len(df)
    overall_mean = df["engagement_rate"].mean()

    # Aggregate by dimension
    format_stats = df.groupby("format")["engagement_rate"].agg(["count", "mean", "std", "min", "max"]).sort_values("mean", ascending=False)
    topic_stats = df.groupby("topic")["engagement_rate"].agg(["count", "mean", "std", "min", "max"]).sort_values("mean", ascending=False)
    hour_stats = df.groupby("posting_hour")["engagement_rate"].agg(["count", "mean", "std", "min", "max"]).sort_values("mean", ascending=False)

    best_format = format_stats.index[0]
    best_format_n = int(format_stats.loc[best_format, "count"])
    best_format_mean = float(format_stats.loc[best_format, "mean"])

    best_topic = topic_stats.index[0]
    best_topic_n = int(topic_stats.loc[best_topic, "count"])
    best_topic_mean = float(topic_stats.loc[best_topic, "mean"])

    best_hour = int(hour_stats.index[0])
    best_hour_n = int(hour_stats.loc[best_hour, "count"])
    best_hour_mean = float(hour_stats.loc[best_hour, "mean"])

    recs = [
        {
            "dimension": "Content Format",
            "recommendation": best_format,
            "sample_count": best_format_n,
            "total_dataset_posts": total_posts,
            "average_engagement": round(best_format_mean, 2),
            "baseline_engagement": round(overall_mean, 2),
            "status": "Candidate to Test" if best_format_n >= MIN_SAMPLE_THRESHOLD else "Low Sample (Exploratory)",
            "caveat": f"Supported by {best_format_n} posts ({best_format_n/total_posts:.1%} of dataset). Descriptive correlation only; not guaranteed causal uplift.",
            "suggested_experiment": f"Publish 3-5 planned {best_format} posts alongside alternating formats to test replication under similar topics."
        },
        {
            "dimension": "Content Topic",
            "recommendation": best_topic,
            "sample_count": best_topic_n,
            "total_dataset_posts": total_posts,
            "average_engagement": round(best_topic_mean, 2),
            "baseline_engagement": round(overall_mean, 2),
            "status": "Candidate to Test" if best_topic_n >= MIN_SAMPLE_THRESHOLD else "Low Sample (Exploratory)",
            "caveat": f"Caution: Supported by only {best_topic_n} posts ({best_topic_n/total_posts:.1%} of dataset). Sample size is too small for statistical certainty.",
            "suggested_experiment": f"Design a controlled batch of 3 {best_topic} posts to verify if above-average engagement replicates with varied hooks."
        },
        {
            "dimension": "Posting Window",
            "recommendation": f"{best_hour:02d}:00 (Local Timezone Required)",
            "sample_count": best_hour_n,
            "total_dataset_posts": total_posts,
            "average_engagement": round(best_hour_mean, 2),
            "baseline_engagement": round(overall_mean, 2),
            "status": "Candidate to Test" if best_hour_n >= MIN_SAMPLE_THRESHOLD else "Low Sample (Exploratory)",
            "caveat": f"Based on {best_hour_n} posts. Assumes audience timezone matches creator clock; posting hour alone does not overcome weak content quality.",
            "suggested_experiment": f"Schedule posts in alternating time slots (e.g. {best_hour:02d}:00 vs 13:00) with identical format to measure timing impact."
        }
    ]

    return {
        "overall_mean": overall_mean,
        "total_posts": total_posts,
        "recommendations": recs,
        "format_stats": format_stats,
        "topic_stats": topic_stats,
        "hour_stats": hour_stats
    }


def main():
    data = generate_recommendations()
    if data is None:
        print("Dataset not found or missing required columns.")
        print("Run: python src/engagement_metrics.py")
        return

    print("\n===== EVIDENCE-LINKED ENGAGEMENT RECOMMENDATIONS =====")
    print(f"Total Posts Analyzed: {data['total_posts']} | Overall Average Engagement: {data['overall_mean']:.2f}%\n")

    for rec in data["recommendations"]:
        print(f"[{rec['dimension']}] Recommended: {rec['recommendation']}")
        print(f"  • Sample Count: {rec['sample_count']} / {rec['total_dataset_posts']} posts")
        print(f"  • Observed Avg Engagement: {rec['average_engagement']:.2f}% (Baseline: {rec['baseline_engagement']:.2f}%)")
        print(f"  • Confidence Status: {rec['status']}")
        print(f"  • Caveat: {rec['caveat']}")
        print(f"  • Suggested Experiment: {rec['suggested_experiment']}\n")

    print("Note: Recommendations are sample-derived hypotheses, not universal platform rules or guaranteed reach.")


if __name__ == "__main__":
    main()
