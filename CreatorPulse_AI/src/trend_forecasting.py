import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "content_engagement_metrics.csv"
)

REPORT_DIR = BASE_DIR / "reports"
OUTPUT_FILE = REPORT_DIR / "trend_forecast.csv"
CHART_FILE = REPORT_DIR / "engagement_trend.png"


def run_trend_forecasting():
    if not DATA_FILE.exists():
        print("Dataset not found. Run the earlier modules first.")
        return None

    df = pd.read_csv(DATA_FILE)

    required = {"post_date", "engagement_rate"}
    if not required.issubset(df.columns):
        print("Required columns: post_date, engagement_rate")
        return None

    df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
    df["engagement_rate"] = pd.to_numeric(df["engagement_rate"], errors="coerce")
    df = df.dropna(subset=["post_date", "engagement_rate"]).sort_values("post_date")

    if df.empty:
        print("No valid data available.")
        return None

    # Calculate date-level average engagement
    daily = (
        df.groupby("post_date")["engagement_rate"]
        .mean()
        .sort_index()
        .reset_index()
    )

    # 3-period rolling average
    daily["rolling_average"] = (
        daily["engagement_rate"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # Linear trend projection on sequence index
    daily["day_number"] = range(len(daily))

    if len(daily) >= 2:
        slope, intercept = np.polyfit(
            daily["day_number"],
            daily["engagement_rate"],
            1
        )
        daily["trend_estimate"] = slope * daily["day_number"] + intercept
        next_day = len(daily)
        next_estimate = float(slope * next_day + intercept)
    else:
        slope = 0.0
        daily["trend_estimate"] = daily["engagement_rate"]
        next_estimate = float(daily["engagement_rate"].iloc[-1])

    # Simple Baseline Comparison (MAE / RMSE)
    actual = daily["engagement_rate"].values
    trend_pred = daily["trend_estimate"].values
    mean_baseline = np.full_like(actual, actual.mean())
    naive_baseline = np.roll(actual, 1)
    naive_baseline[0] = actual[0]

    trend_mae = float(np.mean(np.abs(actual - trend_pred)))
    naive_mae = float(np.mean(np.abs(actual - naive_baseline)))
    mean_mae = float(np.mean(np.abs(actual - mean_baseline)))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_FILE, index=False)

    # High-quality matplotlib chart
    plt.figure(figsize=(10, 5), facecolor="#0e1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")
    
    # Plot historical daily points
    plt.plot(
        daily["post_date"],
        daily["engagement_rate"],
        marker="o",
        color="#58a6ff",
        alpha=0.6,
        linewidth=1.5,
        label="Observed Engagement Rate (%)"
    )
    # Plot rolling average
    plt.plot(
        daily["post_date"],
        daily["rolling_average"],
        color="#3fb950",
        linewidth=2.5,
        label="3-Period Rolling Average (%)"
    )
    # Plot linear trend
    plt.plot(
        daily["post_date"],
        daily["trend_estimate"],
        color="#d29922",
        linestyle="--",
        linewidth=1.8,
        label=f"Linear Trend (slope: {slope:+.3f}%/step)"
    )

    plt.title("Social Engagement Trend Over Time (%)", color="#f0f6fc", fontsize=14, pad=12)
    plt.xlabel("Post Date (YYYY-MM-DD)", color="#c9d1d9", fontsize=11)
    plt.ylabel("Engagement Rate (%)", color="#c9d1d9", fontsize=11)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=":", alpha=0.3, color="#8b949e")
    plt.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc")
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    plt.close()

    print("\n===== TREND FORECASTING & BASELINE VALIDATION =====")
    print(f"Posts/Dates Analyzed: {len(daily)}")
    print(f"Date Span: {daily['post_date'].min().strftime('%Y-%m-%d')} to {daily['post_date'].max().strftime('%Y-%m-%d')}")
    print(f"Latest Observed Engagement: {daily['engagement_rate'].iloc[-1]:.2f}%")
    print(f"Latest 3-Period Rolling Average: {daily['rolling_average'].iloc[-1]:.2f}%")
    print(f"Next-Period Linear Projection: {next_estimate:.2f}% (Slope: {slope:+.4f}% per post)")
    print("\nBaseline Comparison (Mean Absolute Error):")
    print(f"  • Linear Trend Model MAE: {trend_mae:.2f}%")
    print(f"  • Last-Value Naive Baseline MAE: {naive_mae:.2f}%")
    print(f"  • Historical Mean Baseline MAE: {mean_mae:.2f}%")
    print("\nForecast data saved to:", OUTPUT_FILE)
    print("Trend chart saved to:", CHART_FILE)
    print("Note: Projections are exploratory baseline estimates from synthetic data, not guaranteed forecasts.")

    return {
        "daily": daily,
        "next_estimate": next_estimate,
        "slope": slope,
        "trend_mae": trend_mae,
        "naive_mae": naive_mae,
        "mean_mae": mean_mae
    }


if __name__ == "__main__":
    run_trend_forecasting()
