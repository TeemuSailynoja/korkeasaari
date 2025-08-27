import pandas as pd
from pathlib import Path


def filter_and_aggregate_weather(input_path, output_path):
    # Read the CSV file
    df = pd.read_csv(input_path)
    # Combine date and time columns into a datetime
    df["datetime"] = pd.to_datetime(
        df["Vuosi"].astype(str)
        + "-"
        + df["Kuukausi"].astype(str).str.zfill(2)
        + "-"
        + df["Päivä"].astype(str).str.zfill(2)
        + " "
        + df["Aika [Paikallinen aika]"],
        format="%Y-%m-%d %H:%M",
    )
    df["hour"] = df["datetime"].dt.hour
    # Filter out rows between 19:00 and 08:00 (keep 8:00-18:59)
    df = df[(df["hour"] >= 8) & (df["hour"] < 19)]
    # Aggregate to daily (mean for numeric columns)
    df["date"] = df["datetime"].dt.date
    numeric_cols = [
        "Lämpötilan keskiarvo [°C]",
        "Ylin lämpötila [°C]",
        "Alin lämpötila [°C]",
        "Tunnin sademäärä [mm]",
    ]
    daily = df.groupby("date")[numeric_cols].mean().reset_index()
    # Save to output
    daily.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_csv = Path("data/raw/helsinki_kaisaniemi.csv")
    output_csv = Path("data/clean/helsinki_kaisaniemi_daily_filtered.csv")
    filter_and_aggregate_weather(input_csv, output_csv)
