import pandas as pd
from pathlib import Path


def parse_weather_hourly(input_path, output_path):
    """
    Parse weather data to extract hourly rainfall during park opening hours (8-19)
    into separate columns, plus daily temperature averages.
    """
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
    df["date"] = df["datetime"].dt.date

    # Filter to park opening hours (8:00-18:59)
    park_hours_df = df[(df["hour"] >= 8) & (df["hour"] < 19)]

    # Create hourly rainfall columns
    rainfall_pivot = park_hours_df.pivot_table(
        index="date",
        columns="hour",
        values="Tunnin sademäärä [mm]",
        aggfunc="first",  # Should be only one value per date-hour combination
    )

    # Rename columns to be more descriptive
    rainfall_columns = {hour: f"rain_hour_{hour:02d}" for hour in range(8, 19)}
    rainfall_pivot = rainfall_pivot.rename(columns=rainfall_columns)

    # Fill missing values with 0 (no rain)
    rainfall_pivot = rainfall_pivot.fillna(0)

    # Calculate daily temperature averages during park hours
    temp_daily = (
        park_hours_df.groupby("date")
        .agg(
            {
                "Lämpötilan keskiarvo [°C]": "mean",
                "Ylin lämpötila [°C]": "mean",
                "Alin lämpötila [°C]": "mean",
            }
        )
        .rename(
            columns={
                "Lämpötilan keskiarvo [°C]": "temp_avg",
                "Ylin lämpötila [°C]": "temp_max",
                "Alin lämpötila [°C]": "temp_min",
            }
        )
    )

    # Combine rainfall and temperature data
    result = rainfall_pivot.join(temp_daily)

    # Reset index to make date a column
    result = result.reset_index()

    # Ensure all hourly rain columns exist (in case some hours are missing)
    for hour in range(8, 19):
        col_name = f"rain_hour_{hour:02d}"
        if col_name not in result.columns:
            result[col_name] = 0.0

    # Reorder columns: date, temperature columns, then hourly rain columns
    temp_cols = ["temp_avg", "temp_max", "temp_min"]
    rain_cols = [f"rain_hour_{hour:02d}" for hour in range(8, 19)]
    result = result[["date"] + temp_cols + rain_cols]

    # Save to output
    result.to_csv(output_path, index=False)
    print(f"Processed {len(result)} days of hourly weather data")
    print(f"Columns: {list(result.columns)}")
    return result


if __name__ == "__main__":
    input_csv = Path("data/raw/helsinki_kaisaniemi.csv")
    output_csv = Path("data/clean/helsinki_kaisaniemi_hourly.csv")
    parse_weather_hourly(input_csv, output_csv)
