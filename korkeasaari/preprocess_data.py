import polars as pl
import os

# Load cleaned visitors and weather data
df_visitors = pl.read_csv("data/clean/visitors.csv")
df_weather = pl.read_csv("data/clean/weather.csv")

# Standardize date format in weather data for merging
df_weather = df_weather.with_columns([pl.col("date").str.slice(0, 10).alias("date")])

# Find overlapping date range
overlap_start = max(df_visitors["date"].min(), df_weather["date"].min())
overlap_end = min(df_visitors["date"].max(), df_weather["date"].max())

# Filter both datasets to overlapping range
df_visitors = df_visitors.filter(
    (pl.col("date") >= overlap_start) & (pl.col("date") <= overlap_end)
)
df_weather = df_weather.filter(
    (pl.col("date") >= overlap_start) & (pl.col("date") <= overlap_end)
)

# Merge on date
df_merged = df_visitors.join(df_weather, on="date", how="inner")

# Output summary
print(f"Merged dataset: {df_merged.shape[0]} rows, columns: {df_merged.columns}")
print(df_merged.head(5))


# Optionally merge holidays/events data if available

holidays_path = "data/clean/holidays_events.csv"
if os.path.exists(holidays_path):
    df_holidays = pl.read_csv(holidays_path)
    df_merged = df_merged.join(df_holidays, on="date", how="left")
    print(f"Merged holidays/events data: {df_holidays.shape[0]} rows")
else:
    print(
        "No holidays/events data found. To add, create 'data/clean/holidays_events.csv' with columns: date,event_name"
    )

# Save merged data for modeling
df_merged.write_csv("data/clean/visitors_weather_merged.csv")
print("Saved merged data to data/clean/visitors_weather_merged.csv")
