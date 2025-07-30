import polars as pl

# Load cleaned visitors and weather data
df_visitors = pl.read_csv("data/clean/visitors.csv")
df_weather = pl.read_csv("data/clean/weather.csv")

# Remove time and timezone from weather date for merging
# (Assumes weather date is in format 'YYYY-MM-DD HH:MM:SS+00:00')
df_weather = df_weather.with_columns([pl.col("date").str.slice(0, 10).alias("date")])

# Check for missing or duplicate dates in visitors
duplicates_visitors = df_visitors["date"].is_duplicated().sum()
missing_visitors = df_visitors["date"].is_null().sum()

# Check for missing or duplicate dates in weather
duplicates_weather = df_weather["date"].is_duplicated().sum()
missing_weather = df_weather["date"].is_null().sum()

print(
    f"Visitors: {len(df_visitors)} rows, {duplicates_visitors} duplicate dates, {missing_visitors} missing dates"
)
print(
    f"Weather: {len(df_weather)} rows, {duplicates_weather} duplicate dates, {missing_weather} missing dates"
)

# Find overlapping date range for modeling
min_date = max(df_visitors["date"].min(), df_weather["date"].min())
max_date = min(df_visitors["date"].max(), df_weather["date"].max())
print(f"Overlapping date range: {min_date} to {max_date}")
