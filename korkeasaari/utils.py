"""
Various utility functions, for use outside of the `korkeasaari` module.

Do not add any functions here that gets imported into the `korkeasaari` module.
If you need to add logic that gets imported into the `korkeasaari` module,
add it to another file in this directory where it logically belongs. If none such file
exists, create a new file in this directory.

These are are utilities that can be used in runner scripts, tests, etc.
"""

import pandas as pd
import re
import os


def sheet_to_visitors_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a sheet DataFrame into a DataFrame with columns:
    - 'day_of_month': 1-31 (int)
    - 'day_of_week': name of the day (str)
    - 'visitors': number of visitors (int)
    Assumes dates are in the first column and visitor counts in the fourth column.
    Skips non-data rows and summary rows.
    """
    # Select relevant columns and drop rows with missing values
    df = df[[df.columns[0], df.columns[1], df.columns[3]]].dropna()
    # Rename columns
    df = df.rename(
        columns={
            df.columns[0]: "day_of_month",
            df.columns[1]: "day_of_week",
            df.columns[2]: "visitors",
        }
    )
    # Convert date to ISO format
    df["day_of_month"] = df["day_of_month"].astype(int)
    # Convert visitors to int
    df["visitors"] = df["visitors"].astype(int)
    return df


def excel_file_to_visitors_df(file_path: str) -> pd.DataFrame:
    """
    Reads all month sheets from an Excel file and combines them into a single DataFrame
    with columns: 'date', 'visitors', 'month'.
    """
    xl = pd.ExcelFile(file_path)
    month_pattern = re.compile(
        r"^(\d{1,2}|january|february|march|april|may|june|july|august|september|october|november|december|tammikuu|helmikuu|maaliskuu|huhtikuu|toukokuu|kesäkuu|heinäkuu|elokuu|syyskuu|lokakuu|marraskuu|joulukuu)",
        re.IGNORECASE,
    )
    month_lookup = {
        "tammikuu": "01",
        "helmikuu": "02",
        "maaliskuu": "03",
        "huhtikuu": "04",
        "toukokuu": "05",
        "kesäkuu": "06",
        "heinäkuu": "07",
        "elokuu": "08",
        "syyskuu": "09",
        "lokakuu": "10",
        "marraskuu": "11",
        "joulukuu": "12",
    }
    all_months = []
    for sheet in xl.sheet_names:
        sheet_name = sheet.strip()
        if not month_pattern.match(sheet_name):
            continue
        key = sheet_name.lower().lstrip("0")
        month_name = month_lookup.get(key, sheet_name)
        df = xl.parse(sheet)
        month_df = sheet_to_visitors_df(df)
        month_df["month"] = month_name
        all_months.append(month_df)
    if all_months:
        return pd.concat(all_months, ignore_index=True)
    else:
        return pd.DataFrame(
            columns=["day_of_month", "day_of_week", "visitors", "month"]
        )


def all_excels_to_visitors_df(folder_path: str) -> pd.DataFrame:
    """
    Reads all Excel files in a folder and combines all their month sheets into a single DataFrame
    with columns: 'date', 'visitors', 'day_of_week'.
    The year is parsed from the file name (expects ...YYYY.xlsx).
    """
    all_data = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".xlsx"):
            fpath = os.path.join(folder_path, fname)
            # Parse year from filename
            year_match = re.search(r"(\d{4})\.xlsx$", fname)
            year = year_match.group(1) if year_match else None
            df = excel_file_to_visitors_df(fpath)
            if not df.empty:
                df["year"] = year
                df["date"] = df.apply(
                    lambda row: f"{year}-{row['month']}-{int(row['day_of_month']):02d}",
                    axis=1,
                )
                df["file"] = fname
                all_data.append(df[["date", "visitors", "day_of_week"]])
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame(columns=["date", "visitors", "day_of_week"])
