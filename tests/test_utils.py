from korkeasaari.utils import excel_file_to_visitors_df, all_excels_to_visitors_df
import os


def test_january_february_2018_sums():
    folder_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "raw"
    )
    df = excel_file_to_visitors_df(folder_path + "/Kavijatilasto_2018.xlsx")

    # Parse month from ISO date string
    jan = df[df["month"] == "01"]
    feb = df[df["month"] == "02"]
    jan_sum = jan["visitors"].sum()
    feb_sum = feb["visitors"].sum()
    assert jan_sum == 6731, f"Expected 6731 for January, got {jan_sum}"
    assert feb_sum == 5400, f"Expected 5400 for February, got {feb_sum}"


def test_all_excels_to_visitors_df_format():
    folder_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "raw"
    )
    df = all_excels_to_visitors_df(folder_path)
    # Check columns
    expected_columns = ["date", "visitors", "day_of_week"]
    assert list(df.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {list(df.columns)}"
    )
    # Check types
    assert df["date"].apply(lambda x: isinstance(x, str)).all(), (
        "All 'date' values should be strings"
    )
    assert df["visitors"].apply(lambda x: isinstance(x, int)).all(), (
        "All 'visitors' values should be int"
    )
    # Check day_of_week: Finnish two-letter abbreviations
    valid_abbr = {"ma", "ti", "ke", "to", "pe", "la", "su"}
    assert (
        df["day_of_week"].apply(lambda x: isinstance(x, str) and x in valid_abbr).all()
    ), "All 'day_of_week' values should be Finnish two-letter abbreviations"
