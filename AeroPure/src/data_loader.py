"""
Data loading module for the AeroPure air-quality forecasting project.

Source: Beijing Multi-Site Air-Quality Data, UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data

The raw download contains one CSV per monitoring site, e.g.:
    PRSA_Data_Aotizhongxin_20130301-20170228.csv
    PRSA_Data_Changping_20130301-20170228.csv
    ... (12 sites total)

Drop all 12 site CSVs into the project's Data/ folder (unzipped, as-is) and
load_data() will read and concatenate them into a single DataFrame with a
proper datetime column.
"""

import os
import glob
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")


def load_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load and concatenate every PRSA site CSV found in the Data folder."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "PRSA_Data_*.csv")))

    if not csv_files:
        # Fall back to any CSV in the folder (in case the files were renamed)
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(
            f"No site CSV files found in {data_dir}. "
            "Download the 12 PRSA_Data_*.csv files from the UCI Beijing "
            "Multi-Site Air Quality dataset and place them in this folder."
        )

    frames = [pd.read_csv(f).assign(Ticker=os.path.splitext(os.path.basename(f))[0]) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    # Build a single datetime column from either the OHLCV Date column or
    # the original Beijing air-quality year/month/day/hour columns.
    if "Date" in df.columns:
        df["datetime"] = pd.to_datetime(df["Date"])
    if {"year", "month", "day", "hour"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded dataset with shape: {df.shape}")
    print(df.head())
    print(f"Sites: {df['station'].unique().tolist() if 'station' in df.columns else 'n/a'}")
