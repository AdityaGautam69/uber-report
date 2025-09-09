# data_pipeline.py
import os
import sys
import pandas as pd
import numpy as np
from shutil import which

# check if graphviz is installed
if which("dot") is None:
    print("⚠️  graphviz binary not found. Install with: sudo apt-get install graphviz")

try:
    import kagglehub
    from ydata_profiling import ProfileReport
    from sqlalchemy import create_engine
    from eralchemy import render_er
except Exception as e:
    print("❌ Missing package:", e)
    print("➡️  Run: pip install -r requirements.txt")
    sys.exit(1)

def download_dataset():
    print("→ Downloading dataset with kagglehub...")
    path = kagglehub.dataset_download("yashdevladdha/uber-ride-analytics-dashboard")
    print("✅ Dataset downloaded at:", path)
    return path

def load_first_csv(path):
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No CSV files found in dataset folder.")
    file_path = os.path.join(path, files[0])
    print("📌 Using file:", file_path)
    df = pd.read_csv(file_path)
    return df, file_path

def overview(df):
    print("\n--- Shape:", df.shape)
    print("\n--- Data Types:\n", df.dtypes)
    print("\n--- Missing Values:\n", df.isnull().sum())
    print("\n--- Sample Rows:\n", df.head(3))

def generate_eda(df, out_filename="uber_eda_report.html"):
    print("\n→ Generating EDA report (this may take some time)...")
    profile = ProfileReport(df, title="Uber Ride Analytics EDA Report", explorative=True)
    profile.to_file(out_filename)
    print("✅ EDA Report saved:", out_filename)

def clean_data(df):
    print("\n→ Cleaning data (numeric → mean, categorical → mode)...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isnull().any():
            df[c].fillna(df[c].mean(), inplace=True)

    cat_cols = [c for c in df.columns if c not in num_cols]
    for c in cat_cols:
        if df[c].isnull().any():
            mode = df[c].mode()
            fill = mode[0] if not mode.empty else "Unknown"
            df[c].fillna(fill, inplace=True)

    print("✅ Missing values handled.")
    return df

def save_to_sqlite(df, db_name="uber.db", table_name="UberRides"):
    print(f"\n→ Saving cleaned data to SQLite database '{db_name}'...")
    engine = create_engine(f"sqlite:///{db_name}", echo=False)
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    print("✅ Saved to", db_name)
    return db_name

def make_er_diagram(sql_uri="sqlite:///uber.db", out_image="uber_er.png"):
    print("\n→ Generating ER diagram from database...")
    render_er(sql_uri, out_image)
    print("✅ ER Diagram saved:", out_image)

def main():
    path = download_dataset()
    df, csv_path = load_first_csv(path)
    overview(df)
    generate_eda(df, out_filename="uber_eda_report.html")
    df = clean_data(df)
    db = save_to_sqlite(df, db_name="uber.db")
    make_er_diagram(f"sqlite:///{db}", out_image="uber_er.png")
    print("\n🎉 Done! Files created → uber_eda_report.html, uber.db, uber_er.png")

if __name__ == "__main__":
    main()
