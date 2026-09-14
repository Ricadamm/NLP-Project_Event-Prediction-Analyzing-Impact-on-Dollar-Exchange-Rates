import pandas as pd
import os
import sys

raw_path = os.path.join("data", "raw", "Informasi Kurs Jisdor.xlsx")
output_dir = os.path.join("data", "cleaned")
output_path = os.path.join(output_dir, "usd_idr_jisdor_cleaned.csv")


def _drop_invalid(df: pd.DataFrame, subset: list, label: str) -> pd.DataFrame:
    """Drop rows with nulls in subset"""
    before = len(df)
    df = df.dropna(subset=subset)
    dropped = before - len(df)
    if dropped:
        print(f"  WARNING: {dropped} rows with {label} (dropped)")
    return df


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw Excel file and locate where the data table starts"""
    df = pd.read_excel(path, header=None)
    print(f"[1/4] Loaded raw file: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def extract_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date/rate columns from the raw BI export."""
    is_header = df.apply(lambda r: r.astype(str).str.strip().str.lower().eq("tanggal").any(), axis=1)
    header_idx = is_header.idxmax() if is_header.any() else None
    if header_idx is None:
        raise ValueError("Could not find header row with 'Tanggal' column")

    data = df.iloc[header_idx + 1:, [1, 2]].copy()
    data.columns = ["date_raw", "rate_raw"]
    data = _drop_invalid(data, ["date_raw", "rate_raw"], "missing date/rate")
    print(f"[2/4] Extracted {len(data)} data rows (skipped {header_idx + 1} header rows)")
    return data.reset_index(drop=True)


def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw text into real dates and numbers, and remove any row that doesn't convert properly"""
    df["date"] = pd.to_datetime(df["date_raw"], format="mixed").dt.normalize()
    df["usd_idr"] = pd.to_numeric(df["rate_raw"], errors="coerce")
    df = _drop_invalid(df, ["date", "usd_idr"], "unparseable date/rate")

    print(f"[3/4] Cleaned {len(df)} rows: {df['date'].min().date()} to {df['date'].max().date()}, "
          f"rate {df['usd_idr'].min():.2f}-{df['usd_idr'].max():.2f}")
    return df


def finalize_and_save(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
   """Put the rows in date order, remove any repeated dates, and save the cleaned results to a file"""
    df = df[["date", "usd_idr"]].sort_values("date")
    before = len(df)
    df = df.drop_duplicates(subset=["date"], keep="first")
    if len(df) < before:
        print(f"  WARNING: Removed {before - len(df)} duplicate dates")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[4/4] Saved {len(df)} rows to {output_path} "
          f"({df['date'].iloc[0]} to {df['date'].iloc[-1]})")
    return df.reset_index(drop=True)


def main():
    if not os.path.exists(raw_path):
        sys.exit(f"ERROR: Raw file not found at '{raw_path}'. "
                  f"Place 'Informasi Kurs Jisdor.xlsx' in data/raw/ before running.")

    df = load_raw(raw_path)
    df = extract_data(df)
    df = clean_types(df)
    df = finalize_and_save(df, output_path)

    print(f"\nFirst 5 rows:\n{df.head().to_string(index=False)}")
    print(f"\nLast 5 rows:\n{df.tail().to_string(index=False)}")


if __name__ == "__main__":
    main()
