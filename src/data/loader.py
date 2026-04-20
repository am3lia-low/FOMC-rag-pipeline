import pandas as pd
from src.config import FOMC_DATASET
from src.data.cleaner import clean_text


def load_fomc_data(start_year=2020, end_year=2025):
    from datasets import load_dataset

    print("Loading FOMC dataset from HuggingFace...")
    ds = load_dataset(FOMC_DATASET)
    df = ds['train'].to_pandas()

    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()

    print(f"Filtered to {start_year}-{end_year}: {len(df)} documents")
    print(f"  - Statements: {len(df[df['type'] == 'Statement'])}")
    print(f"  - Minutes: {len(df[df['type'] == 'Minute'])}")

    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)

    df['doc_id'] = df.apply(
        lambda row: f"{row['date'].strftime('%Y-%m-%d')}_{row['type'].lower()}",
        axis=1
    )

    df = df.reset_index(drop=True)

    print(f"\nData loaded successfully. {len(df)} documents ready.")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to "
          f"{df['date'].max().strftime('%Y-%m-%d')}")

    return df
