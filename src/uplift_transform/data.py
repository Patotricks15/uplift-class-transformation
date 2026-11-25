from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
FEATURES = [f"f{i}" for i in range(12)]
TREATMENT = "treatment"
OUTCOME = "conversion"


def load_criteo(path: str | Path, sample_n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample_n is not None:
        df = df.sample(n=sample_n, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, test_size: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split stratified by treatment, because the dataset has no dates."""
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[TREATMENT], random_state=RANDOM_STATE
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
