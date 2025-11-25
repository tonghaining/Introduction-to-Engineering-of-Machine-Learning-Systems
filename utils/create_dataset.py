#%%
from pathlib import Path

import pandas as pd
from deepchecks.tabular.datasets.regression import wine_quality
from sklearn.model_selection import train_test_split

# Configuration
output_dir = Path("/home/ubuntu/Introduction-to-Engineering-of-Machine-Learning-Systems/data")
output_dir.mkdir(parents=True, exist_ok=True)

drop_column = "alcohol"   # easy to change
drop_fraction = 0.25      # largest 25% in the chosen column

def drop_top_fraction(df: pd.DataFrame, column: str, frac: float) -> pd.DataFrame:
    """Drop the rows with the largest `frac` values in `column` from df."""
    quantile = df[column].quantile(1 - frac)
    # Use < to drop values at or above the quantile (approximately top `frac`)
    return df[df[column] < quantile].copy()

def main():
    # Load full dataset from deepchecks wrapper
    dataset = wine_quality.load_data(as_train_test=False)
    full_df = dataset.data  # includes target 'quality'

    # Split into train/test once, with full data (features + target)
    train_df, test_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=66,
        shuffle=True,
    )

    # Save the raw splits (2_* files; contain full wine data)
    two_train_path = output_dir / "2_data_train.csv"
    two_test_path = output_dir / "2_data_test.csv"

    train_df.to_csv(two_train_path, index=False)
    test_df.to_csv(two_test_path, index=False)

    # Derive 1_* files from the 2_* splits by dropping top 25% in drop_column
    train_df_filtered = drop_top_fraction(train_df, drop_column, drop_fraction)
    test_df_filtered = drop_top_fraction(test_df, drop_column, drop_fraction)

    one_train_path = output_dir / "1_data_train.csv"
    one_test_path = output_dir / "1_data_test.csv"

    train_df_filtered.to_csv(one_train_path, index=False)
    test_df_filtered.to_csv(one_test_path, index=False)

if __name__ == "__main__":
    main()
