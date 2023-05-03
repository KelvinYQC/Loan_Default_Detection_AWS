import os
from pathlib import Path
import pandas as pd
from typing import Optional

def subsample_csv(input_file: Path, output_file: Path, fraction: float = 0.5, random_seed: Optional[int] = 42) -> None:
    """
    Subsamples a CSV file and saves the subsample to a new file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the subsampled CSV file.
        fraction (float, optional): Fraction of data to subsample (default is 0.5, i.e., 50%).
        random_seed (int, optional): Random seed for reproducibility (default is None).

    Returns:
        None
    """
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(input_file)
    
    # Generate the subsample and save it to a new CSV file
    subsampled_data = data.sample(frac=fraction, random_state=random_seed)
    subsampled_data.to_csv(output_file, index=False)


# Example usage
input_file = Path(os.path.join('..', 'data', 'application_train.csv'))
output_file = Path(os.path.join('..', 'data', 'subsample_train.csv'))
subsample_csv(input_file, output_file)
