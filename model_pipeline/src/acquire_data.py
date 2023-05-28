import logging
from pathlib import Path
import numpy as np
import pandas as pd
import src.aws_utils as aws


logger = logging.getLogger(__name__)

def load_data(data_file: Path, BUCKET_NAME, s3_key: str) -> np.ndarray:
    """
    Load data from a file into memory.

    Args:
        data_file (Path): Path to the data file.
        s3_key (str): S3 key for downloading the data file.

    Returns:
        np.ndarray: Loaded data.
    """
    aws.download_s3(BUCKET_NAME, s3_key, data_file)
    # Load files into memory
    dat = pd.read_csv(data_file)
    logger.info("Data loaded successfully.")
    return dat


def acquire_data(data_address: str, output_file: str) -> pd.DataFrame:
    """
    Acquire data from the specified address and save it to a CSV file.

    Args:
        data_address: Address of the input data file.
        output_file: Path to the output CSV file for saving the acquired data.

    Returns:
        DataFrame containing the acquired data.
    """

    logger.info("Acquiring data from: %s", data_address)
    try:
        dat = pd.read_csv(data_address)
    except FileNotFoundError:
        logger.error("Data file not found at: %s", data_address)
        raise

    logger.info("Saving acquired data to: %s", output_file)
    try:
        dat.to_csv(output_file, index=False)
    except:
        logger.error("Failed to save acquired data to: %s", output_file)
        raise
    logger.info("Data acquired successfully.")
    return dat
