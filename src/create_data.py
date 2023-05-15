import logging
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)

def create_dataset(data: pd.DataFrame, drop: List[str]) -> pd.DataFrame:
    """
    Create a processed dataset by dropping specified columns and performing feature engineering.

    Args:
        data: Input DataFrame containing the data.
        drop: List of column names to be dropped from the dataset.

    Returns:
        Processed DataFrame after dropping columns,
        performing feature engineering,
        and removing NaN values.
    """
    try:
        logger.info("Dropping columns: %s", drop)
        df = data.drop(drop, axis=1)

        numeric_features = list(df.select_dtypes("float64").columns)
        int_features = list(df.select_dtypes("int64").columns)
        categorical_features = list(df.select_dtypes(include=["category", "object"]).columns)

        logger.debug("Numeric features:\n%s", numeric_features)
        logger.debug("Categorical features:\n%s", categorical_features)
        try:
            df_numerical_train = df[numeric_features + int_features[1:5]]
            df_categorical_train = df[categorical_features + int_features[0:1] + int_features[5:]]
            df_categorical_train = pd.get_dummies(df_categorical_train)
        except IndexError as e:
            logger.error("Error occurred while creating the dataset: %s", str(e))
            raise
        df_final_train = pd.concat([df_numerical_train, df_categorical_train], axis=1)
        df_final_train = df_final_train.dropna()
        logger.info("Final dataset shape: %s", df_final_train.shape)
        return df_final_train
    except Exception as e:
        logger.error("Error occurred while creating the dataset: %s", str(e))
        raise


def save_dataset(data: pd.DataFrame, output_file: str) -> None:
    """
    Save the dataset to a CSV file.

    Args:
        data: DataFrame containing the dataset to be saved.
        output_file: Path to the output CSV file.

    Returns:
        None
    """
    try:
        data.to_csv(output_file, index=False)
    except FileNotFoundError as e:
        logger.error("Error occurred while saving the dataset: %s", str(e))
        raise
    logger.info("Dataset saved to: %s", output_file)
