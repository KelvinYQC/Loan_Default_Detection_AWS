from pathlib import Path
import logging
from typing import List
import pandas as pd
logger = logging.getLogger(__name__)

def score_model(test: pd.DataFrame, tmo, target: str, metrics: List[str]) -> pd.DataFrame:
    """
    Score the model on the test set.
    Args:
        test (pd.DataFrame): DataFrame containing test set.
        tmo (sklearn model): Trained model object.
        target (str): Name of the target variable.
        metrics (list): List of metrics to use.
    Returns:
        res (pd.DataFrame): DataFrame containing predicted values.
    """
    # Initialize DataFrame with proper columns
    res = pd.DataFrame(columns=['ypred_proba_test', 'ypred_bin_test'])
    try:
        x_test = test.drop([target], axis=1)
    except KeyError:
        logger.error('Label %s not in test set.', target)
        raise
    else:
        logger.debug('Label %s dropped from test set.', target)
    # Loop through metrics
    for metric in metrics:
        if metric == 'prob':
            # Add predicted probabilities to DataFrame
            res['ypred_proba_test'] = tmo.predict_proba(x_test)[:, 1]
            logger.debug('Predicted probabilities added to DataFrame.')
        elif metric == 'bin':
            # Add binary predictions to DataFrame
            res['ypred_bin_test'] = tmo.predict(x_test)
            logger.debug('Binary predictions added to DataFrame.')
        else:
            logger.warning('Metric %s not recognized.', metric)
    # Return the DataFrame with predicted values
    return res


def save_scores(scores: pd.DataFrame, artifacts: Path) -> None:
    """
    Save the scores.
    Args:
        scores (pd.DataFrame): DataFrame containing scores.
        artifacts (str): Path to save artifacts to.
    Returns:
        None (scores are saved to artifacts
    """
    try:
        scores.to_csv(artifacts, index=False)
    except FileNotFoundError:
        logger.error('Directory %s not found.', artifacts.parent)
        raise
    logger.info('Scores saved to %s', artifacts)
