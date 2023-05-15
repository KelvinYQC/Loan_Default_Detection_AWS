from pathlib import Path
from typing import List, Dict
import logging
import sklearn
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def evaluate_performance(score: pd.DataFrame,
                         test: pd.DataFrame,
                         target: str,
                         metrics: List[str]) -> Dict[str, str]:
    """
    Evaluate the performance of the model on the test set
    Args:
        score: a dictionary containing the predictions on the test set
        test: the test set (ground truth)
        label: the name of the target variable
        metrics: a list of metrics to compute
    Returns:
        results: a list of tuples containing the name of the metric and its value

    """
    # Extract the ground truth
    try:
        y_test = test[target]
    except KeyError as e:
        logger.error('Column %s does not exist from test data', e)
        raise
    # Extract the predictions
    ypred_proba_test = score['ypred_proba_test']
    ypred_bin_test = score['ypred_bin_test']
    # Compute the metricss
    results = {}
    for metric in metrics:
        if metric == 'roc_auc_score':
            result = sklearn.metrics.roc_auc_score(y_test, ypred_proba_test)
            results['roc_auc_score'] = str(result)
            logger.debug('AUC on test: %.3f', result)
        elif metric == 'confusion_matrix':
            result = sklearn.metrics.confusion_matrix(y_test, ypred_bin_test)
            results['confusion_matrix'] = str(result.tolist())
            logger.debug('confusion_matrix on test generated')
        elif metric == 'accuracy_score':
            result = sklearn.metrics.accuracy_score(y_test, ypred_bin_test)
            results['accuracy_score'] = str(result)
            logger.debug('Accuracy on test: %.3f', result)
        elif metric == 'classification_report':
            result = sklearn.metrics.classification_report(y_test, ypred_bin_test, output_dict=True)
            results['classification_report'] = str(result)
            logger.debug('Classification report on test generated')
        elif metric == 'precision_score':
            result = sklearn.metrics.precision_score(y_test, ypred_bin_test)
            results['precision_score'] = str(result)
            logger.debug('Precision score on test: %.3f', result)
        else:
            logger.warning('Metric %s not implemented', metric)
    logger.info('Evaluation completed')
    logger.info('Results: %s', results)
    return results


def save_metrics(result, artifacts: Path) -> None:
    '''
    Save the results of the evaluation
    Args:
        result: a list of tuples containing the name of the metric and its value
        artifacts: path to save the results
    Returns:
        None (File saved to the specified path)
    '''
    try:
        with open(artifacts, 'w', encoding='utf-8') as file:
            yaml.dump(result, file)
            logger.info('Results saved to %s', artifacts)
    except FileNotFoundError:
        logger.error('Cannot save the results to %s', artifacts)
        raise
