import logging
from pathlib import Path

import pickle
from typing import Any, Dict, Tuple
import pandas as pd


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


logger = logging.getLogger(__name__)


def split_data(data: pd.DataFrame, target: str, test_size: float, random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Any, Any]:
    """
    Split the data into training and test sets.

    Args:
        data: Input DataFrame containing the data.
        target: Name of the target column.
        test_size: Proportion of the dataset to include in the test split.
        random_seed: Seed value for random number generator.

    Returns:
        Tuple containing the following elements:
        - train: DataFrame containing the training data.
        - test: DataFrame containing the test data.
        - X_train: DataFrame containing the features of the training data.
        - X_test: DataFrame containing the features of the test data.
        - y_train: Target values of the training data.
        - y_test: Target values of the test data.
    """
    try:
        X = data.drop(target, axis=1)
        try:
            y = data[target]
        except KeyError:
            logger.error("Target column %s does not exist.", target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)
        return train, test, X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error("Error occurred while splitting the data: %s", str(e))
        raise


def save_data(train: pd.DataFrame, test: pd.DataFrame, artifacts: Path) -> None:
    """
    Save the training and test data to CSV files.

    Args:
        train: DataFrame containing the training data.
        test: DataFrame containing the test data.
        artifacts: Path to the directory where the data files will be saved.
    """
    try:
        train.to_csv(artifacts / "train.csv", index=False)
        test.to_csv(artifacts / "test.csv", index=False)
        logger.debug("Training and test data saved to %s", artifacts)
    except Exception as e:
        logger.error("Error occurred while saving the data: %s", str(e))
        raise

def save_model(model, artifacts: Path) -> None:
    """
    Save the model.
    Args:
        model (sklearn model): Trained model object.
        artifacts (str): Path to save artifacts to.
    Returns:
        None(model is saved to artifacts)
    """
    with open(artifacts, "wb") as f:
        pickle.dump(model, f)
    logger.debug("Model saved to %s", artifacts)


def logistic_regression(X_train: Any, y_train: Any, params: Dict[str, Any], cv: int = 5,
                        random_seed: int = 42, scoring: str = 'roc_auc') -> Tuple[Any, Dict[str, Any], float]:
    """
    Perform logistic regression using pipeline and grid search.

    Args:
        X_train: Training features.
        y_train: Training target.
        params: Dictionary of parameter grid for grid search.
        cv: Number of cross-validation folds (default: 5).
        random_seed: Random seed for reproducibility (default: 42).
        scoring: Scoring metric for grid search (default: 'roc_auc').

    Returns:
        Tuple containing the best estimator, best parameters, and best score.
    """
    try:
        model = Pipeline([
            ('sampling', SMOTE(random_state=random_seed)),
            ('classification', LogisticRegression())
        ])
        out = GridSearchCV(model, param_grid=params, scoring=scoring, cv=cv)
        out.fit(X_train, y_train)
        logger.debug("Best parameters: %s", out.best_params_)
        logger.debug("Best score: %f", out.best_score_)
        return out.best_estimator_, out.best_params_, out.best_score_
    except Exception as e:
        logger.error("Error occurred while performing logistic regression: %s", str(e))
        raise



def histgbm_classification(X_train: Any, y_train: Any, 
                           params: Dict[str, Any], 
                           cv: int = 5,
                           random_seed: int = 42, 
                           n_iter: int = 10,
                           scoring: str = 'roc_auc') -> Tuple[Any, Dict[str, Any], float]:
    """
    Perform classification using HistGradientBoostingClassifier with grid search.

    Args:
        X_train: Training features.
        y_train: Training target.
        params: Dictionary of parameter grid for grid search.
        cv: Number of cross-validation folds (default: 5).
        random_seed: Random seed for reproducibility (default: 42).
        scoring: Scoring metric for grid search (default: 'roc_auc').

    Returns:
        Tuple containing the best estimator, best parameters, and best score.
    """
    try:
        model = HistGradientBoostingClassifier(random_state=random_seed)
        out = RandomizedSearchCV(model, param_distributions=params, scoring=scoring,n_iter=n_iter, cv=cv, random_state=random_seed)
        out.fit(X_train, y_train)
        logger.debug("Best parameters: %s", out.best_params_)
        logger.debug("Best score: %f", out.best_score_)
        return out.best_estimator_, out.best_params_, out.best_score_
    except Exception as e:
        logger.error("Error occurred while performing classification with HistGradientBoostingClassifier: %s", str(e))
        raise

def random_forest_classification(X_train: Any, y_train: Any, 
                                 params: Dict[str, Any], 
                                 cv: int = 5,
                                 n_iter: int = 10,
                                 random_seed: int = 42, 
                                 scoring: str = 'roc_auc') -> Tuple[Any, Dict[str, Any], float]:
    """
    Perform classification using Random Forest Classifier with pipeline and random search.

    Args:
        X_train: Training features.
        y_train: Training target.
        params: Dictionary of parameter distributions for random search.
        cv: Number of cross-validation folds (default: 5).
        random_seed: Random seed for reproducibility (default: 42).
        scoring: Scoring metric for random search (default: 'roc_auc').

    Returns:
        Tuple containing the best estimator, best parameters, and best score.
    """
    try:
        model = RandomForestClassifier(random_state=random_seed)
        out = RandomizedSearchCV(model, param_distributions=params, scoring=scoring,n_iter=n_iter, cv=cv, random_state=random_seed)
        out.fit(X_train, y_train)
        logger.debug("Best parameters: %s", out.best_params_)
        logger.debug("Best score: %f", out.best_score_)
        return out.best_estimator_, out.best_params_, out.best_score_
    except Exception as e:
        logger.error("Error occurred while performing classification with Random Forest Classifier: %s", str(e))
        raise