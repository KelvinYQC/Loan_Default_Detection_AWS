import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from app import load_data
from src.aws_utils import load_model_versions


BUCKET_NAME = os.getenv("BUCKET_NAME", "msia-423-group2-loan")
ARTIFACTS_PREFIX = Path(os.getenv("ARTIFACTS_PREFIX", "cloud_models/runs"))


def test_load_model_versions_happy():
    """test available model versions are loaded from S3 bucket"""
    result = load_model_versions(BUCKET_NAME, ARTIFACTS_PREFIX)
    assert result == ["Logistic_classification", "deep-random-forest", "histgbm_classification", "random_forest_classification"]


def test_load_model_versions_unhappy():
    """pass in empty list and integer as bucket name and prefix"""
    with pytest.raises(TypeError):
        load_model_versions([], 0)


def test_load_data_happy():
    """test load data properly from S3 bucket"""
    data = {
    'AMT_INCOME_TOTAL': [247500.0, 247500.0, 112500.0, 141606.0, 270000.0],
    'AMT_CREDIT': [1281712.5, 254700.0, 308133.0, 810000.0, 593010.0],
    'AMT_ANNUITY': [48946.5, 24939.0, 15862.5, 33120.0, 17122.5],
    'AMT_GOODS_PRICE': [1179000.0, 225000.0, 234000.0, 810000.0, 495000.0],
    'REGION_POPULATION_RELATIVE': [0.006852, 0.046220, 0.018850, 0.018801, 0.005144]
    }
    # first two rows of loaded data
    expected_df = pd.DataFrame(data).round(6)
    dat = load_data("cloud_models/runs/random_forest_classification/data.joblib", "cloud_models/runs/random_forest_classification/data.joblib")
    # get first 5 columns of loaded data
    loaded_df = pd.DataFrame(dat).head().iloc[:, :5].reset_index(drop=True)
    pd.testing.assert_frame_equal(loaded_df, expected_df, check_exact=False, atol=1e-5)


def test_load_data_unhappy():
    """pass in integer and empty string as bucket name and prefix"""
    with pytest.raises(TypeError):
        load_data(423, "")