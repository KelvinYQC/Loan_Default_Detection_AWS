import os
from pathlib import Path

import logging
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import aws_utils as aws
import base64


logger = logging.getLogger(__name__)

# Load the loan dataset and trained classifier
BUCKET_NAME = os.getenv("BUCKET_NAME", "msia-423-group2-loan")
ARTIFACTS_PREFIX = Path(os.getenv("ARTIFACTS_PREFIX", "cloud_models/runs"))
MODEL_VERSION = Path(os.getenv("MODEL_VERSION", "random_forest_classification"))


@ st.cache_data
def load_data(data_file: Path, data_s3_key: str) -> np.ndarray:
    """
    Load data from aws s3 into memory.

    Args:
        data_file (Path): Local path to the data file.
        s3_key (str): S3 key for downloading the data file.

    Returns:
        np.ndarray: Loaded data.
    """
    print("Loading data from s3: ")

    aws.download_s3(BUCKET_NAME, data_s3_key, data_file)
    # Load files into memory
    dat = joblib.load(data_file)
    print("data loaded succesfully")
    return dat


@st.cache_resource
def load_model(model_path: Path, model_s3_key: str) -> np.ndarray:
    """
    Load model from aws s3 into memory.

    Args:
        model_path (Path): Local path to the model object.
        s3_key (str): S3 key for downloading the model object.

    Returns:
        model(obj): loaded model object.
    """
    print("Loading model from s3: ")
    aws.download_s3(BUCKET_NAME, model_s3_key, model_path)
    # load model into memory
    model = joblib.load(model_path)
    print("model loaded succesfully")
    return model


@ st.cache_data
def load_model_versions(bucket_name: str, prefix: str):
    """
    Load avalible model versions from S3 bucket

    Args:
        bucket_name (str): S3 bucket name
        prefix (str): S3 bucket prefix

    Returns:
        np.ndarray: Loaded data.
    """
    model_versions = aws.load_model_versions(bucket_name, prefix)
    return model_versions

# Create the application title and description
st.title("Loan Default Prediction")
st.write("Upload a CSV file with features and values to make loan predictions.")


st.subheader("Model Selection")
# Find available model versions in S3 bucket
available_models = load_model_versions(str(BUCKET_NAME), str(ARTIFACTS_PREFIX))
# Create a dropdown to select the model
model_version = st.selectbox("Select Model", list(available_models))
st.write(f"Selected model version: {model_version}")

# load data file and model object from S3 bucket to local directory
data_s3_key = str(ARTIFACTS_PREFIX / model_version / "data.joblib")
model_s3_key = str(ARTIFACTS_PREFIX / model_version / "classifier.joblib")

data_path = str(ARTIFACTS_PREFIX / model_version / "data.joblib")
model_path = str(ARTIFACTS_PREFIX / model_version / "classifier.joblib")

loaded_data = load_data(data_path, data_s3_key)
loaded_model = load_model(model_path, model_s3_key)

# CSV file upload
file = st.file_uploader("Upload CSV", type="csv")
if file is not None:
    try:
        # Perform predictions
        input_df = pd.read_csv(file)
        predictions = loaded_model.predict(input_df)
        probabilities = loaded_model.predict_proba(input_df)

        # Display the predictions and probabilities for each row
        st.subheader("Predictions")
        result_table = pd.DataFrame({
            "Row": range(1, len(input_df) + 1),
            "Prediction": [f"Default" if pred == 1 else f"Non-default" for pred, prob in zip(predictions, probabilities)],
            "Probability": [f"{prob[1]:.2%}" if pred == 1 else f"{1 - prob[1]:.2%}" for pred, prob in zip(predictions, probabilities)]
        })
        st.table(result_table.set_index('Row'))

        # Download link for the prediction table
        csv = result_table.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
        st.markdown(href, unsafe_allow_html=True)


        # Log prediction success
        logging.info("Prediction successfully performed.")

    except Exception as e:
        # Log prediction error
        logging.error(f"Error occurred during prediction: {str(e)}")
        st.error("Error occurred during prediction.")
        st.error(str(e))




