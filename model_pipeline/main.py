#!/usr/bin/python3

import os
import re
import warnings
import logging.config
import logging
from pathlib import Path
from time import sleep
import json
import joblib
import botocore
import typer
import yaml

import src.acquire_data as ad
import src.create_data as cd
import src.train_model as tm
import src.score_model as sm
import src.aws_utils as aws
import src.evaluate_performance as ep


# load config
logging.config.fileConfig("config/logging_config.conf")
logger = logging.getLogger("pipeline")

artifacts = Path() / "artifacts"
ARTIFACTS_PREFIX = os.getenv("ARTIFACTS_PREFIX", "artifacts/")
BUCKET_NAME = os.getenv("BUCKET_NAME", "msia-423-group2-loan")


def load_config(config_ref: str = 'config/pipeline_config.yaml') -> dict:
    """
    Load the configuration file specified by the `config_ref`.
    Args:
        config_ref (str): Optional. The reference to the configuration file.
                          Defaults to 'config/pipeline_config.yaml'.
    Returns:
        dict: The loaded configuration as a dictionary.
    Raises:
        EnvironmentError: If the specified config file does not exist.
    """
    if config_ref.startswith("s3://"):
        # Get config file from S3
        config_file = Path("config/downloaded-config.yaml")
        try:
            bucket, key = re.match(r"s3://([^/]+)/(.+)", config_ref).groups()
            aws.download_s3(bucket, key, config_file)
        except AttributeError:  # If re.match() does not return groups
            logger.error("Could not parse S3 URI: ", config_ref)
            config_file = Path("config/default.yaml")
        except botocore.exceptions.ClientError as e:  # If there is an error downloading
            logger.error("Unable to download config file from S3: ", config_ref)
    else:
        # Load config from local path
        config_file = Path(config_ref)
    if not config_file.exists():
        raise EnvironmentError(f"Config file at {config_file.absolute()} does not exist")
    logger.info("Loading config from: %s", config_file)
    with config_file.open() as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def run_pipeline(config, data_path=None):
    """
    Run the pipeline process with the given configuration.
    Args:
        config (dict): The configuration dictionary for the pipeline.
        data_path (str, optional): The path to the data. Defaults to None.
    Returns:
        None
    """
    logger.info("Starting local pipeline process run")
    run_config = config.get("run_config", {})
    aws_config = config.get("aws", {})
    version = run_config.get("version", "default")
    # Suppress all warnings
    warnings.filterwarnings(run_config.get('warnings_setting', 'ignore'))
    # Set up output directory for saving artifacts
    artifacts = Path(run_config.get("output", "runs"))
    artifacts = artifacts / version
    artifacts.mkdir(parents=True, exist_ok=True)
    # Save config file to artifacts directory for traceability
    with (artifacts / "pipelnie-config.yaml").open("w") as f:
        yaml.dump(config, f)
    if data_path is None:
        s3_key = "data/subsample_train.csv"
        BUCKET_NAME = aws_config.get("bucket_name", " msia-423-group2-loan")
        BUCKET_NAME = os.getenv("BUCKET_NAME", BUCKET_NAME)
        logger.info("Load data from S3: %s", s3_key)
        data = ad.load_data(artifacts / "loans.data", BUCKET_NAME, s3_key)
    else:
        s3_key = f"{data_path}"
        BUCKET_NAME = aws_config.get("bucket_name", " msia-423-group2-loan")
        BUCKET_NAME = os.getenv("BUCKET_NAME", BUCKET_NAME)
        logger.info("Load data from S3: %s", s3_key)
        data = ad.load_data(artifacts / "loans.data", BUCKET_NAME, s3_key)

    # Create structured dataset from raw data; save to disk
    data = cd.create_dataset(data, config["create_dataset"]["drop_columns"])
    cd.save_dataset(data, artifacts / "model_df.csv")

    # Split data into train/test set and train model based on config; save each to disk
    train, test, X_train, X_test, y_train, y_test = tm.split_data(data, **config["split_data"])
    # model
    model_type = run_config.get("model", "Logistic_classification")
    logger.info('model_type: %s', model_type)
    if model_type == "Logistic_classification":
        tmo, best_params_, best_score_ = \
            tm.logistic_regression(X_train,y_train, **config["Logistic_classification"])
    elif model_type == "histgbm_classification":
        tmo, best_params_, best_score_ = \
            tm.histgbm_classification(X_train,y_train, **config["histgbm_classification"])
    elif model_type == "random_forest_classification":
        tmo, best_params_, best_score_ = \
            tm.random_forest_classification(
                X_train,y_train,
                **config["random_forest_classification"])
    tm.save_data(train, test, artifacts)
    # Score model on test set; save scores to disk
    scores = sm.score_model(test, tmo, **config["score_model"])
    # Evaluate model performance metrics; save metrics to disk
    metrics = ep.evaluate_performance(scores,test, **config["evaluate_performance"])
    joblib.dump(data, artifacts / "data.joblib")
    joblib.dump(tmo, artifacts / "classifier.joblib")
    joblib.dump(metrics, artifacts / "metrics.joblib")
    logger.info("Local process run completed successfully")
    # # Upload all artifacts to S3
    aws_config = config.get("aws")
    if aws_config.get("upload", False):
        aws.upload_artifacts(artifacts, aws_config)
    logger.info("Process Finished")

def process_message(msg: aws.Message):
    """
    Process the given AWS message.
    Args:
        msg (aws.Message): The AWS message to process.
    Raises:
        ValueError: If the object key is unknown.
    """
    message_body = json.loads(msg.body)
    bucket_name = message_body["detail"]["bucket"]["name"]
    object_key = message_body["detail"]["object"]["key"]
    print(object_key)
    if object_key.startswith("configs/"):
        config_uri = f"s3://{bucket_name}/{object_key}"
        logger.info("Running pipeline with config from: %s", config_uri)
        config = load_config(config_uri)
        logger.info("Config loaded successfully")
        run_pipeline(config)
    elif object_key.startswith("data/"):
        data_uri = f"s3://{bucket_name}/{object_key}"
        logger.info("Running pipeline with data from: %s", data_uri)
        config = load_config()
        logger.info("Data loaded successfully")
        run_pipeline(config, data_path=object_key)
    else:
        logger.debug("Received message with unknown object key: %s", object_key)
        raise ValueError(f"Unknown object key: {object_key}")

def main(
    sqs_queue_url: str,
    max_empty_receives: int = 3,
    delay_seconds: int = 3,
    wait_time_seconds: int = 10,):
    """
    Polls an SQS queue for messages and processes them.
    Args:
        sqs_queue_url (str): The URL of the SQS queue to poll.
        max_empty_receives (int, optional): 
            The maximum number of consecutive empty receives before exiting the loop. Defaults to 3.
        delay_seconds (int, optional): 
            The delay in seconds before polling for more messages. Defaults to 3.
        wait_time_seconds (int, optional): 
            The maximum time in seconds to wait for a message. Defaults to 10.
    """
    empty_receives = 0
    while empty_receives < max_empty_receives:
        logger.info("Polling queue for messages...")
        messages = aws.get_messages(
            sqs_queue_url,
            max_messages=2,
            wait_time_seconds=wait_time_seconds,
        )
        logger.info("Received %d messages from queue", len(messages))
        if len(messages) == 0:
            # Increment our empty receive count by one if no messages come back
            empty_receives += 1
            sleep(delay_seconds)
            continue

        # Reset empty receive count if we get messages back
        empty_receives = 0
        for m in messages:
            # Perform work based on message content
            try:
                process_message(m)
            # We want to suppress all errors so that we can continue processing next message
            except Exception as e:
                logger.error("Unable to process message, continuing...")
                logger.error(e)
                continue
            # We must explicitly delete the message after processing it
            aws.delete_message(sqs_queue_url, m.handle)
        # Pause before asking the queue for more messages
        sleep(delay_seconds)


if __name__ == "__main__":
    typer.run(main)
