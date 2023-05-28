#!/usr/bin/python3

import argparse
import datetime
import os
import re
import warnings
import logging.config
from pathlib import Path
import yaml
import logging
import joblib
import botocore
import json
from time import sleep
import typer

import src.acquire_data as ad
import src.create_data as cd
import src.train_model as tm
import src.score_model as sm
import src.aws_utils as aws
import src.evaluate_performance as ep



logging.config.fileConfig("config/logging_config.conf")
logger = logging.getLogger("pipeline")

artifacts = Path() / "artifacts"
ARTIFACTS_PREFIX = os.getenv("ARTIFACTS_PREFIX", "artifacts/")
BUCKET_NAME = os.getenv("BUCKET_NAME", "msia-423-group2-loan")


def load_config(config_ref: str) -> dict:
    if config_ref.startswith("s3://"):
        # Get config file from S3
        config_file = Path("config/downloaded-config.yaml")
        try:
            bucket, key = re.match(r"s3://([^/]+)/(.+)", config_ref).groups()
            aws.download_s3(bucket, key, config_file)
        except AttributeError:  # If re.match() does not return groups
            print("Could not parse S3 URI: ", config_ref)
            config_file = Path("config/default.yaml")
        except botocore.exceptions.ClientError as e:  # If there is an error downloading
            print("Unable to download config file from S3: ", config_ref)
            print(e)
    else:
        # Load config from local path
        config_file = Path(config_ref)
    if not config_file.exists():
        raise EnvironmentError(f"Config file at {config_file.absolute()} does not exist")

    with config_file.open() as f:
        return yaml.load(f, Loader=yaml.SafeLoader)
    
def run_pipeline(config):
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

    s3_key = str("subsample_train.csv")
    BUCKET_NAME = aws_config.get("bucket_name", " msia-423-group2-loan")
    BUCKET_NAME = os.getenv("BUCKET_NAME", BUCKET_NAME)
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
        tmo, best_params_, best_score_ = tm.logistic_regression(X_train,y_train, **config["Logistic_classification"])
    elif model_type == "histgbm_classification":
        tmo, best_params_, best_score_ = tm.histgbm_classification(X_train,y_train, **config["histgbm_classification"])
    elif model_type == "random_forest_classification":
        tmo, best_params_, best_score_ = tm.random_forest_classification(X_train,y_train, **config["random_forest_classification"])
    tm.save_data(train, test, artifacts)
    # tm.save_model(tmo, artifacts / "trained_model_object.pkl")

    # Score model on test set; save scores to disk
    scores = sm.score_model(test, tmo, **config["score_model"])
    # sm.save_scores(scores, artifacts / "scores.csv")
    
    # Evaluate model performance metrics; save metrics to disk
    metrics = ep.evaluate_performance(scores,test, **config["evaluate_performance"])
    # ep.save_metrics(metrics, artifacts / "metrics.yaml")
    
        
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
    message_body = json.loads(msg.body)
    bucket_name = message_body["detail"]["bucket"]["name"]
    object_key = message_body["detail"]["object"]["key"]
    config_uri = f"s3://{bucket_name}/{object_key}"
    logger.info("Running pipeline with config from: %s", config_uri)
    config = load_config(config_uri)
    run_pipeline(config)

def main(
    sqs_queue_url: str,
    max_empty_receives: int = 3,
    delay_seconds: int = 3,
    wait_time_seconds: int = 10,):
    

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
