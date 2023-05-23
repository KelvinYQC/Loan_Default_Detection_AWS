import argparse
import datetime
import os
import warnings
import logging.config
from pathlib import Path
import yaml
import logging
import joblib

import src.acquire_data as ad
import src.create_data as cd
import src.train_model as tm
import src.score_model as sm
import src.aws_utils as aws
import src.evaluate_performance as ep



logging.config.fileConfig("config/logging_config.conf")
logger = logging.getLogger("pipeline")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipelines for training and evaluating a model for predicting Loan Default"
    )
    parser.add_argument(
        "--config", default="config/pipeline_config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r", encoding='utf-8') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error("Error while loading configuration from %s", args.config)
        else:
            logger.info("Configuration file loaded from %s", args.config)

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
    data = ad.load_data(artifacts / "clouds.data", BUCKET_NAME, s3_key)

    # Create structured dataset from raw data; save to disk
    data = cd.create_dataset(data, config["create_dataset"]["drop_columns"])
    cd.save_dataset(data, artifacts / "model_df.csv")

    # Split data into train/test set and train model based on config; save each to disk
    train, test, X_train, X_test, y_train, y_test = tm.split_data(data, **config["split_data"])
    # logistic regression
    tmo, logustic_best_params_, logustic_best_score_ = tm.logistic_regression(X_train,y_train, **config["Logistic_regression"])
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
