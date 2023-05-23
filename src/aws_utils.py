import os
import sys
from pathlib import Path
import logging
import glob
import boto3
import botocore.exceptions

logger = logging.getLogger(__name__)

def download_s3(bucket_name: str, object_key: str, local_file_path) -> None:
    """
    Downloads a file from an S3 bucket and saves it to a local file path.

    Args:
        bucket_name (str): The name of the S3 bucket.
        object_key (str): The key of the object in the S3 bucket.
        local_file_path (Union[str, Path]): The local file path to save the downloaded file.

    Returns:
        None

    Raises:
        Exception: If there is an error during the file download.

    """
    local_file_path = Path(local_file_path)
    # Create the parent directory if it doesn't exist
    local_file_path.parent.mkdir(parents=True, exist_ok=True)  
    s3 = boto3.client("s3")
    print(f"Fetching Key: {object_key} from S3 Bucket: {bucket_name}")
    try:
        s3.download_file(bucket_name, object_key, str(local_file_path))
        print(f"File downloaded successfully to {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def upload_artifacts(artifacts: Path, config: dict) -> list[str]:
    """
    Upload all the artifacts in the specified directory to S3 using the default credential chain
    Args:
        artifacts: Directory containing all the artifacts from a given experiment
        config: Config required to upload artifacts to S3, includes: region_name, bucket_name, prefix
    Returns:
        List of S3 uri's for each file that was uploaded
    """
    # Get the bucket name from the environment variable or config
    bucket_name = os.environ.get("BUCKET_NAME", config.get("bucket_name"))
    profile_name = os.environ.get("PROFILE_NAME", None)

    # Create an S3 client with the default credentials and region
    try:
        if profile_name:
            session = boto3.Session(profile_name=profile_name, region_name=config["region_name"])
        else:
            session = boto3.Session(region_name=config["region_name"])
        connection = session.client("s3")

    except botocore.exceptions.BotoCoreError as e:
        logger.error("Failed to create boto3 session: %s", e)
        sys.exit(1)
    else:
        logger.debug("Created boto3 session")
    # Get a list of all files in the artifacts directory
    files = [str(file_path) for file_path in glob.glob("**/*.joblib", recursive=True) if os.path.isfile(file_path)]
    
    print(files)
    # Upload each file to S3
    s3_uris = []
    for file_path in files:
        # Create the S3 key by removing the artifacts directory from the file path
        s3_key = file_path.replace(str(artifacts) + "/", "")
        # Upload the file to S3
        s3_key = os.path.join(config["prefix"], file_path)
        try:
            connection.upload_file(file_path, bucket_name, s3_key)
        except botocore.exceptions.BotoCoreError as e:
            logger.error("Failed to upload file %s to S3 due to error: %s", file_path, e)
            return []
        else:
            logger.debug("Uploaded file %s to S3", file_path)
        # Add the S3 URI to the list of URIs
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        s3_uris.append(s3_uri)
    logger.info("Uploaded artifacts to S3 to bucket: %s", bucket_name)
    return s3_uris
