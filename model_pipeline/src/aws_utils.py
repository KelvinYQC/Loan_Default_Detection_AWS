import os
import sys
from pathlib import Path
import logging
import glob
from dataclasses import dataclass
import boto3
import botocore.exceptions

logger = logging.getLogger(__name__)

def download_s3(bucket_name: str, object_key, local_file_path) -> None:
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
    object_key = str(object_key)
    local_file_path = Path(local_file_path)
    logger.info("Local file path: %s", local_file_path)
    # Create the parent directory if it doesn't exist
    local_file_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    logger.info("Fetching Key: %s from Bucket: %s", object_key, bucket_name)
    try:
        s3.download_file(bucket_name, object_key, str(local_file_path))
        logger.info("File downloaded successfully to% s", local_file_path)
    except Exception as e:
        logger.error("Error downloading file %s from bucket %s: %s", object_key, bucket_name, e)

def upload_artifacts(artifacts: Path, config: dict) -> list[str]:
    """
    Upload all the artifacts in the specified directory to S3 using the default credential chain
    Args:
        artifacts: Directory containing all the artifacts from a given experiment
        config: Config required to upload artifacts to S3
    Returns:
        List of S3 uri's for each file that was uploaded
    """
    # Get the bucket name from the environment variable or config
    bucket_name = os.environ.get("BUCKET_NAME",
                                 config.get("bucket_name"))
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


@dataclass
class Message:
    handle: str
    body: str


def get_messages( queue_url: str,
    max_messages: int = 1,
    wait_time_seconds: int = 20,
    ) -> list[Message]:
    """
    Retrieves messages from an Amazon Simple Queue Service (SQS) queue.
    Args:
        queue_url (str): The URL of the SQS queue.
        max_messages (int, optional): The maximum number of messages to retrieve. Defaults to 1.
        wait_time_seconds (int, optional): The duration (in seconds) to wait for messages if the queue is empty.
                                           Defaults to 20.
    Returns:
        list[Message]: A list of Message objects representing the retrieved messages.
    """
    sqs = boto3.client("sqs")
    try:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time_seconds,
        )
    except botocore.exceptions.ClientError as e:
        logger.error(e)
        return []
    if "Messages" not in response:
        return []
    return [Message(m["ReceiptHandle"], m["Body"]) for m in response["Messages"]]


def delete_message(queue_url: str, receipt_handle: str):
    """
    Deletes a message from an SQS queue.
    Args:
        queue_url (str): The URL of the SQS queue.
        receipt_handle (str): The receipt handle of the message to delete.
    """
    sqs = boto3.client("sqs")
    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
