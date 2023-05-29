from pathlib import Path
import logging
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError

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
    aws_s3 = boto3.client("s3")
    print(f"Fetching Key: {object_key} from S3 Bucket: {bucket_name}")
    try:
        aws_s3.download_file(bucket_name, object_key, str(local_file_path))
        print(f"File downloaded successfully to {local_file_path}")
    except NoCredentialsError as e:
        logger.error("NoCredentialsError occurred while versions from bucket %s: %s",
                    bucket_name, str(e))
    except BotoCoreError as e:
        logger.error("BotoCoreError occurred while loading from bucket %s: %s",
                     bucket_name, str(e))


def load_model_versions(bucket_name: str, prefix: str):
    """
    load available model versions stored in S3 bucket

    Args:
        bucket_name (`str`): Name of the S3 bucket
        prefix (`str`): prefix of model artifacts in the S3 bucket

    Returns:
        model_versions (`list` of `str`): a list of available model versions
    """
    model_versions = []
    try:
        aws_s3 = boto3.client("s3", region_name="us-east-2")
        response = aws_s3.list_objects_v2(
            Bucket=bucket_name, Prefix=str(prefix) + "/", Delimiter="/")
        # Extract the subfolder names
        subfolders = [content.get("Prefix")
                      for content in response.get("CommonPrefixes", [])]
        model_versions = [Path(subfolder).name for subfolder in subfolders]
        logger.info("Loaded %d model versions from bucket %s with prefix %s", len(
            model_versions), bucket_name, prefix)
    except NoCredentialsError as e:
        logger.error("NoCredentialsError occurred while versions from bucket %s with prefix %s: %s",
                    bucket_name, prefix, str(e))
    except BotoCoreError as e:
        logger.error("BotoCoreError occurred while loading from bucket %s with prefix %s: %s",
                     bucket_name, prefix, str(e))

    return model_versions
