from pathlib import Path
from src import evaluate_performance as ep
from typing import Any, List, Tuple
import pytest


@pytest.fixture
def sample_result() -> List[Tuple[str, float]]:
    '''
    Generate a sample result for testing purposes.
    Returns:
        A list of tuples containing metric names and corresponding values.
    '''
    return [('accuracy', 0.85), ('precision', 0.76), ('recall', 0.92)]


def test_save_metrics(tmp_path: Path, sample_result: List[Tuple[str, float]]) -> None:
    '''
    Test the save_metrics function.
    Args:
        tmp_path: A Path object representing a temporary directory for testing.
        sample_result: A sample result to save.
        caplog: A pytest fixture for capturing log messages.
    Returns:
        None
    '''
    output_file = tmp_path / "test_metrics.yaml"
    ep.save_metrics(sample_result, str(output_file))
    assert output_file.is_file()


def test_save_metrics_error(tmp_path: Path, sample_result: List[Tuple[str, float]]) -> None:
    '''
    Test the save_metrics function when a non-existing directory is provided.
    Args:
        tmp_path: A Path object representing a temporary directory for testing.
        sample_result: A sample result to save.
    Returns:
        None
    '''
    output_file = tmp_path / "non_existing_directory" / "test_metrics.yaml"
    with pytest.raises(FileNotFoundError):
        ep.save_metrics(sample_result, str(output_file))
