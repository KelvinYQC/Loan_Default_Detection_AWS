from pathlib import Path
import pytest
import pandas as pd
from src import create_data as cd


@pytest.fixture
def sample_data() -> pd.DataFrame:
    '''
    Generate a sample DataFrame for testing purposes.
    Returns:
        A pandas DataFrame with example data.
    '''
    return pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

def test_save_dataset(tmp_path: Path, sample_data: pd.DataFrame) -> None:
    '''
    Test the save_dataset function.
    Args:
        tmp_path: A Path object representing a temporary directory for testing.
        sample_data: A sample DataFrame to save.
    Returns:
        None
    '''
    output_file = tmp_path / "test_data.csv"
    cd.save_dataset(sample_data, str(output_file))
    assert output_file.is_file()

def test_save_dataset_file_not_found(tmp_path: Path) -> None:
    '''
    Test the save_dataset function when a non-existing directory is provided.
    Args:
        tmp_path: A Path object representing a temporary directory for testing.
    Returns:
        None
    '''
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    output_file = tmp_path / "nonexistent_directory" / "test_data.csv"

    with pytest.raises(OSError) as exc_info:
        cd.save_dataset(data, output_file)

    expected_error_message = f"Cannot save file into a non-existent directory: '{output_file.parent}'"
    assert str(exc_info.value) == expected_error_message
