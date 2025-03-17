"""Unit tests for model_utils.py."""

from collections import OrderedDict
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from util.model_utils import save_model

fixed_date = "2024-10-27"


# Mock model to simulate saving a PyTorch model
class MockModel(nn.Module):
    """Mock model to simulate saving a PyTorch model."""

    def forward(self, x):
        """Forward pass of the model."""
        return x


@pytest.fixture(name="fixed_datetime")
def fixture_datetime():
    """Fixture to mock datetime to a specific date."""
    with patch("util.model_utils.datetime") as mocked_datetime:
        mocked_datetime.now.return_value = datetime.strptime(fixed_date, "%Y-%m-%d")
        mocked_datetime.now.strftime.return_value = fixed_date.replace("-", "")
        yield mocked_datetime


@pytest.fixture(name="mocked_torch_save")
def fixture_torch_save(mocker):
    """Fixture to mock torch.save."""
    return mocker.patch("torch.save")


@pytest.fixture(name="mocked_properties")
def fixture_mocked_properties(mocker, tmp_path):
    """Fixture to mock Properties.get_instance with specific configurations."""
    # Create a mock instance of Properties
    mock_properties_instance = MagicMock()
    mock_properties_instance.train.save_format = "pt"

    # Set up a temporary directory for log_dir and ensure it exists
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    mock_properties_instance.system.log_dir = str(log_dir)

    # Patch Properties.get_instance to return the mock instance
    mocker.patch(
        "util.model_utils.Properties.get_instance",
        return_value=mock_properties_instance,
    )
    return mock_properties_instance


def test_save_model_with_date(fixed_datetime, mocked_torch_save, mocked_properties):
    """Test save_model with date."""
    model = MockModel()
    epoch = 1
    save_dir = "test_models"
    model_name = "test_vae_model"
    use_date = True

    # Assert that the mocked properties are set correctly
    assert mocked_properties.train.save_format == "pt"

    save_model(
        model=model,
        epoch=epoch,
        save_dir=save_dir,
        model_name=model_name,
        use_date=use_date,
    )
    saved_model = OrderedDict()

    # Verify that torch.save was called with the correct filename
    filename = (
        f"{save_dir}/{model_name}_{epoch}_{fixed_datetime.now().strftime('%Y%m%d')}.pt"
    )
    mocked_torch_save.assert_called_once_with(saved_model, filename)


def test_save_model_without_date(mocked_torch_save, mocked_properties):
    """Test save_model without date."""
    model = MockModel()
    epoch = 2
    save_dir = "test_models_no_date"
    model_name = "test_vae_model"

    # Assert that the mocked properties are set correctly
    assert mocked_properties.train.save_format == "pt"

    save_model(
        model=model,
        epoch=epoch,
        save_dir=save_dir,
        model_name=model_name,
    )
    saved_model = OrderedDict()

    # Verify that torch.save was called with the correct filename
    filename = f"{save_dir}/{model_name}_{epoch}.pt"
    mocked_torch_save.assert_called_once_with(saved_model, filename)


def test_save_model_invalid_model():
    """Test save_model with an invalid model."""
    epoch = 0
    with pytest.raises(
        TypeError, match="Expected 'model' to be an instance of nn.Module."
    ):
        save_model(model=None, epoch=epoch)
