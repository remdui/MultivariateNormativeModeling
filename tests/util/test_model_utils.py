from datetime import datetime
from unittest.mock import patch

import pytest
from torch import nn

from util.model_utils import save_model

fixed_date = "2024-10-27"


# Mock model to simulate saving a PyTorch model
class MockModel(nn.Module):
    def forward(self, x):
        return x


@pytest.fixture(name="fixed_datetime")
def fixture_datetime():
    """Fixture to mock datetime to a specific date."""
    with patch("util.model_utils.datetime") as mocked_datetime:
        mocked_datetime.now.return_value = datetime.strptime(fixed_date, "%Y-%m-%d")
        mocked_datetime.now.strftime.return_value = fixed_date.replace("-", "")
        yield mocked_datetime


@pytest.fixture(name="mocked_os_makedirs")
def fixture_os_makedirs(mocker):
    """Fixture to mock os.makedirs."""
    return mocker.patch("os.makedirs")


@pytest.fixture(name="mocked_torch_save")
def fixture_torch_save(mocker):
    """Fixture to mock torch.save."""
    return mocker.patch("torch.save")


def test_save_model_with_date(fixed_datetime, mocked_torch_save, mocked_os_makedirs):
    """Test save_model with date."""
    model = MockModel()
    epoch = 1
    save_dir = "test_models"
    model_name = "test_vae_model"
    use_date = True
    save_model(
        model, epoch, save_dir=save_dir, model_name=model_name, use_date=use_date
    )

    # Check if os.makedirs was called to create the directory
    mocked_os_makedirs.assert_called_once_with(save_dir)

    # Verify that torch.save was called with the correct filename
    filename = (
        f"{save_dir}/{model_name}_{epoch}_{fixed_datetime.now().strftime("%Y%m%d")}.pt"
    )
    mocked_torch_save.assert_called_once_with(model, filename)


def test_save_model_without_date(mocked_os_makedirs, mocked_torch_save):
    """Test save_model without date."""
    model = MockModel()
    epoch = 2
    save_dir = "test_models_no_date"
    model_name = "test_vae_model"
    use_date = False
    save_model(
        model, epoch, save_dir=save_dir, model_name=model_name, use_date=use_date
    )

    # Check if os.makedirs was called to create the directory
    mocked_os_makedirs.assert_called_once_with(save_dir)

    # Verify that torch.save was called with the correct filename
    filename = f"{save_dir}/{model_name}_{epoch}.pt"
    mocked_torch_save.assert_called_once_with(model, filename)


def test_save_model_invalid_model():
    epoch = 0
    with pytest.raises(ValueError, match="Model is invalid."):
        save_model(None, epoch)


def test_save_model_invalid_epoch():
    model = MockModel()
    with pytest.raises(ValueError, match="Epoch cannot be negative."):
        save_model(model, -1)


def test_save_model_directory_creation(mocked_torch_save, mocker, mocked_os_makedirs):
    model = MockModel()
    epoch = 0
    save_dir = "new_dir"

    # Patch os.path.exists to return False, forcing directory creation
    mocker.patch("os.path.exists", return_value=False)

    save_model(model, epoch, save_dir=save_dir)

    # Check if os.makedirs was called to create the directory
    mocked_os_makedirs.assert_called_once_with(save_dir)

    # Verify torch.save was called
    mocked_torch_save.assert_called_once()
