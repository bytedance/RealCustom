import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch
import torch
import numpy as np
from typing import Dict, Any, Generator


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for testing."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    yield temp_file_path


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration dictionary for testing."""
    return {
        "model_name": "test_model",
        "batch_size": 4,
        "learning_rate": 0.001,
        "num_epochs": 10,
        "device": "cpu",
        "seed": 42,
        "output_dir": "/tmp/test_output",
        "checkpoint_interval": 5
    }


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing."""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.device', return_value=torch.device('cpu')):
            yield torch.device('cpu')


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def sample_image_batch():
    """Create a sample batch of images as numpy arrays."""
    return np.random.rand(4, 224, 224, 3).astype(np.float32)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.forward.return_value = torch.randn(4, 10)
    model.eval.return_value = model
    model.train.return_value = model
    model.parameters.return_value = [torch.randn(10, 5), torch.randn(10)]
    return model


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler for testing."""
    scheduler = Mock()
    scheduler.step.return_value = None
    scheduler.get_last_lr.return_value = [0.001]
    return scheduler


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing."""
    optimizer = Mock()
    optimizer.step.return_value = None
    optimizer.zero_grad.return_value = None
    optimizer.state_dict.return_value = {"state": {}, "param_groups": []}
    return optimizer


@pytest.fixture
def sample_text_prompts():
    """Provide sample text prompts for testing."""
    return [
        "a beautiful sunset over the ocean",
        "a cat sitting on a windowsill",
        "abstract digital art with vibrant colors",
        "a cozy reading nook with books and tea"
    ]


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.vocab_size = 10000
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def mock_image_encoder():
    """Create a mock image encoder for testing."""
    encoder = Mock()
    encoder.encode.return_value = torch.randn(1, 512)
    return encoder


@pytest.fixture
def mock_diffusion_pipeline():
    """Create a mock diffusion pipeline for testing."""
    pipeline = Mock()
    pipeline.generate.return_value = [np.random.rand(512, 512, 3)]
    pipeline.device = torch.device('cpu')
    return pipeline


@pytest.fixture
def sample_checkpoint_data():
    """Provide sample checkpoint data for testing."""
    return {
        "epoch": 10,
        "model_state_dict": {"layer1.weight": torch.randn(10, 5)},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"_step_count": 100},
        "loss": 0.5,
        "metrics": {"accuracy": 0.85, "f1_score": 0.82}
    }


@pytest.fixture
def mock_config_file(temp_dir: Path):
    """Create a mock configuration file for testing."""
    config_file = temp_dir / "config.json"
    config_data = {
        "model": {
            "name": "test_model",
            "architecture": "transformer"
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.001
        }
    }
    import json
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    return config_file


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    logger.info.return_value = None
    logger.warning.return_value = None
    logger.error.return_value = None
    logger.debug.return_value = None
    return logger


@pytest.fixture
def sample_mask():
    """Create a sample mask tensor for testing."""
    return torch.ones(1, 224, 224, dtype=torch.bool)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_huggingface_model():
    """Create a mock HuggingFace model for testing."""
    with patch('transformers.AutoModel.from_pretrained') as mock_model:
        mock_instance = Mock()
        mock_instance.config.hidden_size = 768
        mock_instance.forward.return_value = Mock(last_hidden_state=torch.randn(1, 10, 768))
        mock_model.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_gradio_interface():
    """Create a mock Gradio interface for testing."""
    interface = Mock()
    interface.launch.return_value = None
    interface.close.return_value = None
    return interface


@pytest.fixture
def cleanup_files():
    """Fixture to clean up generated files after tests."""
    generated_files = []
    
    def _add_file(filepath):
        generated_files.append(filepath)
    
    yield _add_file
    
    # Cleanup after test
    for filepath in generated_files:
        if os.path.exists(filepath):
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)