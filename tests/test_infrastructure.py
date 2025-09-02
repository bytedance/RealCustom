import pytest
import sys
import os
from pathlib import Path
import importlib.util


class TestInfrastructure:
    """Test suite to validate the testing infrastructure setup."""

    def test_pytest_working(self):
        """Test that pytest is working correctly."""
        assert True

    def test_python_version(self):
        """Test that Python version is compatible."""
        assert sys.version_info >= (3, 8), "Python 3.8+ required"

    def test_project_structure(self):
        """Test that the project structure exists."""
        project_root = Path(__file__).parent.parent
        
        # Check main directories
        assert (project_root / "models").exists()
        assert (project_root / "inference").exists()
        assert (project_root / "schedulers").exists()
        
        # Check config files
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "README.md").exists()

    def test_testing_directories(self):
        """Test that testing directories are properly set up."""
        tests_dir = Path(__file__).parent
        
        assert tests_dir.name == "tests"
        assert (tests_dir / "__init__.py").exists()
        assert (tests_dir / "conftest.py").exists()
        assert (tests_dir / "unit").exists()
        assert (tests_dir / "integration").exists()
        assert (tests_dir / "unit" / "__init__.py").exists()
        assert (tests_dir / "integration" / "__init__.py").exists()

    def test_imports_available(self):
        """Test that essential modules can be imported."""
        try:
            import pytest
            import pytest_cov
            import pytest_mock
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import testing dependencies: {e}")

    def test_fixtures_available(self, temp_dir, sample_config, mock_model):
        """Test that conftest fixtures are available."""
        assert temp_dir.exists()
        assert isinstance(sample_config, dict)
        assert sample_config["model_name"] == "test_model"
        assert mock_model is not None

    def test_project_modules_importable(self):
        """Test that main project modules can be imported."""
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        try:
            # Test that we can find the main modules
            models_path = project_root / "models"
            inference_path = project_root / "inference"
            schedulers_path = project_root / "schedulers"
            
            assert models_path.exists()
            assert inference_path.exists()
            assert schedulers_path.exists()
            
            # Check for Python files in each directory
            assert any(models_path.glob("*.py"))
            assert any(inference_path.glob("*.py"))
            assert any(schedulers_path.glob("*.py"))
            
        finally:
            sys.path.pop(0)

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        assert True

    def test_temp_dir_fixture(self, temp_dir):
        """Test that temporary directory fixture works."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can write to it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.read_text() == "test content"

    def test_mock_functionality(self, mock_model, mock_optimizer):
        """Test that mock fixtures work properly."""
        # Test mock model
        mock_model.forward.return_value = "mocked_output"
        result = mock_model.forward()
        assert result == "mocked_output"
        mock_model.forward.assert_called_once()
        
        # Test mock optimizer
        mock_optimizer.step()
        mock_optimizer.step.assert_called_once()

    def test_sample_data_fixtures(self, sample_config, sample_text_prompts):
        """Test that sample data fixtures provide expected data."""
        assert isinstance(sample_config, dict)
        assert "model_name" in sample_config
        assert "batch_size" in sample_config
        
        assert isinstance(sample_text_prompts, list)
        assert len(sample_text_prompts) > 0
        assert all(isinstance(prompt, str) for prompt in sample_text_prompts)

    def test_cleanup_functionality(self, cleanup_files, temp_dir):
        """Test that cleanup fixture works."""
        # Create a test file
        test_file = temp_dir / "cleanup_test.txt"
        test_file.write_text("cleanup test")
        cleanup_files(str(test_file))
        
        assert test_file.exists()
        # File should be cleaned up automatically after test