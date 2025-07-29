# Contributing to NeMo 2.0 Qwen2.5B Japanese Fine-tuning

We welcome contributions to this project! This document provides guidelines for contributing.

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/nemo2-qwen2.5b-japanese-finetune.git
   cd nemo2-qwen2.5b-japanese-finetune
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev,docs,notebooks]"
   ```

4. **Setup Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## 📝 Development Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:
```bash
black src scripts examples
isort src scripts examples
flake8 src scripts examples
mypy src --ignore-missing-imports
```

### Code Structure

- **Modular Design**: Keep components focused and reusable
- **Type Hints**: Use type hints for all public functions
- **Docstrings**: Use Google-style docstrings
- **Error Handling**: Provide clear error messages and proper exception handling
- **Logging**: Use the provided logging utilities

### Example Code Style

```python
#!/usr/bin/env python3
"""
Module description here.
"""

import os
from typing import Optional, Dict, Any

from nemo_japanese_ft.utils import setup_logging

logger = setup_logging(__name__)


class ExampleClass:
    """Example class with proper documentation."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the example class.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("ExampleClass initialized")
    
    def process_data(self, data: str, validate: bool = True) -> Optional[str]:
        """
        Process the input data.
        
        Args:
            data: Input data to process
            validate: Whether to validate the data
            
        Returns:
            Processed data or None if processing fails
            
        Raises:
            ValueError: If data is invalid and validate=True
        """
        if validate and not data:
            raise ValueError("Data cannot be empty")
        
        try:
            result = data.strip().lower()
            logger.debug(f"Processed data: {result}")
            return result
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
```

## 🧪 Testing

### Test Structure

- **Unit Tests**: `tests/unit/` - Test individual components
- **Integration Tests**: `tests/integration/` - Test component interactions
- **Test Fixtures**: `tests/conftest.py` - Shared test utilities

### Writing Tests

```python
import pytest
from nemo_japanese_ft.models import QwenModelConfig


@pytest.mark.unit
class TestQwenModelConfig:
    """Test Qwen model configuration."""
    
    def test_init_default(self):
        """Test default initialization."""
        config = QwenModelConfig()
        assert config.model_size == "0.5b"
        assert config.seq_length == 2048
    
    def test_validation_error(self):
        """Test validation raises appropriate errors."""
        with pytest.raises(ValueError, match="Invalid model size"):
            QwenModelConfig(model_size="invalid")
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run with coverage
pytest --cov=src tests/

# Run specific markers
pytest -m unit
pytest -m "not gpu"
```

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Example function with proper documentation.
    
    This function demonstrates the expected docstring format
    for this project.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter with default value
        
    Returns:
        Boolean indicating success or failure
        
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer
        
    Example:
        >>> result = example_function("test", 20)
        >>> assert result is True
    """
    pass
```

### Building Documentation

```bash
cd docs
make html
make linkcheck  # Check for broken links
```

## 🔄 Pull Request Process

### Before Submitting

1. **Run Tests**: Ensure all tests pass
   ```bash
   pytest tests/
   ```

2. **Check Code Quality**: Run linting and formatting
   ```bash
   black src scripts examples
   flake8 src scripts examples
   mypy src
   ```

3. **Update Documentation**: Add or update relevant documentation

4. **Add Tests**: Include tests for new functionality

### PR Guidelines

1. **Clear Title**: Use descriptive titles (e.g., "Add PEFT support for Qwen2.5-7B models")

2. **Detailed Description**: Include:
   - What changes were made
   - Why the changes were necessary
   - How to test the changes
   - Any breaking changes

3. **Link Issues**: Reference related issues with "Fixes #123" or "Addresses #123"

4. **Small PRs**: Keep PRs focused and reasonably sized

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] New tests added for new functionality

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Try to reproduce with latest version
3. Gather relevant information

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. See error

**Expected Behavior**
What should have happened

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- NeMo: [e.g., 2.0.0]
- Package Version: [e.g., 0.1.0]

**Additional Context**
Any other relevant information
```

## ✨ Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature

**Motivation**
Why is this feature needed?

**Proposed Solution**
How should this be implemented?

**Alternative Solutions**
Any alternative approaches considered?

**Additional Context**
Any other relevant information
```

## 🏷️ Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release tag
5. Update documentation

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the docs first

## 🙏 Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing to the NeMo Japanese Fine-tuning project! 