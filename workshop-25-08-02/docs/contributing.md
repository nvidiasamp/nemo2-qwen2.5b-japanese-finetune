# Contributing to Japanese Continual Learning with NeMo 2.0

Thank you for your interest in contributing to this project! We welcome contributions from the community and are excited to see what you can build with NeMo 2.0 and Japanese language models.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- Docker and nvidia-docker2
- Git
- Python 3.8+ (if developing outside Docker)

### First-time Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/japanese-continual-learning-nemo.git
   cd japanese-continual-learning-nemo
   ```

2. **Set up the development environment**
   ```bash
   # Run the setup script
   python scripts/setup_environment.py
   
   # Start the development container
   ./scripts/start_container.sh
   ```

3. **Install development dependencies**
   ```bash
   # Inside the container
   pip install -e ".[dev]"
   ```

## Development Setup

### Using Docker (Recommended)

```bash
# Start development container
./scripts/start_container.sh --interactive

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"
```

## Contribution Guidelines

### Types of Contributions

We welcome the following types of contributions:

- **Bug fixes**: Fix existing bugs or issues
- **New features**: Add new functionality to the project
- **Documentation**: Improve or add documentation
- **Examples**: Add new examples or tutorials
- **Performance improvements**: Optimize existing code
- **Tests**: Add or improve test coverage

### Before Contributing

1. **Check existing issues**: Look for existing issues or feature requests
2. **Create an issue**: For major changes, create an issue first to discuss
3. **Fork and branch**: Create a feature branch from `main`

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific test file
   pytest tests/test_models.py
   
   # Run with coverage
   pytest --cov=src tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add Japanese text preprocessing utility"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

## Code Style

### Python Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **isort**: Import sorting
- **MyPy**: Static type checking

### Running Code Quality Checks

```bash
# Format code with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with Flake8
flake8 src/ tests/

# Type checking with MyPy
mypy src/
```

### Pre-commit Hooks

We recommend setting up pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Code Style Guidelines

1. **Python**:
   - Follow PEP 8 style guide
   - Use type hints for function signatures
   - Write descriptive docstrings
   - Keep functions focused and small

2. **Documentation**:
   - Use clear, concise language
   - Include code examples
   - Keep documentation up to date

3. **Commit Messages**:
   - Use conventional commit format
   - Examples: `feat:`, `fix:`, `docs:`, `test:`

### Example Code Format

```python
def process_japanese_text(
    text: str,
    max_length: int = 512,
    tokenizer: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process Japanese text for model input.
    
    Args:
        text: Input Japanese text
        max_length: Maximum sequence length
        tokenizer: Tokenizer to use
        
    Returns:
        Processed text data
        
    Example:
        >>> result = process_japanese_text("こんにちは世界")
        >>> print(result['tokens'])
    """
    # Implementation here
    pass
```

## Testing

### Test Structure

```
tests/
├── __init__.py
├── test_models.py          # Model-related tests
├── test_data.py            # Data processing tests
├── test_training.py        # Training tests
├── test_evaluation.py      # Evaluation tests
└── fixtures/               # Test data and fixtures
```

### Writing Tests

1. **Test file naming**: `test_*.py`
2. **Test function naming**: `test_*`
3. **Use fixtures**: For setup and teardown
4. **Mock external dependencies**: Use pytest-mock

### Example Test

```python
import pytest
from src.models.import_qwen25 import import_qwen25_model

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path / "models"

def test_import_qwen25_model(temp_output_dir):
    """Test Qwen2.5 model import functionality."""
    # This would be a mock test in real scenario
    model_path = import_qwen25_model(
        model_size="0.5B",
        output_dir=str(temp_output_dir)
    )
    
    assert model_path.endswith(".nemo")
    assert Path(model_path).exists()
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run tests in parallel
pytest -n auto
```

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**: All tests should pass
2. **Update documentation**: Update relevant documentation
3. **Add/update tests**: Include tests for new functionality
4. **Check code style**: Run linting and formatting tools

### PR Template

When creating a pull request, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Screenshots/Results
Include any relevant screenshots or experimental results

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
```

### Review Process

1. **Automated checks**: GitHub Actions will run tests and checks
2. **Code review**: Maintainers will review your code
3. **Address feedback**: Make requested changes
4. **Merge**: Once approved, maintainers will merge

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Environment details**:
   - OS and version
   - Python version
   - NeMo version
   - GPU details

2. **Steps to reproduce**:
   - Exact commands used
   - Input data/configuration
   - Expected vs actual behavior

3. **Error logs**:
   - Full error messages
   - Stack traces
   - Log files (if applicable)

### Feature Requests

For feature requests, please include:

1. **Use case**: Describe why this feature would be useful
2. **Proposed solution**: How you envision it working
3. **Alternatives**: Other approaches you considered

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### Recognition

Contributors will be recognized in:
- README acknowledgments
- Release notes
- Contributor list

## Getting Help

If you need help with contributing:

1. Check existing documentation
2. Search through issues and discussions
3. Create a new issue with the "question" label
4. Join our community discussions

Thank you for contributing to Japanese Continual Learning with NeMo 2.0! 🎉 