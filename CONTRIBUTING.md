# Contributing to Neural Chess Engine

We welcome contributions to the Neural Chess Engine! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/neural-chess-engine.git
   cd neural-chess-engine
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install with development dependencies
   ```

4. **Test Installation**
   ```bash
   python main.py init
   python check_device.py
   ```

## 🎯 Areas for Contribution

### High Priority
- **Neural Network Architectures**: Experiment with new network designs
- **Training Optimizations**: Improve training speed and efficiency  
- **Opening Book**: Expand the opening book with more positions
- **Position Evaluation**: Enhance position evaluation methods

### Medium Priority
- **User Interface**: Improve web interface design and functionality
- **Game Analysis**: Add position analysis and move quality scoring
- **Performance**: Optimize inference speed and memory usage
- **Documentation**: Improve code documentation and tutorials

### Creative Ideas
- **Strategy Discovery**: Help the AI discover new chess strategies
- **Unique Features**: Add innovative features not found in other engines
- **Visualization**: Create training progress and game analysis visualizations
- **Integration**: Connect with chess databases and analysis tools

## 🔧 Development Guidelines

### Code Style
- Follow PEP 8 Python style guide
- Use type hints where possible
- Add docstrings to functions and classes
- Keep functions focused and small

### Testing
- Write tests for new functionality
- Ensure all tests pass before submitting PR
- Test on multiple Python versions if possible

### Commits
- Use clear, descriptive commit messages
- Make atomic commits (one logical change per commit)
- Reference issues in commit messages where applicable

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_neural_network.py
```

### Manual Testing
```bash
# Test basic functionality
python main.py init
python main.py train --iterations 1 --games 5

# Test web interface
python main.py web --port 5001
```

## 📝 Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Test Your Changes**
   ```bash
   pytest
   python main.py init  # Test basic functionality
   ```

4. **Submit Pull Request**
   - Provide clear description of changes
   - Reference any related issues
   - Include screenshots for UI changes

## 🎮 Training Data Contributions

### Self-Play Games
- Contribute interesting self-play games
- Share unique strategies discovered by the AI
- Report unusual or creative moves

### Opening Positions
- Add new opening lines to the opening book
- Suggest creative opening variations
- Test opening performance

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version and OS
- GPU/CPU information
- Steps to reproduce
- Error messages and logs
- Expected vs actual behavior

## 💡 Feature Requests

For feature requests, please provide:
- Clear description of the feature
- Use case and motivation
- Potential implementation approach
- Examples from other chess engines if applicable

## 📊 Performance Benchmarks

When contributing performance improvements:
- Provide before/after benchmarks
- Test on different hardware if possible
- Document any trade-offs
- Include memory usage measurements

## 🎯 Chess Engine Specific Guidelines

### Neural Network Changes
- Test convergence on small datasets first
- Document architecture choices
- Consider memory and speed implications
- Validate against baseline models

### Training Improvements
- Measure training speed improvements
- Ensure training stability
- Test on different hardware configurations
- Document hyperparameter choices

### Game Logic
- Ensure full chess rule compliance
- Test edge cases (castling, en passant, promotion)
- Validate against standard chess libraries
- Consider performance implications

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤝 Community

- Be respectful and inclusive
- Help other contributors
- Share knowledge and insights
- Have fun building amazing chess AI!

Thank you for contributing to Neural Chess Engine! 🧠♟️