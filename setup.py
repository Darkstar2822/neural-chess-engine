#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = []
    for line in fh:
        line = line.strip()
        # Skip comments and empty lines
        if line and not line.startswith("#"):
            # Remove inline comments
            requirement = line.split('#')[0].strip()
            if requirement:
                requirements.append(requirement)

setup(
    name="evolutionary-chess-engine",
    version="2.0.0",
    author="Evolutionary Chess Engine Team",
    author_email="research@chessai.dev",
    description="Revolutionary Co-Evolutionary Chess Engine with Neuroevolution and Anti-Stockfish Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/evolutionary-chess-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "gpu": [
            "cupy>=12.0.0",
            "torch[cuda]>=2.0.0",
        ],
        "fast": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "optuna>=3.0.0",
            "ray>=2.5.0",
        ],
        "visualization": [
            "plotly>=5.15.0",
            "bokeh>=3.0.0",
            "dash>=2.10.0",
            "seaborn>=0.12.0",
        ],
        "experimental": [
            "torch-geometric>=2.3.0",
            "neat-python>=0.92",
            "deap>=1.3.3",
            "pymoo>=0.6.0",
        ],
        "engines": [
            "stockfish>=3.28.0",
            "chess-com>=1.9.0",
            "lichess-python>=0.10.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "chess-engine=main:main",
            "chess-evolve=train_stockfish_killer:main",
            "chess-benchmark=benchmark_engines:run_benchmark",
        ],
    },
    include_package_data=True,
    package_data={
        "src.ui.static": ["images/*.png", "css/*.css", "js/*.js"],
        "src.ui.templates": ["*.html"],
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "chess", "ai", "machine-learning", "neural-networks", 
        "evolution", "neuroevolution", "genetic-algorithms",
        "stockfish", "alphazero", "game-ai", "reinforcement-learning",
        "neat", "topology-evolution", "multi-objective-optimization"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/evolutionary-chess-engine/issues",
        "Source": "https://github.com/yourusername/evolutionary-chess-engine",
        "Documentation": "https://evolutionary-chess-engine.readthedocs.io/",
    },
)