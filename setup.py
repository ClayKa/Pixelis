"""
Setup script for the Pixelis project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "Pixelis: A novel vision-language agent for pixel-space reasoning"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

# Development dependencies
dev_requirements = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-xdist>=3.0",
    "pytest-timeout>=2.0",
    "black>=24.0",
    "ruff>=0.2.0",
    "mypy>=1.8",
    "pre-commit>=3.0",
    "ipython>=8.0",
    "ipdb>=0.13",
    "bandit>=1.7",
    "safety>=3.0",
    "pip-audit>=2.0",
]

# Documentation dependencies
docs_requirements = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=2.0",
    "sphinx-autodoc-typehints>=1.25",
    "myst-parser>=2.0",
]

# Visualization dependencies
viz_requirements = [
    "matplotlib>=3.5",
    "seaborn>=0.12",
    "plotly>=5.0",
    "graphviz>=0.20",
]

setup(
    name="pixelis",
    version="0.1.0",
    author="Pixelis Team",
    author_email="pixelis@example.com",
    description="A novel vision-language agent for pixel-space reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pixelis",
    project_urls={
        "Documentation": "https://pixelis.readthedocs.io",
        "Source": "https://github.com/yourusername/pixelis",
        "Issues": "https://github.com/yourusername/pixelis/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*", "reference", "reference.*"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": docs_requirements,
        "viz": viz_requirements,
        "all": dev_requirements + docs_requirements + viz_requirements,
    },
    entry_points={
        "console_scripts": [
            "pixelis-train=scripts.train:main",
            "pixelis-eval=scripts.evaluate:main",
            "pixelis-demo=scripts.demo_reproducibility:main",
        ],
    },
    zip_safe=False,
)