from setuptools import find_packages, setup

setup(
    name="f1-race-prediction",
    version="1.0.0",
    description="F1 Race Prediction and Strategy Recommendation System",
    packages=find_packages(include=["src*", "api*"]),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "httpx>=0.24.0",
        "dvc>=3.0.0",
        "joblib>=1.2.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "flake8",
            "black",
            "isort",
        ]
    },
)
