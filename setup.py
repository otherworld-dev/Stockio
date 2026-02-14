from setuptools import setup, find_packages

setup(
    name="stockio",
    version="0.1.0",
    description="AI-powered stock trading bot that learns and improves over time",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "yfinance>=0.2.31",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "ta>=0.11.0",
        "feedparser>=6.0.0",
        "requests>=2.31.0",
        "transformers>=4.35.0",
        "alpaca-py>=0.21.0",
        "schedule>=1.2.0",
        "click>=8.1.0",
        "flask>=3.0.0",
        "gunicorn>=21.0.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "stockio=stockio.cli:main",
        ],
    },
)
