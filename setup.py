from setuptools import setup, find_packages

setup(
    name="stockio",
    version="0.1.0",
    description="AI-powered stock trading bot that learns and improves over time",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "stockio=stockio.cli:main",
        ],
    },
)
