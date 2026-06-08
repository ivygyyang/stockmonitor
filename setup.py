from setuptools import setup, find_packages

setup(
    name="stockmonitor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "yfinance>=0.2.40",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "rich>=13.0.0",
        "typer>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "stockmonitor=stockmonitor.cli:main",
        ],
    },
    python_requires=">=3.11",
)
