from setuptools import setup, find_packages
import os

# Read the long description from README
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name="quantedge",
    version="0.1.0",
    description="Cryptocurrency trading analysis with machine learning",
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author="Your Name",
    url="https://github.com/rezam747/quantedge",
    license="MIT",
    
    # Package configuration
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Project dependencies
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "yfinance==0.2.61",
        "scikit-learn>=1.0.0",
        "plotly>=5.18.0",
        "ta>=0.10.0",
        "python-binance>=1.0.0",
        "pandas-datareader>=0.10.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "xgboost>=3.0.2",
        "lightgbm>=4.6.0",
    ],
    
    # Development dependencies
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=3.0',
            'black>=22.0',
            'isort>=5.0',
            'flake8>=4.0',
            'mypy>=0.950',
        ],
    },
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'quantedge=quantedge.analytics.examples.btc_trading_example:main',
        ],
    },
    
    # Metadata
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    keywords='cryptocurrency trading machine-learning analysis',
)

