from setuptools import setup, find_packages

setup(
    name="fraud-detection-dashboard",
    version="1.0.0",
    description="Real-time Fraud Detection Dashboard with Machine Learning",
    author="Tshepo Manyisa",
    packages=find_packages(),
    install_requires=[
        "flask>=2.3.0",
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.18.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
    ],
)