from setuptools import setup, find_packages

setup(
    name="financial-fraud-detection",
    version="0.1.0",
    description="Financial fraud detection project",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)