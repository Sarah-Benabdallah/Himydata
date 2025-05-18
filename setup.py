from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="himydata",
    version="0.1.0",
    packages=find_packages(),
    description="Product Clustering Analysis",
    author="Sarah Benabdallah",
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'himydata=main:main',
            'himydata-explore=scripts.run_exploration:run_exploration_interactive',
            'himydata-cluster=scripts.run_clustering:run_clustering_interactive',
        ],
    },
)