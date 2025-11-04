"""
Setup script for Gmail Dataset Creator.

Installs the package with CLI command support.
"""

from setuptools import setup, find_packages

with open("README_gmail_dataset.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements_gmail_dataset.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gmail-dataset-creator",
    version="1.0.0",
    author="Gmail Dataset Creator Team",
    description="Create email classification datasets from Gmail data using Gemini API",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications :: Email",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gmail-dataset-creator=gmail_dataset_creator.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)