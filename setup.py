#!/usr/bin/env python3
"""
The message "Please avoid running setup.py directly.
Instead, use pypa/build, pypa/installer or other standards-based tools" indicates that direct
invocation of setup.py is a deprecated practice in Python packaging. This is due to several reasons, 
including the inability of setup.py to manage its own dependencies effectively and the shift towards 
standardized build backends and frontends.

Recommended Alternatives:
Building Distributions: Instead of python setup.py sdist or python setup.py bdist_wheel,
use python -m build. This command will create source distributions (sdist) and/or wheel
distributions (bdist_wheel) based on your project's configuration (typically in pyproject.toml).
Code

    python -m build
Installing Packages: Instead of python setup.py install, use pip install ..
This installs your package in editable mode if you are developing it, or as a regular
installation if not. For editable installs, use pip install --editable ..
Code

    pip install .
Code

    pip install --editable .

Standards-Based Tools: The Python Packaging Authority (PyPA) provides a suite of
tools that adhere to modern packaging standards (like PEP 517 and PEP 518). These tools include:
build (pypa/build): A build frontend for creating distribution packages.
installer (pypa/installer): A low-level library for installing Python packages from wheel distributions.
pip: The standard package installer for Python, which now handles installation based on these standards.
Why the Change:
The deprecation of direct setup.py invocation is part of a broader effort to
standardize and improve the Python packaging ecosystem. Modern tools and standards offer
better dependency management, build isolation, and a more robust and predictable build process.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# list of contents of the current folder
print("Contents of the current folder:")
import os
for item in os.listdir("."):
    print(f"- {item}")


with open("./requirements_dev.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="CaseStudy_FinanceRocks",
    version="0.1.0",
    author="Sharat Sharma",
    author_email="sharat.katz@example.com",
    description="Automated Exploratory Data Analysis tool for Parquet datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Data Scientists",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "finance-rocks-eda=CaseStudy_FinanceRocks.core:main",
        ],
    },
)
