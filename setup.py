#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "metaflow==2.4.3",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Jimmie Goode",
    author_email="jimmiegoode@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Lots of tabular models running in Metaflow",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="metaflow_tabular",
    name="metaflow_tabular",
    packages=find_packages(include=["metaflow_tabular", "metaflow_tabular.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jimgoo/metaflow-tabular",
    version="0.1.0",
    zip_safe=False,
)
