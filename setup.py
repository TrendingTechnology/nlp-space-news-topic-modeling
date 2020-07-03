#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""The setup script."""


import os
from glob import glob

from setuptools import find_packages, setup

exclude_file_types = [
    ".yaml",
    ".yml",
    "*.egg-info",
    ".gitignore",
    "papermill_*.py",
]
packages_exclude = [
    os.path.basename(file_path)
    for file_paths in [
        glob(os.path.join(os.getcwd(), f"*{ft}")) for ft in exclude_file_types
    ]
    for file_path in file_paths
]

setup(
    name="nlp_space_news_topic_modeling",
    packages=find_packages(exclude=packages_exclude),
    use_scm_version=True,
    version="0.0.1",
    description="Python DataScience project for Natural Language Processing",
    author="Elstan DeSouza",
    url="https://github.com/edesz/nlp-space-news-topic-modeling",
    project_urls={
        "Repository": (
            "https://github.com/edesz/nlp-space-news-topic-modeling"
        ),
        "Issues": (
            "https://github.com/edesz/nlp-space-news-topic-modeling/issues"
        ),
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Topic :: Utilities",
        "Topic :: Software Development",
        "Intended Audience :: Developers",
        "Framework :: tox",
    ],
    python_requires=">=3.8",
)
