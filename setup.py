#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup

# get long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# What packages are required for this module to be executed?
REQUIRED = ['gym==0.23.0', 'pygame==2.1.2']

# Where the magic happens:
setup(
    name='AI_agents',
    version='0.0.0',
    description='Various AI agents',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Sarah Keren',
    author_email='sarahk@technion.ac.il',
    python_requires='>=3.6.0',
    url='https://github.com/sarah-keren/AI_agents',
    packages=find_packages(exclude=["Tests"]),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
