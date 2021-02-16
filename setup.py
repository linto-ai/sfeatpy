#!/usr/bin/env python3
from setuptools import setup, find_packages
import json

with open("manifest.json", 'r') as f:
    manifest = json.load(f)
    VERSION = manifest["version"]
    NAME = manifest["name"]

setup(
    name=NAME,
    version=VERSION,
    include_package_data=True, 
    packages=find_packages(),
    install_requires=[
        'numpy>=1.16',
        'scipy>=1.6.0',
    ],
    author="Rudy Baraglia",
    author_email="rbaraglia@linagora.com",
    description="Functions to extract signal speech parameters",
    long_description="Details on github",
    long_description_content_type="text/markdown",
    licence="AGPLv3+",
    keywords="speech features mfcc lmfe",
    url="https://github.com/linto-ai/sfeatpy.git",
    py_modules=["sfeatpy"],
    project_urls={
        "github" : "https://github.com/linto-ai/sfeatpy.git",
        "pypi" : "https://pypi.org/project/sfeatpy/"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ]
)