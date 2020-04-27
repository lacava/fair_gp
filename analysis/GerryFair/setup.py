# from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name="gerryfair",
    version="1.0.1",
    install_requires=["numpy",
                "pandas",
                "scikit-learn",
                "matplotlib"
    ],
    packages = find_packages()
)
