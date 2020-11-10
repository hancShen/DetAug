# coding=utf-8
from setuptools import setup, find_packages


setup(
    name='detaug',
    version='0.1.8',
    author='hcshen',
    author_email='hcshen729@gmail.com',
    maintainer='hcshen',
    packages=find_packages(),
    install_requires=['cython', 'numpy', 'opencv-python'],
    setup_requires=["cython>=0.28", "numpy>=1.14.0"],
)
