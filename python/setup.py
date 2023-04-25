from setuptools import setup, find_packages
import os

setup(name='gs',
      version='0.1',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      data_files=[('gs', ['../build/libgs.so'])])
