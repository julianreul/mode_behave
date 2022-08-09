# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:43:38 2020

@author: j.reul
"""

from setuptools import setup

setup(name='mode_behave_public',
      version='0.1',
      description='Estimation of Mixed Logit Models',
      author='Julian Reul',
      author_email='j.reul@fz-juelich.de',
      license='MIT',
      packages=['mode_behave_public'],
      install_requires=[
          'pandas>=1.3.1',
          'numpy>=1.19.1',
          'scipy>=1.5.2',
          'matplotlib>=3.3.1',
          'gputil>=1.4.0'
      ],
      include_package_data=True,
      zip_safe=False)