# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from setuptools import setup
from opencood.version import __version__

setup(
    name='OpenCOOD',
    version=__version__,
    license='MIT',
    author='Yijie CHEN',
    description='An opensource pytorch framework for collaborative semantic segmentation and forecasting, built upon OpenCOOD',
)
