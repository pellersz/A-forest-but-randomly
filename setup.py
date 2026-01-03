from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Random forest implementation',
    ext_modules=cythonize("random_forest.pyx"),
    include_dirs=[numpy.get_include()]
)
