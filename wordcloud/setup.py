"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='word_cloud',
    ext_modules=cythonize("*.pyx"),
    package_dir={'word_cloud': '.'}
)
"""

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy.distutils.misc_util

c_ext = [
    Extension("query_integral_image", ["query_integral_image.pyx"])
]


setup(
    name="word_cloud",
    cmdclass={"build_ext": build_ext},
    ext_modules=c_ext,
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)

# python setup.py build_ext --inplace
