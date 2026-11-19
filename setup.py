"""Pure-Python wheel by default; optional native extension, no CMake/nvcc."""
import os
import sys
from setuptools import setup, Extension

extensions = []
if os.environ.get("DUPIRE_NATIVE", "0") == "1":
    import pybind11
    win = sys.platform == "win32"
    omp = os.environ.get("DUPIRE_OPENMP", "1") == "1"
    compile_args = ["/O2", "/std:c++17"] if win else ["-O3", "-std=c++17"]
    link_args = []
    if omp:
        compile_args += ["/openmp"] if win else ["-fopenmp"]
        link_args += [] if win else ["-fopenmp"]
    extensions = [Extension(
        "dupire_identify._native",
        ["src/dupire_identify/kernels/native.cpp"],
        include_dirs=[pybind11.get_include(), "src/dupire_identify/kernels"],
        language="c++", extra_compile_args=compile_args,
        extra_link_args=link_args,
    )]
setup(ext_modules=extensions)
