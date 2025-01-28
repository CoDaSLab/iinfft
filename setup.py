from setuptools import setup, find_packages, Extension
import numpy

# Define the C extension
c_extension = Extension(
    name="iinfft.core.sym_matrix",  # Module name (matches the path to the C code)
    sources=["iinfft/core/sym_matrix.c"],  # Path to the C source file
    include_dirs=[numpy.get_include()],  # Include Numpy's headers
)

# Setup configuration
setup(
    name="iinfft",  # Name of the package
    version="0.1.0",  # Version of the package
    description="A package for FFT-based image processing",
    author="Michael Sorochan Armstrong",
    author_email="mdarmstr@ugr.es",
    url="https://github.com/your_github_repo",  # Replace with your repo
    packages=find_packages(),  # Automatically find all Python packages
    ext_modules=[c_extension],  # Include the C extension
    package_data={
        "iinfft.data": ["*.mat", "*.npy", "*.csv"],  # Include data files
    },
    include_package_data=True,  # Ensure package data is included
    install_requires=[
        "numpy",  # List Python dependencies
        "scipy",
        "matplotlib",
        "nfft",
        "h5py",
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Specify the Python version compatibility
)
