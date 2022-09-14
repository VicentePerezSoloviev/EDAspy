import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = '1.0.0'  # also defined in EDAspy/__init__.py

setuptools.setup(
    name="EDAspy",
    version=__version__,
    author="Vicente P. Soloviev",
    author_email="vicente.perez.soloviev@gmail.com",
    description="EDAspy is a Python package that implements Estimation of Distribution Algorithms. EDAspy allows to"
                "either use already existing implementations or customize the EDAs baseline easily building it by"
                "modules so new research can be easily developed. It also has several benchmarks for comparisons.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    download_url="https://github.com/VicentePerezSoloviev/EDAspy/archive/1.0.0.tar.gz",
    url="https://github.com/VicentePerezSoloviev/EDAspy",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ],
    keywords=['EDA', 'estimation', 'bayesian', 'evolutionary', 'algorithm', 'optimization', 'time_series', 'feature',
              'selection', 'semiparametric', 'Gaussian'],
    python_requires='>=3.0',
    license="bsd-3-clause",
    install_requires=["pandas>=1.2.0", "numpy>1.15.0", "pybnesian>=0.3.4"]
)
