import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = '1.1.4'  # also defined in EDAspy/__init__.py and in conf.py (docs)

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
    download_url="https://github.com/VicentePerezSoloviev/EDAspy/archive/1.1.4.tar.gz",
    url="https://github.com/VicentePerezSoloviev/EDAspy",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=['EDA', 'estimation', 'bayesian', 'evolutionary', 'algorithm', 'optimization', 'time_series', 'feature',
              'selection', 'semiparametric', 'Gaussian'],
    python_requires='>=3.8, <3.11',
    setup_requires=["networkx", "pandas", "pgmpy", "pyarrow==9.0.0", "pybnesian==0.4.3", "scipy", "multiprocess",
                    "matplotlib", "numpy"],
    install_requires=["networkx", "pandas", "pgmpy", "pyarrow==9.0.0", "pybnesian==0.4.3", "scipy", "multiprocess",
                      "matplotlib", "numpy"],
    license="MIT",
    include_package_data=True,
    package_data={'': ['benchmarks/input_data/*.txt']}
)
