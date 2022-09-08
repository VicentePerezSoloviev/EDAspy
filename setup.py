import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EDAspy",
    version="1.0.0",
    author="Vicente P. Soloviev",
    author_email="vicente.perez.soloviev@gmail.com",
    description="Estimation of Distribution Algorithms for optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    download_url="https://github.com/VicentePerezSoloviev/EDAspy/archive/1.0.0.tar.gz",
    url="https://github.com/VicentePerezSoloviev/EDAspy",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Operating System :: OS Independent",
    ],
    keywords=['EDA', 'estimation', 'bayesian', 'evolutionary', 'algorithm', 'optimization', 'time_series', 'feature',
              'selection', 'semiparametric'],
    python_requires='>=3.0',
    license="bsd-3-clause",
    install_requires=["pandas>=1.2.0", "numpy>1.15.0", "pybnesian>=0.3.4"]
)
