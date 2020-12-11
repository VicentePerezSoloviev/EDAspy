import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EDAspy",
    version="0.2.0",
    author="Vicente P. Soloviev",
    author_email="vicente.perez.soloviev@gmail.com",
    description="Estimation of Distribution Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    download_url="https://github.com/VicentePerezSoloviev/EDAspy/archive/0.2.1.tar.gz",
    url="https://github.com/VicentePerezSoloviev/EDAspy",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Operating System :: OS Independent",
    ],
    keywords=['EDA', 'estimation', 'bayesian', 'evolutionary', 'algorithm', 'optimization', 'time_series', 'feature',
              'selection'],
    python_requires='>=3.6',
    license="LGPLv2+",
)
