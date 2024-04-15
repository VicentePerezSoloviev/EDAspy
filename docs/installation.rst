********************
Installing EDAspy
********************

Here you can find a detailed installation guide to use EDAspy.
Please, follow carefully the next steps in order to reduce the
number of errors during the installation. We have found issues installing PyBNesian library. Thus, following
steps include the installation of C++ and GPU tools.

We acknowledge all the members from Computational Intelligence Group (UPM) for
further discussions related to the installation procedure.

Ubuntu and Linux sub-systems
============================

EDAspy uses C++ and OpenCL in the backend to speed up certain
computations. Thus, some software is required to ensure everything
works. Note that, although setting up a Conda environment is usually
recommended, it is not mandatory. The following commands ensure that the C++ and OpenCL requirements are
satisfied.

.. code-block:: bash

    sudo apt update
    sudo apt install cmake
    sudo apt install g++
    sudo apt install opencl-headers
    sudo apt install ocl-icd-opencl-dev


After the previous steps you should be able to install EDAspy and its dependencies.

Installing from source
**********************

To install from source, we will download git to be able to download the
repository from GitHub.

.. code-block:: bash

    sudo apt install git

Now, clone the repository, install its dependencies, and install the package.

.. code-block:: bash

    git clone https://github.com/VicentePerezSoloviev/EDAspy.git
    cd EDAspy
    pip install -r requirements.txt
    python setup.py install


Installing directly from PyPi
******************************

Before installing EDAspy, ensure that all the dependencies are already installed in
your Python environment.

.. code-block:: bash

    pip install EDAspy


If no errors were raised, then the software is ready to be used. Otherwise, please
restart the process or raise an issue in the repository.

Windows
=======
Sometimes, in order to reduce possible inconvenient regarding Windows OS,
a Linux sub-system is installed (https://learn.microsoft.com/es-es/windows/wsl/install).
If this was the case, please go to [Ubuntu and Linux sub-systems](#ubuntu-and-linux-sub-systems) section.
Otherwise, please follow the next steps.

1. Download Visual Studio 2022 from https://visualstudio.microsoft.com/es/vs/

   1.1. Download the requirements for C++
3. Download Visual Studio Build Tools 2022.

.. code-block:: bash

    winget install "Visual Studio Build Tools 2022"


3. Download developer tools for GPU.

   3.1. For Nvidia, download Nvidia Toolkit (https://developer.nvidia.com/cuda-downloads)

   3.2. For Intel, download OneApi (https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)

5. Download OpenCL for windows. This guide explains the installation process: https://windowsreport.com/opencl-install-windows-11/

6. Install EDAspy

Installing from source
**********************

To install from source, we will download git to be able to download the
repository from GitHub.

.. code-block:: bash

    sudo apt install git

Now, clone the repository, install its dependencies, and install the package.

.. code-block:: bash

    git clone https://github.com/VicentePerezSoloviev/EDAspy.git
    cd EDAspy
    pip install -r requirements.txt
    python setup.py install


Installing directly from PyPi
******************************

Before installing EDAspy, ensure that all the dependencies are already installed in
your Python environment.

.. code-block:: bash

    pip install EDAspy


If no errors were raised, then the software is ready to be used. Otherwise, please
restart the process or raise an issue in the repository.

Installation issues
====================

Please refer to the installation discussion (https://github.com/VicentePerezSoloviev/EDAspy/discussions/18) section
to discuss further issues with the developer community.

1. If default installation for Linux
fails, there might be necessary to install GPU toolkits for Linux. Please,
visit https://developer.nvidia.com/cuda-downloads for Nvidia, and
https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html for
Intel.

2. If *undefined symbol _ZNK5arrow6Status8ToStringEv* error has been raised, please update the version
of pyarrow to a greater one, and reinstall EDAspy and respective dependencies.



