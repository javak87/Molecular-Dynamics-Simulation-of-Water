# 26.02.2021
# Author: Florian Anders

from setuptools import setup

setup\
    (
        name='molecular_simulation_water',
        version='0.1',
        author='Javad Kasravi, Wu Hongmiao, Cenk Üstündag, Muhtasim Fuad, Dennis Mattuat, Florian Anders',
        author_email='florian.anders@fu-berlin.de',
        packages=['molecular_simulation_python_files'],
        description='A test of packaging the project',
        install_requires=
        [
            "numpy",
            "scipy",
            "matplotlib",
            "h5py",
            "pycallgraph",
            "pandas",
            "typing",
            "transformations",
        ],
    )
