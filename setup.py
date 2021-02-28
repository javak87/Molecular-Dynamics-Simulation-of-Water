# 26.02.2021
# Author: Florian Anders

from setuptools import setup

setup\
    (
        name='molecular_simulation_water_semester_project',
        version='1.0',
        author='Javad Kasravi, Hongmiao Wu, Cenk Üstündag, Muhtasim Fuad, Dennis Mattutat, Florian Anders',
        author_email='javad.kasravi@fu-berlin.de, hongmiao.wu@fu-berlin.de, cenku95@zedat.fu-berlin.de, muhtasif71@zedat.fu-berlin.de, florian.anders@fu-berlin.de',
        packages=['molecular_simulation_water'],
        description='Package containing python files for the molecular dynamics simulation of water',
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