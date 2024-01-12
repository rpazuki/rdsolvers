# -*- coding: utf-8 -*-

from setuptools import setup


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='rdsolvers',
    version='0.1.0',
    ackages=['solvers'],
    description='Reaction-Diffusion PDE Solver',
    long_description=readme,
    author='Roozbeh H. Pazuki',
    author_email='rpazuki@gmail.com',
    url='https://github.com/rpazuki/rdsolvers',
    license=license,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
        "Environment :: Other Environment",
        "Topic :: Scientific/Engineering :: Physics",
    ]
)