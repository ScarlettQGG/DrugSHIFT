#!/usr/bin/env python
"""setup.py for DrugSHIFT — perturbation-aware multimodal cell-mapping.

Packages the two-stage model (``two_stage``) plus the standalone SEC-MS GNN
joint-embedding step (``joint_embed.py`` at the repo root), and installs the
latter on the PATH as a command-line script (``joint_embedcmd.py``), following
the cellmaps convention of listing a ``*cmd.py`` in ``scripts=``.
"""
import os

from setuptools import setup, find_packages

_here = os.path.abspath(os.path.dirname(__file__))


def _read(fname):
    path = os.path.join(_here, fname)
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as fh:
            return fh.read()
    return ''


def _requirements():
    reqs = []
    for line in _read('requirements.txt').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            reqs.append(line)
    return reqs


setup(
    name='drugshift',
    version='0.1.0',
    description='Perturbation-aware multimodal cell-mapping (two-stage model + SEC-MS GNN joint embedding)',
    long_description=_read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/ScarlettQGG/DrugSHIFT',
    license='MIT',
    packages=find_packages(include=['two_stage', 'two_stage.*']),
    py_modules=['joint_embed'],
    scripts=[
        'joint_embedcmd.py',
    ],
    install_requires=_requirements(),
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
