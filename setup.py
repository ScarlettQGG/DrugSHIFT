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


# Core runtime dependencies (hard imports across joint_embed.py + two_stage).
INSTALL_REQUIRES = [
    'numpy>=1.24,<2.0',
    'pandas>=2.0',
    'scikit-learn>=1.2',
    'scipy>=1.10',
    'torch>=2.0',
    'requests>=2.28',        # two_stage.direction_modules (Enrichr REST)
]

# Optional feature groups (lazy-imported; the code degrades gracefully without).
EXTRAS_REQUIRE = {
    'viz': ['matplotlib>=3.5'],                          # figures in two_stage.eval
    'leiden': ['python-igraph>=0.10', 'leidenalg>=0.9'],  # cluster-aware kNN (else KMeans)
    'test': ['pytest>=6.0'],
}
EXTRAS_REQUIRE['all'] = sorted({dep for deps in EXTRAS_REQUIRE.values() for dep in deps})


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
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
