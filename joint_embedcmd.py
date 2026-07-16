#!/usr/bin/env python
"""Command-line entry point for the SEC-MS GNN joint-embedding step.

Installed on the PATH by ``setup.py`` (``scripts=['joint_embedcmd.py']``),
analogous to cellmaps' ``cellmaps_pipelinecmd.py``. It is a thin wrapper that
delegates to :func:`joint_embed.main`, so ``joint_embedcmd.py --help`` shows the
full ``joint_embed.py`` argument set.
"""
import sys

from joint_embed import main

if __name__ == '__main__':
    sys.exit(main())
