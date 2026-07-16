"""Run HiDeF multi-resolution community detection over a multiplex of PPI
networks.

HiDeF (`hidef_finder.py`, installed with the ``hidef`` package) scans a range of
Leiden resolutions, collects communities that recur across resolutions, and
weaves them into a containment hierarchy. Two parameters govern the scan:

* ``maxres`` -- the maximum resolution explored; higher values expose finer
  communities.
* ``k`` (the persistence / chi threshold) -- the minimum number of resolutions a
  community must survive to be reported. Its per-community value is recorded as
  the ``persistence`` field of the output and, at a low ``k``, spans the whole
  persistence spectrum -- the signal the adaptive fusion step relies on.

The command is invoked once with every cutoff edgelist supplied together, so
communities are detected jointly across network densities. Output is written as
``<prefix>.nodes`` and ``<prefix>.edges``; member ids are the input node ids
(co-embedding row indices).
"""

import os
import subprocess
import sys


def _hidef_finder_path():
    """Locate the ``hidef_finder.py`` console script.

    Preference is given to the copy installed alongside the active Python
    interpreter; otherwise the bare command name is returned and resolved via
    ``PATH``.
    """
    candidate = os.path.join(os.path.dirname(sys.executable), "hidef_finder.py")
    return candidate if os.path.exists(candidate) else "hidef_finder.py"


def run_hidef(
    edgelist_files,
    outprefix,
    k=5,
    maxres=80,
    algorithm="leiden",
    numthreads=1,
):
    """Run HiDeF and return the paths of the raw node/edge files.

    :param edgelist_files: list of PPI edgelist file paths (the multiplex).
    :param outprefix: output path prefix; ``<outprefix>.nodes`` and
        ``<outprefix>.edges`` are written.
    :param k: persistence threshold (chi). A low value (5) is used so the
        persistence column spans the full spectrum for the fusion step.
    :param maxres: maximum Leiden resolution to scan.
    :param algorithm: community-detection algorithm passed to HiDeF.
    :param numthreads: worker threads for the resolution scan.
    :returns: tuple ``(nodes_path, edges_path)``.
    :raises RuntimeError: if HiDeF exits non-zero or produces no node file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(outprefix)), exist_ok=True)
    command = [
        sys.executable,
        _hidef_finder_path(),
        "--g",
        *edgelist_files,
        "--o", outprefix,
        "--alg", algorithm,
        "--maxres", str(maxres),
        "--k", str(k),
        "--numthreads", str(numthreads),
        "--skipgml",
    ]

    # Pin BLAS/OpenMP threading so the run is single-threaded and reproducible.
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "hidef_finder.py failed (exit "
            f"{result.returncode}):\n{result.stdout}\n{result.stderr}"
        )

    nodes_path = outprefix + ".nodes"
    edges_path = outprefix + ".edges"
    if not os.path.exists(nodes_path):
        raise RuntimeError(f"HiDeF produced no node file at {nodes_path}")
    return nodes_path, edges_path
