"""
The two directory anchors everything else is built from.

`PACKAGE_ROOT`  this `reconstruction/` folder — all the reconstruction code and
                the NEURON models it loads.
`REPO_ROOT`     the repository above it, which also holds `simulate/` (the NEST
                model this reads its parameters and spike trains from), `RESULTS/`
                (shared with `simulate`, which writes the `.pic` files) and
                `data/`.

Keeping the two apart is what lets the reconstruction sit in its own folder while
still reading the professor's model and writing into the shared results tree.

WHY THIS MODULE CANNOT DO THE PATH SETUP ITSELF
-----------------------------------------------
`python reconstruction/ABR_reconstruction/main_abr.py` puts *that* directory on
`sys.path`, not `reconstruction/`, so `import recon_core` fails before this file
could ever be imported.  Every entry point therefore starts with the two lines
that fix it:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

This module exists so that everything *downstream* of that bootstrap shares one
definition of each root, instead of the eighteen ad-hoc copies it replaced.
Running via `python -m` or through `reconstruction/main.py` needs no bootstrap.
"""

import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)


def ensure_package_on_path():
    """Idempotently put `reconstruction/` on `sys.path`."""
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)
    return PACKAGE_ROOT
