"""
Shared framework for the LFP and ABR reconstruction pipelines.

`LFP_reconstruction/` (near-field probe potentials) and `ABR_reconstruction/`
(far-field scalp potentials) are two views of the same simulated populations, so
they share their parameters, their spike/geometry bookkeeping, their HDF5 layout
and their signal processing.  All of that lives here; the two pipeline packages
hold only what is genuinely specific to their own view.

Import as a package from the repository root:

    from recon_core import params, paths, signal_utils

Entry-point scripts are still runnable directly (`python ABR_reconstruction/
main_abr.py ...`); they put the repository root on `sys.path` first — see
`recon_core.bootstrap`.
"""
