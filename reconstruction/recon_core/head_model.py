"""
Projection of current-dipole moments onto the scalp through the 4-sphere model.

This is the one place the volume conductor is built.  Every ABR generator — MSO,
LSO, the two AVCN bushy-cell types, the MNTB principal cell and its calyx — is a
dipole at its own anatomical position, and the scalp potential is their exact
linear superposition:

    Phi_total(r_e) = sum_g  L(r_e, r_g) . p_g(t)

so a nucleus can be simulated once, its head-frame dipole stored, and composites
assembled later without re-running NEURON (see `ABR_reconstruction/main_abr_full`).

UNITS — all lfpykit-native, no conversion anywhere but the final mV -> uV:
    positions / electrodes      um
    dipole moment `p`           nA.um
    `get_dipole_potential`      mV
    saved / plotted ABR         uV
"""

from collections import defaultdict

import numpy as np

from recon_core import head_geometry as hg
from recon_core.signal_utils import bandpass


def volume_conductor(electrode_names):
    """`FourSphereVolumeConductor` for the given electrodes, plus their positions."""
    from lfpykit.eegmegcalc import FourSphereVolumeConductor

    r_elec = np.stack([hg.ELECTRODE_POS[e] for e in electrode_names])
    fsc = FourSphereVolumeConductor(
        r_electrodes=r_elec,
        radii=hg.FOUR_SPHERE_RADII,
        sigmas=hg.FOUR_SPHERE_SIGMAS,
    )
    return fsc, r_elec


def project_sources(sources, electrode_names):
    """Sum unfiltered scalp potentials, in mV, grouped by source label.

    `sources` is an iterable of `(label, p_head (3, T), r_dipole (3,))`.  Sources
    sharing a label are summed — that is how the L and R populations of one
    generator collapse into a single trace.
    """
    fsc, _ = volume_conductor(electrode_names)
    V_mV = defaultdict(float)
    for label, p_head, r_dipole in sources:
        V_mV[label] = V_mV[label] + fsc.get_dipole_potential(
            np.asarray(p_head, dtype=float), np.asarray(r_dipole, dtype=float))
    return dict(V_mV)


def superpose_sources(sources, electrode_names, srate, lo=150., hi=3000.):
    """Project, band-pass and superpose — the shared core of every ABR script.

    `sources` entries are `(label, p_head, r_dipole)`.  Returns
    `{label: (n_electrodes, T) uV}` with an extra `'composite'` key holding the
    sum of every label.

    Both the per-label traces and the composite are filtered, and they stay
    mutually consistent because the filter is linear: summing filtered traces
    equals filtering their sum.  Callers therefore group them further
    (per-nucleus totals, monaural differences) without re-filtering.
    """
    V_mV = project_sources(sources, electrode_names)

    out, total = {}, 0.0
    for label, v in V_mV.items():
        total = total + v
        out[label] = bandpass(v * 1e3, fs=srate, lo=lo, hi=hi)   # mV -> uV
    out['composite'] = bandpass(np.asarray(total) * 1e3, fs=srate, lo=lo, hi=hi)
    return out


def project_by_side(p_head_by_side, position_by_side, electrode_names, srate,
                    lo=150., hi=3000.):
    """Convenience wrapper for a single generator simulated on one or both sides.

    Each side's dipole is projected from ITS OWN position — the sides are mirrored
    about the midline, so this is what makes a lateralised generator produce
    different potentials at the left and right mastoids.

    Returns `(V_uV (n_electrodes, T), srate)`.
    """
    sources = [('generator', p_head, position_by_side[side])
               for side, p_head in p_head_by_side.items()]
    return superpose_sources(sources, electrode_names, srate, lo=lo, hi=hi)['composite'], srate


def rotate_to_head(p_model, rotation):
    """Model-frame dipole (3, T) -> head frame, via a nucleus's 3x3 rotation."""
    return np.asarray(rotation) @ np.asarray(p_model, dtype=float)
