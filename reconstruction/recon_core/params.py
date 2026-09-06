"""
THE parameter file for the LFP and ABR reconstruction pipelines.

Open this to see or change anything the reconstruction does.  Nothing downstream
declares a parameter of its own — before this file the same numbers lived in
about thirty places, each annotated with the source it had been copied from, and
several had silently drifted from it.

TWO KINDS OF PARAMETER
----------------------
1. **Mirrored from the NEST network.**  Population sizes, convergences, synaptic
   delays, reversal potentials, resting potentials and synaptic time constants
   all belong to `simulate/models/BrainstemModel/params.py`, the professor's
   model definition.  They are read from it LIVE via `recon_core.brainstem`, so
   editing that file changes the reconstruction too and the two can no longer
   disagree.  Never retype one of those numbers here.

2. **Reconstruction-only.**  Everything the NEST model has no opinion about: the
   NEURON timestep, morphology geometry, probe layout, ABR filter bands, and the
   NEURON synaptic conductances.

WHY THE CONDUCTANCES ARE NOT MIRRORED
-------------------------------------
`params.SYN_WEIGHTS` are NEST conductances in nS, tuned for the point-neuron
`iaf_cond_beta` model.  The reconstruction inserts `Exp2Syn` synapses (µS) onto
morphologically detailed cells with real input impedances, so its weights were
tuned independently and are NOT convertible.  They are declared here as
first-class reconstruction parameters.

Likewise `DT` and `TSTOP`: the NEST kernel's `resolution = 0.01 ms` is its own
integration step and says nothing about the NEURON timestep the reconstruction
needs, which is set by the ABR band (see `DT` below).

COORDINATE FRAME (head coordinates, µm)
    x  dorsoventral
    y  rostrocaudal
    z  mediolateral
Head-frame geometry — 4-sphere radii, electrodes, nucleus positions, model→head
rotations — lives in `recon_core.head_geometry`, re-exported at the end of this
file.
"""

import numpy as np

from recon_core import brainstem
from recon_core.signal_utils import srate_from_dt

# The live NEST parameter set.  Attribute access mirrors params.py exactly:
# BRAINSTEM.POP_NUM.n_MSOs, BRAINSTEM.E_L.MSO, BRAINSTEM.POP_CONV.SBCs2MSOs, ...
BRAINSTEM = brainstem.parameters()


# ===========================================================================
# Simulation (reconstruction-only)
# ===========================================================================
# 0.026 ms ≈ 38.5 kHz.  The ABR band reaches 3 kHz and the analysis resolves
# sub-millisecond wave latencies, so the sample rate must sit far above both.
DT = 0.026        # ms   NEURON timestep, shared by every pipeline
TSTOP = 50.0      # ms   simulated duration
SRATE = srate_from_dt(DT)   # Hz

# Post-onset settling window discarded from the LSO spiking dipole: the axon is
# initialised out of equilibrium and its relaxation is not a physiological signal.
SETTLE_MS = 3.0


# ===========================================================================
# Scalp recording
# ===========================================================================
ELECTRODES = ('Cz', 'M1', 'M2')       # vertex + both mastoids
DERIVATIONS = ('Cz-M1', 'Cz-M2', 'Cz-avg')

# Two band-passes are in use, each matching the study being reproduced.
BAND_TOLNAI = (100., 1500.)     # Tolnai & Klump 2020 (MSO / LSO figures)
BAND_CLINICAL = (150., 3000.)   # clinical BAEP standard; Curio & Weigel 1990
BAND_DEFAULT = BAND_CLINICAL


# ===========================================================================
# Probe (near-field LFP)
# ===========================================================================
N_CH = 16                 # linear probe channels, spanning z
SIGMA_EXTRACELLULAR = 0.3   # S/m


def probe_z(half_span_um):
    """Channel depths for a linear probe spanning ±`half_span_um` along z."""
    return np.linspace(-half_span_um, half_span_um, N_CH)


# ===========================================================================
# MSO — medial superior olive (binaural coincidence detector)
# ===========================================================================
N_MSO_TOTAL = BRAINSTEM.POP_NUM.n_MSOs
MSO_V_INIT = BRAINSTEM.E_L.MSO

# Elliptic-cylinder insertion volume.
MSO_RADIUS_X = 255.0     # µm  tonotopic half-axis
MSO_RADIUS_Y = 2845.0    # µm  rostrocaudal half-axis
MSO_PROBE_HALF_SPAN = 400.0   # µm

# Bipolar morphology: contralateral SBC → medial dendrite, ipsilateral SBC →
# lateral dendrite, inhibition onto the soma (Cant & Hyson 1992; Joris 1998).
MSO_LAYERS = [
    [10., 160.],     # medial dendrite
    [-160., -10.],   # lateral dendrite
    [-10., 10.],     # soma
]

# Input band: the MSO is restricted to low CFs, mapped onto its tonotopic index
# through the ERB scale (see `tonotopic_index`).
MSO_FREQ_MIN = 200.0     # Hz
MSO_FREQ_MAX = 4000.0    # Hz

MSO_SYNAPSES = {
    'SBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.MSO,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.MSO,
        'e': BRAINSTEM.EXC_REV.MSO,
        'weight': 0.055,      # µS  (reconstruction-only, see module docstring)
    },
    'MNTBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_IN_RISE.MSO,
        'tau2': BRAINSTEM.TAUS_IN_DECAY.MSO,
        'e': BRAINSTEM.INH_REV.MSO,
        'weight': 0.025,      # µS
    },
    'LNTBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_IN_RISE.MSO,
        'tau2': BRAINSTEM.TAUS_IN_DECAY.MSO,
        'e': BRAINSTEM.INH_REV.MSO,
        'weight': 0.025,      # µS
    },
}

# Synapses per presynaptic population per layer, matching MSO_LAYERS.
# Column order: SBC_contra, SBC_ipsi, MNTBC_ipsi, LNTBC_ipsi.
#
# One table for both pipelines.  They previously disagreed with each other and
# with the NEST model (LFP: 6 excitatory synapses per dendrite, ABR: 3, both: 2
# MNTBC); all three now come from POP_CONV.  See RESULTS/regression/param_snap.md.
MSO_CONVERGENCE = [
    [BRAINSTEM.POP_CONV.SBCs2MSOs, 0, 0, 0],    # medial dendrite  <- contra SBC
    [0, BRAINSTEM.POP_CONV.SBCs2MSOs, 0, 0],    # lateral dendrite <- ipsi SBC
    [0, 0, BRAINSTEM.POP_CONV.MNTBCs2MSOs,      # soma             <- inhibition
     BRAINSTEM.POP_CONV.LNTBCs2MSOs],
]

MSO_DELAYS = [BRAINSTEM.SYN_DELAYS.SBCs2MSOcontra,
              BRAINSTEM.SYN_DELAYS.SBCs2MSOipsi,
              BRAINSTEM.SYN_DELAYS.MNTBCs2MSO,
              BRAINSTEM.SYN_DELAYS.LNTBCs2MSO]

# hybridLFPy bookkeeping: `J_yX` and `tau_yX` are recorded in the population
# metadata but never reach the cell — insert_all_synapses overrides both from the
# *_SYNAPSES tables above.  They had drifted (the LFP and ABR pipelines passed
# different J_yX for the same synapses); now both are derived from the one
# synapse table, so the metadata describes what was actually simulated.
def _bookkeeping(synapses, order):
    """(J_yX, tau_yX) for hybridLFPy, read off the real synapse table."""
    return ([synapses[pop]['weight'] for pop in order],
            [synapses[pop]['tau2'] for pop in order])


MSO_J_YX, MSO_TAU_YX = _bookkeeping(
    MSO_SYNAPSES, ['SBC', 'SBC', 'MNTBC', 'LNTBC'])


# ===========================================================================
# LSO — lateral superior olive (ILD coder)
# ===========================================================================
N_LSO_TOTAL = BRAINSTEM.POP_NUM.n_LSOs
LSO_V_INIT = BRAINSTEM.E_L.LSO

# Human dimensions, anatomically reoriented: the long axis runs rostrocaudally
# and tonotopy runs mediolaterally (lateral = low CF).
LSO_HALF_HEIGHT_Y = 1400.0   # µm  rostrocaudal half-height (2.8 mm span)
LSO_RADIUS_X = 400.0         # µm  dorsoventral semi-axis
LSO_RADIUS_Z = 600.0         # µm  mediolateral / tonotopic semi-axis
LSO_PROBE_HALF_SPAN = 500.0  # µm  brackets the ±413.5 µm dendrite tips

# One all-encompassing layer: hybridLFPy assigns synapses by absolute z, but the
# reoriented nucleus spreads somas along z, so a narrow band would miss most
# cells.  Placement is by section name instead (see LSOPopulation).
LSO_LAYERS = [[-1.0e4, 1.0e4]]

LSO_SYNAPSES = {
    'SBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.LSO,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.LSO,
        'e': BRAINSTEM.EXC_REV.LSO,
        'weight': 0.040,      # µS
    },
    'MNTBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_IN_RISE.LSO,
        'tau2': BRAINSTEM.TAUS_IN_DECAY.LSO,
        'e': BRAINSTEM.INH_REV.LSO,
        'weight': 0.020,      # µS
    },
}
LSO_CONVERGENCE = [[BRAINSTEM.POP_CONV.SBCs2LSOs, BRAINSTEM.POP_CONV.MNTBCs2LSOs]]
LSO_DELAYS = [BRAINSTEM.SYN_DELAYS.SBCs2LSO, BRAINSTEM.SYN_DELAYS.MNTBCs2LSO]
LSO_J_YX, LSO_TAU_YX = _bookkeeping(LSO_SYNAPSES, ['SBC', 'MNTBC'])

# Spiking drive: one suprathreshold synapse on the AIS makes the cell fire once
# per NEST output spike, seeding the travelling wave up the ascending-LL axon.
LSO_SPIKING_SYNAPSES = {
    'LSO': {
        'syntype': 'Exp2Syn',
        'tau1': 0.1,          # ms  fast rise
        'tau2': 0.2,          # ms  fast decay → exactly one AP per input
        'e': BRAINSTEM.EXC_REV.LSO,
        'weight': 0.30,       # µS  suprathreshold (single-cell rheobase ~6 nA)
    },
}
LSO_SPIKING_CONVERGENCE = [[1]]
LSO_SPIKING_DELAYS = [0.05]   # ms  nominal; the drive is the cell's own output
LSO_SPIKING_J_YX, LSO_SPIKING_TAU_YX = _bookkeeping(
    LSO_SPIKING_SYNAPSES, ['LSO'])


# ===========================================================================
# AVCN — bushy cells (globular and spherical)
# ===========================================================================
# Both are driven by ANF endbulbs of Held and share one Population class; they
# differ in morphology, channel densities and endbulb count.
GBC_V_INIT = -65.0    # mV  leak reversal of the cnmodel XM13_nacncoop decoration,
SBC_V_INIT = -65.0    #     not a NEST resting potential — hence not mirrored

N_GBC_TOTAL = BRAINSTEM.POP_NUM.n_GBCs
N_SBC_TOTAL = BRAINSTEM.POP_NUM.n_SBCs
GBC_ENDBULBS = BRAINSTEM.POP_CONV.ANFs2GBCs   # 20 modified endbulbs
SBC_ENDBULBS = BRAINSTEM.POP_CONV.ANFs2SBCs   # 3 large axosomatic endbulbs

AVCN_RADIUS_X = 400.0    # µm  tonotopic half-axis
GBC_RADIUS_Y = 600.0     # µm  rostrocaudal half-axis (caudal AVCN)
SBC_RADIUS_Y = 875.0     # µm  the spherical-cell area spans the rostral ~1.75 mm
AVCN_PROBE_HALF_SPAN = 300.0   # µm  (bushy cells are smaller than MSO/LSO)

AVCN_LAYERS = [[-100.0, 100.0]]   # placement is by section name, not by depth

# The axon exits ventromedially in MODEL coordinates for both sides; laterality
# is carried by the input population, not by flipping the geometry.  Aligning
# every axon this way makes the axial currents summate into a large-scale dipole.
AVCN_AXON_TARGET = np.array([-1., 0., -1.]) / np.sqrt(2.)

# Few, large, axosomatic endbulbs on the SBC → mostly somatic placement.
SBC_ENDBULB_WEIGHTS = {'soma': 0.85, 'primarydendrite': 0.15}

GBC_SYNAPSES = {
    'ANF': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.GBC,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.GBC,
        'e': BRAINSTEM.EXC_REV.GBC,
        'weight': 0.005,      # µS  one of 20 endbulbs
    },
}
SBC_SYNAPSES = {
    'ANF': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.SBC,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.SBC,
        'e': BRAINSTEM.EXC_REV.SBC,
        'weight': 0.030,      # µS  one of 3 large endbulbs
    },
}
GBC_CONVERGENCE = [[GBC_ENDBULBS]]
SBC_CONVERGENCE = [[SBC_ENDBULBS]]
GBC_DELAYS = [BRAINSTEM.SYN_DELAYS.ANFs2GBCs]
SBC_DELAYS = [BRAINSTEM.SYN_DELAYS.ANFs2SBCs]
SYN_DELAY_SCALE = [None]      # no jitter on the endbulb delay


# ===========================================================================
# MNTB — medial nucleus of the trapezoid body
# ===========================================================================
N_MNTB_TOTAL = BRAINSTEM.POP_NUM.n_MNTBCs
MNTB_V_INIT = BRAINSTEM.E_L.MNTBC

MNTB_RADIUS_X = 200.0     # µm  tonotopic half-axis
MNTB_RADIUS_Y = 2000.0    # µm  rostrocaudal half-axis (~4 mm, coextensive with MSO)
MNTB_PROBE_HALF_SPAN = 300.0   # µm  brackets the ±67 µm dendrite tips

# Must match the section extents in models/mntb/mntb_model_active.hoc.
MNTB_LAYERS = [
    [10.0, 67.0],     # dend_A (+z)
    [-67.0, -10.0],   # dend_B (−z)
    [-10.0, 10.0],    # soma — the axosomatic calyx lands here
]

# The calyx of Held is the fastest synapse in the model.
MNTB_SYNAPSES = {
    'GBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.MNTBC,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.MNTBC,
        'e': BRAINSTEM.EXC_REV.MNTBC,
        'weight': 0.050,      # µS  suprathreshold calyx
    },
}
# The presynaptic terminal itself, driven suprathreshold to generate the
# extracellular prespike; co-located with the principal cell.
CALYX_SYNAPSES = {
    'GBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.MNTBC,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.MNTBC,
        'e': BRAINSTEM.EXC_REV.MNTBC,
        'weight': 0.150,      # µS  fires the terminal once per input
    },
}
MNTB_CONVERGENCE = [[0], [0], [BRAINSTEM.POP_CONV.GBCs2MNTBCs]]   # soma only
MNTB_DELAYS = [BRAINSTEM.SYN_DELAYS.GBCs2MNTBCs]
MNTB_J_YX, MNTB_TAU_YX = _bookkeeping(MNTB_SYNAPSES, ['GBC'])
CALYX_J_YX, CALYX_TAU_YX = _bookkeeping(CALYX_SYNAPSES, ['GBC'])
# The prespike LEADS the postsynaptic sink; the GBC→MNTB conduction delay is
# already in the spike train, so the terminal fires essentially on arrival.
CALYX_DELAYS = [0.05]


# ===========================================================================
# Cochlear tonotopy
# ===========================================================================
# The cochlear model's CF range, from simulate/utils/cochlea_utils.py.
CF_MIN_HZ = 125.0
CF_MAX_HZ = 20000.0
N_ANF_TOTAL = BRAINSTEM.n_ANFs


def _erb_number(freq_hz):
    """Glasberg & Moore (1990) ERB-rate scale."""
    return 21.3 * np.log10(1.0 + freq_hz / 229.0)


def tonotopic_index(freq_hz, n_total):
    """Characteristic frequency → index into a tonotopically ordered population.

    Populations are laid out linearly on the ERB scale, matching how the cochlear
    model distributes its characteristic frequencies.
    """
    span = _erb_number(CF_MAX_HZ) - _erb_number(CF_MIN_HZ)
    frac = (_erb_number(freq_hz) - _erb_number(CF_MIN_HZ)) / span
    return int(round(float(np.clip(frac, 0.0, 1.0)) * (n_total - 1)))


# ===========================================================================
# Head model (re-exported; defined in recon_core/head_geometry.py)
# ===========================================================================
from recon_core.head_geometry import (          # noqa: E402,F401
    FOUR_SPHERE_RADII, FOUR_SPHERE_SIGMAS, ELECTRODE_POS,
    NUCLEUS_POS_UM, MSO_POS_UM, LSO_POS_UM, AVCN_POS_UM, SBC_POS_UM, MNTB_POS_UM,
    ROTATION_MSO, ROTATION_LSO, ROTATION_AVCN, ROTATION_MNTB,
)
