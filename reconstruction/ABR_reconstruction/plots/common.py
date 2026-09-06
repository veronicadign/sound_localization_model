"""
Shared helpers for the ABR figure scripts.

Loaders, the binaural-interaction extraction, the output-directory templates and
the digitised reference values from the papers being reproduced all used to be
copied into each figure script — and had drifted: the Tolnai Fig-4 z-scores
existed in three files with three different values for the same point.
"""

import os

import numpy as np

from recon_core import head_geometry as hg
from recon_core import io_utils, params as P, paths
# derive/onset_peak/zscore are re-exported for the figure scripts.
from recon_core.signal_utils import derive, onset_peak, zscore   # noqa: F401

ELECTRODES = list(P.ELECTRODES)
DEFAULT_WINDOW = (1.0, 15.0)      # ms, the onset-peak search window
DEFAULT_STIM_LABEL = 'Click 70 dB'


# ---------------------------------------------------------------------------
# Locating results
# ---------------------------------------------------------------------------
def condition_label(value, sweep):
    """Sweep value -> the directory label the producer used.

    `sweep` is 'itd' (µs), 'ild' (dB) or 'angle' (degrees).
    """
    if sweep == 'itd':
        return f'itd{value:g}us'
    if sweep == 'ild':
        return f'ild{value:g}dB'
    return f'angle{int(value)}'


def abr_dir(stem, cond_label, side='both', prefix=None, suffix=''):
    """The ABR output directory a producer wrote, rebuilt from its parts."""
    return paths.output_dir_for('abr', stem, cond_label, side, prefix=prefix,
                                suffix=suffix)


def lfp_dir(stem, cond_label, side='L', prefix=None, suffix=''):
    """The near-field LFP output directory a producer wrote."""
    return paths.output_dir_for('lfp', stem, cond_label, side, prefix=prefix,
                                suffix=suffix)


# ---------------------------------------------------------------------------
# Loading traces
# ---------------------------------------------------------------------------
def load_trace(directory, key='data', filename='ABR.h5'):
    """`(traces, electrode_names, srate)` from one ABR file, or None if absent.

    MSO and LSO store a single `data` array; the multi-generator nuclei store one
    array per generator plus `composite`, so the key varies by producer.
    """
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        return None
    traces, names, srate = io_utils.read_named_traces(path)
    if key not in traces:
        return None
    return traces[key], names, srate


def load_derivation(directory, derivation='Cz-M1', key='data', filename='ABR.h5'):
    """One scalp derivation from an ABR file -> `(trace, t_ms)`, or None."""
    loaded = load_trace(directory, key=key, filename=filename)
    if loaded is None:
        return None
    V, names, srate = loaded
    trace, _label = derive(V, names, derivation)
    return trace, np.arange(trace.size) / srate * 1e3


# ---------------------------------------------------------------------------
# Binaural interaction
# ---------------------------------------------------------------------------
def binaural_interaction(binaural, left, right):
    """`BI = RL - (L + R)`, truncated to the shortest trace.

    Every monaural generator contributes equally to both sides of the
    subtraction, so what survives is only what the binaural nuclei added.
    """
    n = min(len(binaural), len(left), len(right))
    return binaural[:n] - (left[:n] + right[:n])


def bic_peak(binaural, left, right, t_ms, window=DEFAULT_WINDOW):
    """Binaural-interaction component -> `(latency_ms, amplitude_µV)`."""
    bi = binaural_interaction(binaural, left, right)
    return onset_peak(bi, t_ms[:len(bi)], window=window)


# ---------------------------------------------------------------------------
# Head-model drawing (shared by every position figure)
# ---------------------------------------------------------------------------
SHELL_RADII_MM = [r * 1e-3 for r in hg.FOUR_SPHERE_RADII]
SHELL_LABELS = ['brain', 'CSF', 'skull', 'scalp']
SHELL_COLOURS = ['#d4e6f1', '#abebc6', '#f9e79f', '#f5cba7']
SHELL_ALPHAS = [0.25, 0.25, 0.25, 0.20]

ELECTRODE_POS_MM = {k: hg.ELECTRODE_POS[k] * 1e-3 for k in ('Cz', 'A1', 'A2')}
NUCLEUS_POS_MM = {name: {side: pos[side] * 1e-3 for side in ('L', 'R')}
                  for name, pos in hg.NUCLEUS_POS_UM.items()}


def on_scalp_mm(direction):
    """Unit direction -> the point where it meets the scalp, in mm."""
    v = np.asarray(direction, dtype=float)
    return v / np.linalg.norm(v) * (SHELL_RADII_MM[-1])


# ---------------------------------------------------------------------------
# Digitised reference data — Tolnai & Klump (2020) Figure 4
# ---------------------------------------------------------------------------
# z-scores read BY EYE off the rendered page (page 8), so approximate.
#
# ONE copy.  These lived in three scripts and had already drifted apart: the
# complete set below comes from the Fig-4 digitisation script, the purpose-built
# digitisation of the whole figure; the two paper-vs-model scripts carried
# partial re-digitisations that disagreed with it by up to 0.15 z
# (abr/lso amplitude at 0 µs: -1.15 here vs -1.10 there; abr/lso latency at
# 2000 µs: 1.15 vs 1.20; lfp/lso latency at 0 µs: -0.55 vs -0.60).  The
# differences are well inside the accuracy of reading points off a printed
# figure, and the complete set is the one kept.
PAPER_ITD_US = [0, 125, 500, 1000, 2000]

PAPER_FIG4_ZSCORES = {
    ('abr', 'lso'): {'amplitude': [-1.15, -1.00, 0.15, 0.30, 1.15],
                     'latency': [-0.80, -0.75, -0.25, 0.25, 1.15]},
    ('abr', 'mso'): {'amplitude': [-1.15, -1.05, 0.35, 0.35, 1.15],
                     'latency': [-0.80, -0.75, -0.10, 0.20, 0.85]},
    ('lfp', 'lso'): {'amplitude': [-0.50, -0.40, -0.45, 0.05, 1.30],
                     'latency': [-0.55, -0.60, -0.25, 0.25, 0.95]},
    ('lfp', 'mso'): {'amplitude': [-0.35, -0.40, -0.65, -0.50, 1.40],
                     'latency': [-0.85, -0.75, -0.35, -0.05, 1.40]},
}

# Paper styling, matching the published figure: grey diamonds for the ABR,
# lime triangles for the LFP.
PAPER_STYLE = {'abr': dict(marker='D', color='0.4', label='paper ABR'),
               'lfp': dict(marker='^', color='#8DC63F', label='paper LFP')}
MODEL_STYLE = {'abr': dict(marker='D', color='#C0392B', label='model ABR'),
               'lfp': dict(marker='v', color='#2E86C1', label='model LFP')}


def add_common_args(parser, sweep=False, window=True, stim_label=True, side=False):
    """Flags every figure script shares, declared once."""
    parser.add_argument('--pic-file', dest='pic_file', default=None,
                        help='.pic the results were produced from (default: baseline)')
    parser.add_argument('--angle', type=int, default=0)
    parser.add_argument('--out', default=None, help='output directory or file')
    parser.add_argument('--derivation', default='Cz-M1', choices=list(P.DERIVATIONS))
    if sweep:
        parser.add_argument('--sweep', choices=['itd', 'ild', 'angle'], default='itd')
        parser.add_argument('--conditions', type=float, nargs='+', default=None,
                            help='sweep values (µs / dB / degrees)')
    if window:
        parser.add_argument('--win', type=float, nargs=2, default=list(DEFAULT_WINDOW),
                            metavar=('T0', 'T1'),
                            help='onset-peak search window in ms (default 1 15)')
    if stim_label:
        parser.add_argument('--stim-label', dest='stim_label',
                            default=DEFAULT_STIM_LABEL, help='label for figure titles')
    if side:
        parser.add_argument('--side', default='both', choices=['L', 'R', 'both'])
    return parser


def resolve_stem(args):
    """`--pic-file` -> the directory stem the producers used."""
    return paths.pic_stem(paths.resolve_pic(args.pic_file))


def save(fig, path, dpi=150):
    """Write a figure, creating its directory."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'figure saved → {path}')
    return path
