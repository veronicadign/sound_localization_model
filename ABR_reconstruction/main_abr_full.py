"""
Complete brainstem ABR — superpose ALL nuclei into one composite.

Each per-nucleus ABR script (main_abr{,_avcn,_lso,_mntb}.py) writes a standardised
head-frame dipole record per (generator, side) into

    RESULTS/abr_tmp/dipoles/<stem>_angle<A>/<nucleus>__<generator>__<side>.h5

(see main_abr.save_dipole_record).  This orchestrator collects those records for
one stimulus, projects every generator through the SAME 4-sphere head model from
its own anatomical position, and SUMS the scalp potentials (exact linear
superposition).  No NEURON simulation is re-run here — run each nucleus once (at
full N), then this.

Because all four pipelines share DT=0.026 ms / TSTOP=50 ms, every dipole lives on
the same time axis, so the inter-wave latencies fall straight out of the NEST
spike timing.  Full-count runs are assumed (no N_total/n_cells scaling): the raw
head dipoles are summed as-is.

Generators (each modelled exactly once — no double counting):
  AVCN:GBC   wave II–III   (GBC soma+dendrites + 4 mm crossing axon = the volley)
  AVCN:SBC   wave II
  MNTB:principal / MNTB:calyx   small own field (wave III proper is the GBC axon)
  MSO:postsynaptic            wave IV–V
  LSO:spiking|synaptic        wave IV–V (lateral-lemniscus travelling wave / BIC)
Wave I (ANF) is NOT modelled — the earliest generator is the GBC.

Output:
  RESULTS/abr_tmp/output_full_<stem>_angle<A>_<sidespec>/ABR_full.h5
        (per-generator + per-nucleus + composite, µV) + figures/full_abr.png

Usage:
  python ABR_reconstruction/main_abr_full.py --pic-file RESULTS/<f>.pic \
        --angle 0 --side both [--nuclei MSO,AVCN,LSO,MNTB] [--derivation Cz-M1]
"""

import os
import re
import sys
import glob
import argparse
from collections import defaultdict

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'ABR_reconstruction'))

from main_abr import dipoles_dir_for, superpose_sources   # noqa: E402

ELECTRODES = ['Cz', 'M1', 'M2']


def _pic_stem(pic_file):
    """Sanitised basename of a .pic path (identical to main_reconstruct._pic_stem
    so the stem matches what the producers wrote — inlined to keep this
    orchestrator NEURON-free)."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_',
                  os.path.splitext(os.path.basename(pic_file))[0])


# ---------------------------------------------------------------------------
def _discover(stem, angle, want_nuclei, want_sides, want_condition,
              lso_drive='spiking'):
    """Load matching dipole records → list of (label, nucleus, side, p_head, r).

    Per (nucleus, generator, side) the requested `want_condition` is used if a
    record for it exists, else the 'binaural' record is used as fallback — so a
    monaural composite reuses the condition-invariant AVCN/MNTB records (only ever
    written as 'binaural') while picking the condition-specific MSO/LSO records.

    `lso_drive` selects the LSO generator: 'spiking' (output travelling wave) and
    'synaptic' (postsynaptic dipole) are ALTERNATIVE models of the LSO's ABR
    contribution — never summed — so only the requested one is kept.

    Under a MONAURAL condition (left_ear/right_ear) the monaural-by-construction
    nuclei are restricted to the ear-appropriate hemisphere: the silent ear drives
    no click-locked response, so AVCN (ipsilateral to the ear) keeps only that side
    and MNTB (contralateral — the GBC→calyx decussates) keeps the opposite side.
    MSO/LSO keep both hemispheres (both receive monaural drive).
    """
    # ear -> (AVCN side, MNTB side) for monaural conditions
    _EAR_SIDES = {'right_ear': ('R', 'L'), 'left_ear': ('L', 'R')}
    mono = _EAR_SIDES.get(want_condition)

    ddir = dipoles_dir_for(stem, angle)
    paths = sorted(glob.glob(os.path.join(ddir, '*.h5')))
    if not paths:
        raise FileNotFoundError(
            f'no dipole records in {ddir}\n  Run each nucleus ABR first, e.g.\n'
            f'    python ABR_reconstruction/main_abr_avcn.py --pic-file ... '
            f'--angle {angle} --side both')

    cand = {}   # (nucleus, generator, side) -> {condition: (label, nucleus, side, p_head, r)}
    for p in paths:
        with h5py.File(p, 'r') as f:
            nucleus   = f.attrs['nucleus']
            generator = f.attrs['generator']
            side      = f.attrs['side']
            cond      = f.attrs.get('condition', 'binaural')
            if want_nuclei and nucleus not in want_nuclei:
                continue
            if side not in want_sides:
                continue
            if nucleus == 'LSO' and generator != lso_drive:
                continue   # keep only the requested LSO drive (mutually exclusive)
            if mono is not None:
                # monaural: keep only the ear-driven hemisphere for AVCN/MNTB
                if nucleus == 'AVCN' and side != mono[0]:
                    continue
                if nucleus == 'MNTB' and side != mono[1]:
                    continue
            cand.setdefault((nucleus, generator, side), {})[cond] = (
                f'{nucleus}:{generator}', nucleus, side,
                f['p_head'][:], f['r_dipole'][:])

    recs = []
    for by_cond in cand.values():
        if want_condition in by_cond:
            recs.append(by_cond[want_condition])
        elif 'binaural' in by_cond:
            recs.append(by_cond['binaural'])
    if not recs:
        raise FileNotFoundError(
            f'records exist in {ddir} but none match nuclei={want_nuclei} '
            f'sides={want_sides} condition={want_condition}')
    return recs


def assemble(stem, angle, want_nuclei, sides, condition, lso_drive='synaptic',
             band=(150., 3000.)):
    """Collect dipole records for one condition and superpose → composite.

    Returns (V_gen, V_nuc, srate): V_gen maps 'nucleus:generator' + 'composite' to
    band-passed (n_e, T) µV; V_nuc maps nucleus → summed µV. Reused by main() and
    by the Curio & Weigel BI reproduction (main_abr_bi.py).
    """
    lo, hi = band
    recs = _discover(stem, angle, want_nuclei, set(sides), condition,
                     lso_drive=lso_drive)
    sources = [(label, side, p_head, r) for (label, _nuc, side, p_head, r) in recs]
    V_gen, srate = superpose_sources(sources, ELECTRODES, hi=hi, lo=lo)

    # per-nucleus = sum of that nucleus's generator traces (linear filter ⇒ safe)
    nuc_of = {f'{n}:{g}': n for (lbl, n, s, _p, _r) in recs
              for (g,) in [(lbl.split(':', 1)[1],)]}
    V_nuc = defaultdict(lambda: 0.0)
    for label, V in V_gen.items():
        if label == 'composite':
            continue
        V_nuc[nuc_of[label]] = V_nuc[nuc_of[label]] + V
    return V_gen, dict(V_nuc), srate


def _derive(V, names, kind):
    cz, m1, m2 = (V[names.index(e)] for e in ('Cz', 'M1', 'M2'))
    if kind == 'Cz-M1':  return cz - m1, 'Cz−M1'
    if kind == 'Cz-M2':  return cz - m2, 'Cz−M2'
    return cz - 0.5 * (m1 + m2), 'Cz−(M1+M2)/2'


def _plot(out_dir, V_gen, V_nuc, srate, side, derivation):
    t = np.arange(V_gen['composite'].shape[1]) / srate * 1e3
    cz = ELECTRODES.index('Cz')
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)

    for nuc in sorted(V_nuc):
        ax0.plot(t, V_nuc[nuc][cz], lw=0.9, label=nuc)
    ax0.plot(t, V_gen['composite'][cz], color='k', lw=1.6, label='composite', zorder=5)
    ax0.axhline(0, color='k', lw=0.4, ls=':')
    ax0.set_xlabel('Time (ms)'); ax0.set_ylabel('Cz potential (µV)')
    ax0.set_title(f'composite ABR AND per-nucleus decomposition (Cz)  |  side {side}')
    ax0.legend(fontsize=8, ncol=2)

    diff, lbl = _derive(V_gen['composite'], ELECTRODES, derivation)
    ax1.plot(t, diff, color='darkorchid', lw=1.1, label=f'composite {lbl}')
    ax1.axhline(0, color='k', lw=0.4, ls=':')
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'Composite ABR {lbl}  (vertex-positive upward)')
    ax1.legend(fontsize=9)

    path = os.path.join(out_dir, 'figures', 'full_abr.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'figure saved → {path}')


def _summary(V_nuc, V_gen, srate):
    """Per-nucleus Cz onset/peak table — the wave-ordering check.

    Onset (first crossing of 30 % of the peak, after the band-pass edge) is the
    robust ordering metric; peak latency is coarser (waveform-shape dependent).
    """
    t = np.arange(V_gen['composite'].shape[1]) / srate * 1e3
    cz = ELECTRODES.index('Cz')
    m = t > 1.0   # skip band-pass edge transient

    def onset(v, frac=0.3):
        a = np.abs(v) * m
        return t[int(np.argmax(a > frac * a.max()))]

    print('\n  nucleus        onset (ms)   peak (ms)   peak|Cz| (µV)')
    print('  ' + '-' * 52)
    for nuc in sorted(V_nuc):
        v = V_nuc[nuc][cz]; i = int(np.argmax(np.abs(v) * m))
        print(f'  {nuc:<12s}   {onset(v):>8.2f}   {t[i]:>8.2f}   {np.abs(v[i]):>12.3e}')
    v = V_gen['composite'][cz]; i = int(np.argmax(np.abs(v) * m))
    print(f'  {"composite":<12s}   {onset(v):>8.2f}   {t[i]:>8.2f}   {np.abs(v[i]):>12.3e}\n')


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Complete brainstem ABR (all nuclei)')
    ap.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    ap.add_argument('--angle',    type=int, default=0)
    ap.add_argument('--side',     type=str, default='both', choices=['L', 'R', 'both'])
    ap.add_argument('--nuclei',   type=str, default=None,
                    help='comma-separated subset (e.g. MSO,AVCN); default = all present')
    ap.add_argument('--derivation', type=str, default='Cz-M1',
                    choices=['Cz-M1', 'Cz-M2', 'Cz-avg'])
    ap.add_argument('--condition', type=str, default='binaural',
                    choices=['binaural', 'left_ear', 'right_ear'],
                    help='acoustic condition: picks condition-specific MSO/LSO '
                         'records, falling back to binaural for AVCN/MNTB')
    ap.add_argument('--lso-drive', type=str, default='synaptic',
                    choices=['spiking', 'synaptic'], dest='lso_drive',
                    help='which LSO generator to include (alternatives, not summed)')
    ap.add_argument('--band', type=str, default='150,3000',
                    help='band-pass lo,hi in Hz (default 150,3000)')
    args = ap.parse_args()

    pic_file = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                             'baseline_simulation.pic')
    stem  = _pic_stem(pic_file)
    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    want_nuclei = (set(n.strip() for n in args.nuclei.split(',')) if args.nuclei
                   else None)
    lo, hi = (float(x) for x in args.band.split(','))

    V_gen, V_nuc, srate = assemble(stem, args.angle, want_nuclei, sides,
                                   args.condition, lso_drive=args.lso_drive,
                                   band=(lo, hi))
    print(f'collected generators ({args.condition}, LSO={args.lso_drive}): '
          f'{sorted(k for k in V_gen if k != "composite")}')

    cond_tag = '' if args.condition == 'binaural' else f'_{args.condition}'
    lso_tag  = '' if args.lso_drive == 'spiking' else f'_lso{args.lso_drive}'
    out_dir = os.path.join(REPO_ROOT, 'RESULTS', 'abr_tmp',
                           f'output_full_{stem}_angle{args.angle}_{args.side}{cond_tag}{lso_tag}')
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)

    with h5py.File(os.path.join(out_dir, 'ABR_full.h5'), 'w') as f:
        for label, V in V_gen.items():          # per-generator + composite
            f.create_dataset(label.replace(':', '__'), data=V)
        for nuc, V in V_nuc.items():            # per-nucleus
            f.create_dataset(f'nucleus__{nuc}', data=V)
        f.create_dataset('srate', data=srate)
        f.create_dataset('electrode_names', data=np.array(ELECTRODES, dtype='S'))
        f.attrs.update(units='µV', stem=stem, angle=str(args.angle),
                       side=args.side, band=f'{lo}-{hi} Hz')
    print(f'ABR_full.h5 saved → {out_dir}')

    _summary(V_nuc, V_gen, srate)
    _plot(out_dir, V_gen, V_nuc, srate, args.side, args.derivation)


if __name__ == '__main__':
    main()
