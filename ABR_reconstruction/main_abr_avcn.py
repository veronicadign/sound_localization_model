#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
AVCN ABR reconstruction via current-dipole moment + 4-sphere head model.

The cochlear nucleus is an early ABR generator (human wave III), upstream of the
MSO.  This mirrors ABR_reconstruction/main_abr.py but drives morphologically
detailed bushy-cell populations (AVCNPopulation, ANF endbulb input) and places
each dipole at the cochlear-nucleus location.

Two generators are modelled and, per the request, UNIFIED at the ABR stage:
  * GBC — globular bushy cell (Type II, VCN_c09 EM), central/caudal VCN.
  * SBC — spherical bushy cell (Type II-I, SBC_S113 EM), rostral VCN.
The spherical-cell area occupies the rostral ~1.5-2.0 mm of the nucleus before
the globular region, so the SBC dipole is placed ~1.5 mm anterior (+head_y) of
the GBC.  Each population's dipole is projected through the 4-sphere model from
its OWN position and the scalp potentials are SUMMED — exact linear superposition
that honours the rostrocaudal offset (a plain vector sum of the two dipole
moments would only be valid for co-located sources).

CLI (single or MPI):
  python ABR_reconstruction/main_abr_avcn.py --pic-file RESULTS/x.pic \
      --angle 0 --side L --n-cells 200 --populations both
  mpiexec -n 4 python ABR_reconstruction/main_abr_avcn.py ... --side both

Outputs:
  RESULTS/abr_tmp/output_{pop}_{stem}_angle{A}_{side}/population_dipole.h5
  RESULTS/abr_tmp/output_avcn_{stem}_angle{A}_{sidespec}/ABR.h5   (per-pop + composite)
  .../figures/avcn_abr.png                                        (SBC/GBC/composite)
"""

import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI

import hybridLFPy

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# AVCN populations + geometry (mechanisms loaded on import of main_reconstruct_avcn)
sys.path.insert(0, os.path.join(REPO_ROOT, 'LFP_reconstruction'))
from main_reconstruct_avcn import (          # noqa: E402
    AVCNPopulation,
    ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y, LAYER_BOUNDARIES, K_YXL,
    HOC_FILE, _decorate, DT, TSTOP, V_INIT, SYN_DELAY_LOC, SYN_DELAY_SCALE,
    AXON_TARGET, N_GBC_TOTAL, N_ENDBULBS,
)
import main_reconstruct_sbc as sbc            # noqa: E402  (SBC deltas)
from main_reconstruct import _pic_stem, _extract_spikes   # noqa: E402
sys.path.insert(0, os.path.join(REPO_ROOT, 'AVCN_models'))
import gbc_biophysics                         # noqa: E402

# Shared head-model constants + band-pass (reuse MSO ABR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_abr import save_dipole_record, superpose_sources   # noqa: E402

# ---------------------------------------------------------------------------
# AVCN dipole source position in head-centred coordinates (µm).
#
# The human cochlear nucleus sits on the DORSOLATERAL surface at the pontomedullary
# junction — more lateral and caudal than the MSO, but dorsal, NOT deep-ventral.
# So z sits only ~5-6 mm below the SOC (MSO/LSO z ≈ -30 mm), reflecting the more
# caudal level while keeping the nucleus dorsal (an earlier -42 mm placement was
# too ventral).  First-pass estimate (refine from MRI/atlas as was done for
# MSO_POS_UM in main_abr.py); |r| stays inside the 79 mm brain sphere.
# x = mediolateral (±), y = anteroposterior (ant +), z = inferosuperior.
# GBC position is the reference; SBC sits ~1.5 mm rostral (+head_y).
# ---------------------------------------------------------------------------
AVCN_POS_UM = {
    'R': np.array([ 10_000., -38_000., -35_000.]),
    'L': np.array([-10_000., -38_000., -35_000.]),
}
SBC_ROSTRAL_OFFSET_UM = np.array([0., 1_500., 0.])   # +head_y = anterior/rostral
# SBCs sit slightly MEDIAL to GBCs: GBCs cluster around/lateral to the auditory
# nerve-root entry, while SBCs dominate the rostral pole / ascending-branch field
# (Moore/Osen).  Offset is toward the midline (sign opposite to the side's x).
SBC_MEDIAL_OFFSET_UM = 600.
SBC_POS_UM = {
    s: (AVCN_POS_UM[s] + SBC_ROSTRAL_OFFSET_UM
        - np.sign(AVCN_POS_UM[s][0]) * np.array([SBC_MEDIAL_OFFSET_UM, 0., 0.]))
    for s in AVCN_POS_UM
}

# ---------------------------------------------------------------------------
# Rotation: model axes -> head-centred axes.  SIDE-SPECIFIC (mirror across the
# mid-sagittal plane), so the GBC axon — aligned ventromedially in the MODEL
# frame by set_rotations (AXON_TARGET = [-1,0,-1]) — points toward the
# CONTRALATERAL MNTB, i.e. it crosses the midline on BOTH sides.  This follows
# the MSO convention (main_abr.py): flip head_x AND head_z between sides, keep
# head_y (both proper rotations, det = +1).
#
#   R:  head_x=+model_x, head_y=-model_z, head_z=+model_y   (Rx(+90°))
#   L:  head_x=-model_x, head_y=-model_z, head_z=-model_y   (sagittal mirror)
#
# Axon crossing check (AXON_TARGET=[-1,0,-1] -> head, before tilt):
#   R -> [-1,+1,0]  head_x=-1  → toward midline from the RIGHT AVCN  ✓ crosses
#   L -> [+1,+1,0]  head_x=+1  → toward midline from the LEFT  AVCN  ✓ crosses
#
# Anatomical refinement (Moore/Osen): the human CN rostral pole is rotated
# OUTWARD ~30-35° from the neuraxis (upright, oblique orientation).  We add a
# rotation about the head vertical axis (head_z) by ∓AVCN_ROSTRAL_TILT_DEG so the
# rostral end (+head_y) swings toward +head_x on the RIGHT and -head_x on the
# LEFT.  The midline crossing survives any tilt; set the angle to 0 (or
# --avcn-tilt-deg 0) to recover the untilted mapping.
# ---------------------------------------------------------------------------
AVCN_ROSTRAL_TILT_DEG = 32.5   # human CN outward rostral rotation; 0 disables

_R_BASE = {
    'R': np.array([[ 1., 0.,  0.],
                   [ 0., 0., -1.],
                   [ 0., 1.,  0.]]),
    'L': np.array([[-1., 0.,  0.],
                   [ 0., 0., -1.],
                   [ 0., -1., 0.]]),
}


def _rz(deg):
    """Proper rotation about the head vertical (z) axis, `deg` degrees."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.],
                     [s,  c, 0.],
                     [0., 0., 1.]])


def _build_rotation(tilt_deg):
    """Side-specific model->head rotation with an outward rostral tilt.

    Outward = rostral end (+head_y) toward +head_x on the RIGHT (phi<0) and
    -head_x on the LEFT (phi>0), mirroring across the mid-sagittal plane.
    """
    return {'R': _rz(-tilt_deg) @ _R_BASE['R'],
            'L': _rz(+tilt_deg) @ _R_BASE['L']}


ROTATION = _build_rotation(AVCN_ROSTRAL_TILT_DEG)

# ---------------------------------------------------------------------------
# WHY THE DIPOLE IS *NOT* SPLIT INTO SYNAPTIC + AXONAL PARTS
#
# It is tempting to compute a somatodendritic dipole at the AVCN and a separate
# axonal (travelling-volley) dipole displaced along the tract.  That is
# PHYSICALLY INVALID with dipole-only tooling, and it was measured to be so:
#
#   sum of i_membrane over ALL segments      = 3.0 nA  (= injected current; the
#                                              cell as a whole conserves charge)
#   sum over the AXONAL segments alone       = 3.2 nA  (NOT zero)
#
# A dipole moment p = sum(r * i) is translation-invariant — i.e. a well-defined
# dipole — only when the group's NET current is zero.  Each sub-group carries a
# large net current, so its "dipole" is origin-dependent, and displacing the two
# groups to different positions drops the monopole terms.  Since q_axon = -q_syn
# separated by d ~ 3.5 mm, the dropped term is itself a dipole of moment q*d —
# large, and absent from such a model.  Splitting produced a spurious anti-phase
# cancellation between the two parts.
#
# The correct treatment with FourSphereVolumeConductor (which models a current
# DIPOLE, not monopoles) is ONE dipole per cell over all segments: sum(i) = 0
# makes it well defined, and it ALREADY contains the axonal travelling wave
# because CurrentDipoleMoment sums every segment, extended axon included.  The
# only approximation left is the far-field point-dipole placement of a ~4 mm
# source at 79 mm — a small error.  A genuinely distributed treatment would need
# monopole (point-source) support in the head model.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Population registry: everything that differs between GBC and SBC.  Defaults in
# AVCNPopulation reproduce the GBC pipeline, so the GBC entry mostly names them.
#
# 'parts' lists the dipole sources a population contributes, each with its own
# head position.  Both populations contribute ONE whole-cell dipole (see above);
# they differ in position (SBC 1.5 mm rostral) and in morphology — the GBC uses
# the EXTENDED active axon so its dipole carries the travelling volley.
# ---------------------------------------------------------------------------
GBC_EXTENDED_HOC = os.path.join(REPO_ROOT, 'AVCN_models', 'morphology',
                                'extended', 'VCN_c09_extended_axon.hoc')

POPULATIONS = {
    'gbc': dict(
        label='GBC', morphology=GBC_EXTENDED_HOC, decorate=_decorate,
        n_post_total=N_GBC_TOTAL, n_endbulbs=N_ENDBULBS, endbulb_weights=None,
        per_pop_syn=AVCNPopulation.PER_POP_SYN, k_yxl=K_YXL,
        ellipse_y=ELLIPSE_RADIUS_Y, seed=44,
        parts=[('GBC', None, AVCN_POS_UM)],   # whole-cell dipole (incl. axon)
    ),
    'sbc': dict(
        label='SBC', morphology=sbc.HOC_FILE_SBC, decorate=sbc._decorate_sbc,
        n_post_total=sbc.N_SBC_TOTAL, n_endbulbs=sbc.N_ENDBULBS,
        endbulb_weights=sbc.ENDBULB_WEIGHTS_SBC, per_pop_syn=sbc.PER_POP_SYN_SBC,
        k_yxl=sbc.K_YXL, ellipse_y=sbc.ELLIPSE_RADIUS_Y, seed=45,
        parts=[('SBC', None, SBC_POS_UM)],   # None = plain CurrentDipoleMoment
    ),
}


# ---------------------------------------------------------------------------
# Per-(population, side) simulation -> population dipole (3, T) nA·µm
# ---------------------------------------------------------------------------
def _run_one_source(pop_name, side, args, meta):
    from lfpykit import CurrentDipoleMoment
    cfg = POPULATIONS[pop_name]

    X_pops       = [f'ANF_{side}']
    k_yxl_local  = cfg['k_yxl']
    j_yx_local   = [cfg['per_pop_syn']['ANF']['weight']]
    tau_yx_local = [cfg['per_pop_syn']['ANF']['tau2']]

    pic_file   = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                               'baseline_simulation.pic')
    stem       = _pic_stem(pic_file)
    spikes_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'spikes_{stem}_angle{args.angle}_{side}')
    output_dir = os.path.join(REPO_ROOT, 'RESULTS', 'abr_tmp',
                              f'output_{pop_name}_{stem}_angle{args.angle}_{side}')
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    k_arr         = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT,
        spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops,
    )

    # one probe per dipole part (somatodendritic / axonal, or a single plain one)
    probes, part_keys = [], []
    for part_label, probe_cls, _pos in cfg['parts']:
        cls = probe_cls or CurrentDipoleMoment
        probes.append(cls(None))            # cell set by hybridLFPy in cellsim()
        part_keys.append((part_label, cls.__name__))

    pop_label = f'{cfg["label"]}_{side}'
    pop = AVCNPopulation(
        n_syn_per_pop=n_syn_per_pop,
        axon_target=AXON_TARGET,   # fixed ventromedial (laterality via X_pops)
        n_post_total=cfg['n_post_total'],
        n_endbulbs=cfg['n_endbulbs'],
        endbulb_weights=cfg['endbulb_weights'],
        per_pop_syn=cfg['per_pop_syn'],
        y=pop_label,
        cellParams={
            'morphology': cfg['morphology'], 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP,
            'nsegs_method': 'lambda_f', 'lambda_f': 100,
            'custom_fun': [cfg['decorate']], 'custom_fun_args': [{}],
        },
        rand_rot_axis=[],   # deterministic ventromedial axon orientation instead
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   args.n_cells,
            'radius':   cfg['ellipse_y'],
            'radius_x': ELLIPSE_RADIUS_X,
            'radius_y': cfg['ellipse_y'],
            'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=probes,
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=cfg['seed'],
        X=X_pops,
        networkSim=networkSim,
        k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=SYN_DELAY_LOC,
        synDelayScale=SYN_DELAY_SCALE,
        J_yX=j_yx_local,
        tau_yX=tau_yx_local,
    )

    pop.run()
    COMM.Barrier()

    # Sum per-cell dipole moments (3, T) nA·µm on this rank, per part
    n_t = int(round(TSTOP / DT)) + 1
    dipoles = {}
    for part_label, out_key in part_keys:
        local = None
        for i in pop.RANK_CELLINDICES:
            d = pop.output[i][out_key].astype(np.float64)
            local = d.copy() if local is None else local + d
        if local is None:
            local = np.zeros((3, n_t), dtype=np.float64)
        glob = np.zeros_like(local)
        COMM.Reduce(local, glob, op=MPI.SUM, root=0)
        dipoles[part_label] = glob
    COMM.Barrier()
    return dipoles, output_dir


# ---------------------------------------------------------------------------
# Head-model projection + saving (rank 0)
# ---------------------------------------------------------------------------
def _project_and_save(side, dipoles, output_dir):
    """Rotate each part's dipole into head coords and save. Returns {part: p_head}."""
    srate = 1.0 / (DT * 1e-3)
    out = {}
    with h5py.File(os.path.join(output_dir, 'population_dipole.h5'), 'w') as f:
        for part, p_model in dipoles.items():
            p_head = ROTATION[side] @ p_model
            f.create_dataset(part, data=p_head)
            out[part] = p_head
        f.create_dataset('srate', data=srate)
        f.attrs['axes']  = 'x=mediolateral, y=anteroposterior, z=inferosuperior'
        f.attrs['units'] = 'nA·µm'
        f.attrs['parts'] = ','.join(out)
    return out


def _apply_head_model(sources, output_dir, electrode_names):
    """Project each source dipole from ITS OWN position and SUM scalp potentials.

    Thin wrapper over the shared ``main_abr.superpose_sources`` (the single
    canonical 4-sphere superposition, reused by the cross-nucleus composite in
    main_abr_full.py) that additionally writes this run's ABR.h5.  `sources` is a
    list of (part_label, side, p_head, r_dipole); sources sharing a part_label
    (e.g. both sides of one generator) sum.

    Returns (V_by_key, srate); V_by_key maps each part label AND 'composite' to a
    band-passed scalp potential array (n_electrodes, n_t) in µV.
    """
    V_out, srate = superpose_sources(sources, electrode_names, hi=3000., lo=150.)

    with h5py.File(os.path.join(output_dir, 'ABR.h5'), 'w') as f:
        for key, V_uV in V_out.items():
            f.create_dataset(key, data=V_uV)
        f.create_dataset('srate', data=srate)
        f.create_dataset('electrode_names', data=np.array(electrode_names, dtype='S'))
        f.attrs['units'] = 'µV'
        f.attrs['keys']  = ','.join(V_out)
    print(f'ABR saved → {os.path.join(output_dir, "ABR.h5")}  keys={list(V_out)}')
    return V_out, srate


def _derivation(V_uV, electrode_names, derivation):
    idx = {e: electrode_names.index(e) for e in electrode_names}
    cz, m1, m2 = V_uV[idx['Cz']], V_uV[idx['M1']], V_uV[idx['M2']]
    if derivation == 'Cz-M2':
        return cz - m2, 'Cz−M2'
    if derivation == 'Cz-avg':
        return cz - (m1 + m2) / 2.0, 'Cz−(M1+M2)/2'
    return cz - m1, 'Cz−M1'


def _plot_abr(output_dir, V_out, electrode_names, srate, angle, side, n_cells,
              derivation='Cz-M1'):
    """Top: Cz for each population + composite. Bottom: composite derivation."""
    cz_idx = electrode_names.index('Cz')
    any_V  = next(iter(V_out.values()))
    tvec   = np.arange(any_V.shape[1]) / srate * 1e3
    colours = {'GBC': 'seagreen', 'SBC': 'darkorange', 'composite': 'black'}

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for key, V_uV in V_out.items():
        ax0.plot(tvec, V_uV[cz_idx], lw=1.1 if key == 'composite' else 0.9,
                 color=colours.get(key, None),
                 label=f'{key} (Cz)', zorder=3 if key == 'composite' else 2)
    ax0.axhline(0, color='k', lw=0.4, ls=':')
    ax0.set_ylabel('Cz potential (µV)')
    ax0.set_title(f'AVCN ABR (SBC + GBC) | angle {angle}° | side {side} | N={n_cells}')
    ax0.legend(fontsize=9)

    diff, lbl = _derivation(V_out['composite'], electrode_names, derivation)
    ax1.plot(tvec, diff, color='darkorchid', lw=1.0, label=f'composite {lbl}')
    ax1.axhline(0, color='k', lw=0.4, ls=':')
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'{lbl}  (vertex-positive upward)')
    ax1.legend(fontsize=9)

    path = os.path.join(output_dir, 'figures', 'avcn_abr.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'ABR figure saved → {path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='AVCN (SBC + GBC) ABR reconstruction')
    parser.add_argument('--pic-file',    type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',       type=int, default=0)
    parser.add_argument('--side',        type=str, default='L',
                        choices=['L', 'R', 'both'])
    parser.add_argument('--n-cells',     type=int, default=200, dest='n_cells')
    parser.add_argument('--populations', type=str, default='both',
                        choices=['gbc', 'sbc', 'both'],
                        help='which bushy-cell generator(s) to include')
    parser.add_argument('--derivation',  type=str, default='Cz-M1',
                        choices=['Cz-M1', 'Cz-M2', 'Cz-avg'])
    parser.add_argument('--gbc-hoc', type=str, default=None, dest='gbc_hoc',
                        help='override the GBC morphology (default: extended '
                             'active axon; pass the plain VCN_c* EM hoc to '
                             'measure the axonal travelling-wave contribution)')
    parser.add_argument('--avcn-tilt-deg', type=float, default=AVCN_ROSTRAL_TILT_DEG,
                        dest='avcn_tilt_deg',
                        help='outward rostral tilt of the CN about the head '
                             'vertical axis (deg); 0 = untilted (default %(default)s)')
    args = parser.parse_args()

    global ROTATION
    ROTATION = _build_rotation(args.avcn_tilt_deg)

    if args.gbc_hoc:
        POPULATIONS['gbc']['morphology'] = args.gbc_hoc

    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    pops  = ['gbc', 'sbc'] if args.populations == 'both' else [args.populations]
    stem  = _pic_stem(args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                                     'baseline_simulation.pic'))

    # Extract presynaptic spikes once per side (same ANF drive for both pops).
    meta_by_side = {}
    for side in sides:
        if RANK == 0:
            meta_by_side[side] = _extract_spikes(args.angle, side, pic_file=args.pic_file)
        else:
            meta_by_side[side] = None
        meta_by_side[side] = COMM.bcast(meta_by_side[side], root=0)
    COMM.Barrier()

    sources = []   # (part_label, side, p_head, r_dipole) on rank 0
    for pop_name in pops:
        for side in sides:
            dipoles, output_dir = _run_one_source(pop_name, side, args,
                                                  meta_by_side[side])
            if RANK == 0:
                p_heads = _project_and_save(side, dipoles, output_dir)
                for part_label, _probe_cls, pos_map in POPULATIONS[pop_name]['parts']:
                    sources.append((part_label, side, p_heads[part_label],
                                    pos_map[side]))
                    save_dipole_record(stem, args.angle, 'AVCN', part_label, side,
                                       p_heads[part_label], pos_map[side],
                                       POPULATIONS[pop_name]['n_post_total'],
                                       args.n_cells)

    if RANK == 0:
        pic_file = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                                 'baseline_simulation.pic')
        stem = _pic_stem(pic_file)
        tag  = 'avcn' if len(pops) > 1 else pops[0]
        final_dir = os.path.join(REPO_ROOT, 'RESULTS', 'abr_tmp',
                                 f'output_{tag}_{stem}_angle{args.angle}_{args.side}')
        os.makedirs(os.path.join(final_dir, 'figures'), exist_ok=True)

        electrode_names = ['Cz', 'M1', 'M2']
        V_out, srate = _apply_head_model(sources, final_dir, electrode_names)
        _plot_abr(final_dir, V_out, electrode_names, srate,
                  args.angle, args.side, args.n_cells, derivation=args.derivation)


if __name__ == '__main__':
    main()
