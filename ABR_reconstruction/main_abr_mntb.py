#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
MNTB ABR reconstruction: current-dipole moment -> 4-sphere head model.

Two co-located generators, each a whole-cell CurrentDipoleMoment summed at the
scalp (exact linear superposition; the SBC+GBC composite pattern of
main_abr_avcn.py):
  * PRINCIPAL — MNTB principal cell, calyx-EPSC somatodendritic sink.
  * CALYX     — calyx of Held presynaptic terminal, the "prespike" (leads by
                ~0.5 ms; the postsynaptic sink carries the GBCs2MNTBCs delay).

Both are driven by the CONTRALATERAL GBC spike train and placed at the MNTB
position (~400 µm from the midline, medial + anterior to the MSO).

NON-DOUBLE-COUNTING: the GBC axon crossing the midline — the DOMINANT wave-III
generator (Karadas 2021) — is modelled in the AVCN GBC ABR
(main_abr_avcn.py, VCN_c09_extended_axon.hoc) and is NOT re-added here. The MNTB's
own contribution (closed-field soma + the radial calyx current) is expected small.

CLI:
  mpiexec -n 4 python ABR_reconstruction/main_abr_mntb.py --pic-file RESULTS/<f>.pic \
      --angle 0 --side both --n-cells 200 [--populations principal|calyx|both]

Outputs -> RESULTS/abr_tmp/output_mntb_{stem}_angle{A}_{sidespec}/
           ABR.h5 (principal + calyx + composite), figures/mntb_abr.png
"""
import os
import sys
from collections import defaultdict

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import neuron  # noqa: F401  (mechanisms loaded via the imports below)
import hybridLFPy
import lfpykit.models as lfpykit_models  # noqa: F401
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'LFP_reconstruction'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'ABR_reconstruction'))

# MNTB LFP machinery (also loads MNTB/MSO mechanisms + geometry constants)
from main_reconstruct_mntb import (
    MNTBPopulation, DT, TSTOP, V_INIT, N_MNTB_TOTAL,
    ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y, LAYER_BOUNDARIES, HOC_FILE as MNTB_HOC,
)
from main_reconstruct_calyx import CalyxPopulation, CALYX_HOC
from main_reconstruct import _pic_stem, _extract_spikes
# Head model shared with the MSO/AVCN ABR
from main_abr import (FOUR_SPHERE_RADII, FOUR_SPHERE_SIGMAS, ELECTRODE_POS,
                      MSO_POS_UM, _bandpass, save_dipole_record)

# ---------------------------------------------------------------------------
# MNTB dipole position (head coords, µm). Medial (~400 µm from midline) and
# anterior to the MSO (Kulesza 2015). First-pass estimate anchored to MSO_POS_UM
# (refine from atlas; flagged like AVCN_POS_UM).
# ---------------------------------------------------------------------------
_ANT_OFFSET = 2_000.    # µm anterior to MSO
MNTB_POS_UM = {
    'R': MSO_POS_UM['R'] + np.array([-4_600.,  _ANT_OFFSET, 0.]),   # x: 5000 -> 400
    'L': MSO_POS_UM['L'] + np.array([ 4_600.,  _ANT_OFFSET, 0.]),   # x: -5000 -> -400
}

# ---------------------------------------------------------------------------
# Rotation: model axes -> head-centred axes.  SIDE-SPECIFIC, built from human
# MNTB morphometry (Kulesza 2014/2015) rather than the borrowed MSO placeholder.
#
# MNTB MODEL frame (mntb_model_active.hoc / calyx_model.hoc):
#   model_x  = calyx pre-calyx axon = the CROSSING trapezoid-body fibre (the
#              dominant generator; giant mediolateral axial current)
#   model_y  = principal-cell axon (efferent to MSO/LSO/SPN)  -> rostrocaudal
#   model_z  = principal dendritic long axis (bipolar tufts)
# HEAD frame: x = mediolateral, y = anteroposterior (+ant), z = dorsoventral (+sup).
#
# Base map:
#   RIGHT : identity  -> calyx model_x -> +head_x (mediolateral crossing volley),
#           dendrite model_z -> head_z (vertical), principal axon model_y -> head_y.
#   LEFT  : sagittal mirror diag(-1,1,-1) (flip head_x AND head_z, keep head_y;
#           proper rotation, det=+1) -> calyx -> -head_x.
# So the calyx crossing dipole is mediolateral and MIRRORED across sides (R:+x,
# L:-x) -> sums COHERENTLY bilaterally, and is consistent with the AVCN GBC axon
# orientation (left GBC axon -> +x -> right MNTB calyx +x, and vice-versa).
#
# Literature orientation: MNTB principal long (dendritic) axis is ~73±5 deg
# (transverse) / 79±7 deg (coronal) from horizontal, dendrites PERPENDICULAR to
# the mediolateral trapezoid-body fibres, and the whole SOC has a slight
# superior-rostral tilt.  All three are realised by a single tilt about the
# mediolateral (head_x) axis: it lifts the model_z dendritic axis to
# MNTB_LONGAXIS_FROM_HORIZ_DEG from horizontal (keeping it in the sagittal plane,
# hence still perpendicular to the head_x TB fibres, and with the dorsal end
# leaning rostrally = the SOC superior-rostral lean).  The tilt leaves the
# dominant mediolateral calyx dipole (along head_x) invariant, so the ABR is
# insensitive to its exact value; the closed-field principal dendritic dipole is
# minor by design (opposed tufts self-cancel).
# ---------------------------------------------------------------------------
MNTB_LONGAXIS_FROM_HORIZ_DEG = 76.0    # mean of transverse 73 + coronal 79 (Kulesza);
                                       # subsumes the SOC superior-rostral tilt
                                       # (the measured in-situ angle already reflects it)

_MNTB_BASE = {
    'R': np.eye(3),
    'L': np.diag([-1., 1., -1.]),
}


def _rx(deg):
    """Proper rotation about the mediolateral (head_x) axis, `deg` degrees."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1., 0., 0.],
                     [0., c, -s],
                     [0., s,  c]])


# Tilt the (vertical) dendritic axis toward the rostral-superior, so its final
# elevation is MNTB_LONGAXIS_FROM_HORIZ_DEG and the dorsal end leans rostrally.
_DEND_TILT_FROM_VERTICAL = -(90.0 - MNTB_LONGAXIS_FROM_HORIZ_DEG)   # ~ -14 deg
ROTATION = {s: _rx(_DEND_TILT_FROM_VERTICAL) @ _MNTB_BASE[s] for s in _MNTB_BASE}

# per-generator config
POPULATIONS = {
    'principal': dict(popclass=MNTBPopulation, hoc=MNTB_HOC, seed=46,
                      syn_delay=0.5,  label='principal'),
    'calyx':     dict(popclass=CalyxPopulation, hoc=CALYX_HOC, seed=47,
                      syn_delay=0.05, label='calyx'),
}


def _run_one_source(pop_name, side, args, meta):
    """Run one MNTB generator -> summed whole-cell dipole (3, T) nA·µm on rank 0."""
    from lfpykit import CurrentDipoleMoment
    cfg = POPULATIONS[pop_name]
    contra_side = 'R' if side == 'L' else 'L'

    X_pops       = [f'GBC_{contra_side}']
    k_yxl_local  = [[0], [0], [1]]
    per_pop_syn  = cfg['popclass'].PER_POP_SYN
    weight       = per_pop_syn['GBC']['weight']
    j_yx_local   = [weight]
    tau_yx_local = [per_pop_syn['GBC']['tau2']]

    pic_file   = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                               'baseline_simulation.pic')
    stem       = _pic_stem(pic_file)
    spikes_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'spikes_{stem}_angle{args.angle}_{side}')
    output_dir = os.path.join(REPO_ROOT, 'RESULTS', 'abr_tmp',
                              f'output_mntb_{pop_name}_{stem}_angle{args.angle}_{side}')
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    k_arr         = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT, spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops)

    probe = CurrentDipoleMoment(None)   # cell set by hybridLFPy in cellsim()

    pop_label = f'{cfg["label"]}_{side}'
    pop = cfg['popclass'](
        n_syn_per_pop=n_syn_per_pop, y=pop_label,
        cellParams={'morphology': cfg['hoc'], 'passive': False, 'v_init': V_INIT,
                    'dt': DT, 'tstart': 0., 'tstop': TSTOP, 'nsegs_method': None},
        # NO random rotation for the ABR: the GBC axons cross the midline in a
        # common mediolateral direction, so the pre-calyx axon dipoles must sum
        # COHERENTLY (Karadas 2021 aligned axial current). A random z-rotation
        # (as used for the near-field LFP) would average the x-dipole to zero.
        rand_rot_axis=[],
        simulationParams={'rec_imem': True},
        populationParams={'number': args.n_cells,
                          'radius': ELLIPSE_RADIUS_Y,
                          'radius_x': ELLIPSE_RADIUS_X, 'radius_y': ELLIPSE_RADIUS_Y,
                          'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
                          'min_r': np.array([[0.], [0.]])},
        layerBoundaries=LAYER_BOUNDARIES, probes=[probe], savelist=['somapos'],
        savefolder=output_dir, dt_output=DT, POPULATIONSEED=cfg['seed'], X=X_pops,
        networkSim=networkSim, k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=[cfg['syn_delay']], synDelayScale=[None],
        J_yX=j_yx_local, tau_yX=tau_yx_local)

    pop.run()
    COMM.Barrier()

    n_t = int(round(TSTOP / DT)) + 1
    local = None
    for i in pop.RANK_CELLINDICES:
        d = pop.output[i]['CurrentDipoleMoment'].astype(np.float64)
        local = d.copy() if local is None else local + d
    if local is None:
        local = np.zeros((3, n_t), dtype=np.float64)
    glob = np.zeros_like(local)
    COMM.Reduce(local, glob, op=MPI.SUM, root=0)
    COMM.Barrier()
    return glob, output_dir


def _apply_head_model(sources, output_dir, electrode_names):
    """Project each source dipole from MNTB_POS and SUM scalp potentials."""
    from lfpykit.eegmegcalc import FourSphereVolumeConductor
    r_elec = np.stack([ELECTRODE_POS[e] for e in electrode_names])
    fsc = FourSphereVolumeConductor(r_electrodes=r_elec, radii=FOUR_SPHERE_RADII,
                                    sigmas=FOUR_SPHERE_SIGMAS)
    V_by_part = defaultdict(float)
    for part_label, side, p_head, r_dipole in sources:
        V_by_part[part_label] += fsc.get_dipole_potential(p_head, r_dipole)

    srate = 1.0 / (DT * 1e-3)
    V_out, total_mV = {}, 0.0
    for part_label, V_mV in V_by_part.items():
        total_mV = total_mV + V_mV
        V_out[part_label] = _bandpass(V_mV * 1e3, hi=3000., lo=150.)
    V_out['composite'] = _bandpass(np.asarray(total_mV) * 1e3, hi=3000., lo=150.)

    with h5py.File(os.path.join(output_dir, 'ABR.h5'), 'w') as f:
        for key, V_uV in V_out.items():
            f.create_dataset(key, data=V_uV)
        f.create_dataset('srate', data=srate)
        f.create_dataset('electrode_names', data=np.array(electrode_names, dtype='S'))
        f.attrs['units'] = 'µV'
        f.attrs['keys']  = ','.join(V_out)
    print(f'ABR saved -> {os.path.join(output_dir, "ABR.h5")}  keys={list(V_out)}')
    return V_out, srate


def _plot_abr(output_dir, V_out, electrode_names, srate, angle, side, n_cells):
    cz = electrode_names.index('Cz')
    m1 = electrode_names.index('M1')
    tvec = np.arange(next(iter(V_out.values())).shape[1]) / srate * 1e3
    colours = {'principal': 'steelblue', 'calyx': 'darkorange', 'composite': 'black'}
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for key, V_uV in V_out.items():
        ax0.plot(tvec, V_uV[cz], lw=1.2 if key == 'composite' else 0.9,
                 color=colours.get(key), label=f'{key} (Cz)',
                 zorder=3 if key == 'composite' else 2)
    ax0.axhline(0, color='k', lw=0.4, ls=':')
    ax0.set_ylabel('Cz potential (µV)')
    ax0.set_title(f'MNTB ABR (principal + calyx prespike) | angle {angle}° | '
                  f'side {side} | N={n_cells}')
    ax0.legend(fontsize=9)
    diff = V_out['composite'][cz] - V_out['composite'][m1]
    ax1.plot(tvec, diff, color='darkorchid', lw=1.0, label='composite Cz-M1')
    ax1.axhline(0, color='k', lw=0.4, ls=':')
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title('Cz-M1 (vertex-positive upward)')
    ax1.legend(fontsize=9)
    path = os.path.join(output_dir, 'figures', 'mntb_abr.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'ABR figure saved -> {path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MNTB ABR reconstruction')
    parser.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',    type=int, default=0)
    parser.add_argument('--side',     type=str, default='L', choices=['L', 'R', 'both'])
    parser.add_argument('--n-cells',  type=int, default=200, dest='n_cells')
    parser.add_argument('--populations', type=str, default='both',
                        choices=['principal', 'calyx', 'both'])
    args = parser.parse_args()

    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    parts = (['principal', 'calyx'] if args.populations == 'both'
             else [args.populations])
    stem  = _pic_stem(args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                                    'baseline_simulation.pic'))

    meta_by_side = {}
    for side in sides:
        meta_by_side[side] = (_extract_spikes(args.angle, side, pic_file=args.pic_file)
                              if RANK == 0 else None)
        meta_by_side[side] = COMM.bcast(meta_by_side[side], root=0)
    COMM.Barrier()

    sources = []
    for pop_name in parts:
        for side in sides:
            dipole, _ = _run_one_source(pop_name, side, args, meta_by_side[side])
            if RANK == 0:
                p_head = ROTATION[side] @ dipole
                sources.append((pop_name, side, p_head, MNTB_POS_UM[side]))
                save_dipole_record(stem, args.angle, 'MNTB', pop_name, side,
                                   p_head, MNTB_POS_UM[side], N_MNTB_TOTAL,
                                   args.n_cells)

    if RANK == 0:
        pic_file = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                                 'baseline_simulation.pic')
        stem = _pic_stem(pic_file)
        final_dir = os.path.join(REPO_ROOT, 'RESULTS', 'abr_tmp',
                                 f'output_mntb_{stem}_angle{args.angle}_{args.side}')
        os.makedirs(os.path.join(final_dir, 'figures'), exist_ok=True)
        electrode_names = ['Cz', 'M1', 'M2']
        V_out, srate = _apply_head_model(sources, final_dir, electrode_names)
        _plot_abr(final_dir, V_out, electrode_names, srate,
                  args.angle, args.side, args.n_cells)


if __name__ == '__main__':
    main()
