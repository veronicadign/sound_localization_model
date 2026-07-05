#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
MSO ABR reconstruction via current-dipole moment + 4-sphere head model.

CLI:
  python ABR_reconstruction/main_abr.py [options]
  mpiexec -n 4 python ABR_reconstruction/main_abr.py --angle 45 --side R --n-cells 200

Options:
  --pic-file FILE          Path to .pic simulation result
  --angle DEGREES          Sound azimuth angle (default: 0)
  --side L|R|both          Brain side (default: L)
  --n-cells N              MSO cells to simulate (default: 100)
  --derivation Cz-M1|Cz-M2|Cz-avg  Differential to plot (default: Cz-M1)
  --condition binaural|left_ear|right_ear  Acoustic condition (default: binaural)

Outputs saved to RESULTS/abr_tmp/output_<stem>_angle<A>_<side>/:
  ABR.h5                   — scalp potentials (n_electrodes, n_t), mV
  population_dipole.h5     — population dipole moment (3, n_t), nA·µm
  figures/mso_abr.png      — ABR waveforms + differentials
  figures/mso_abr_phase_cycle.png  — phase-averaged ABR

Unit conventions (all lfpykit-native, no conversion needed):
  Positions / electrode coords : µm
  Dipole moment (CurrentDipoleMoment output) : nA·µm
  FourSphereVolumeConductor input p : nA·µm
  FourSphereVolumeConductor input dipole_location : µm
  FourSphereVolumeConductor output : mV
  Saved / plotted ABR : µV  (× 1000 from mV)
"""

import os
import re
import sys

import numpy as np
import h5py
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI

import LFPy
import hybridLFPy

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOC_FILE  = os.path.join(REPO_ROOT, 'MSO_models', 'mso_model.hoc')

sys.path.insert(0, os.path.join(REPO_ROOT, 'LFP_reconstruction'))
from main_reconstruct import (
    MSOPopulation, N_MSO_TOTAL,
    ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y,
    LAYER_BOUNDARIES, _pic_stem, _extract_spikes,
)

# ---------------------------------------------------------------------------
# Simulation parameters (identical to LFP pipeline)
# ---------------------------------------------------------------------------
DT     = 0.026   # ms
TSTOP  = 50.0    # ms
V_INIT = -57.0   # mV
N_CELLS = 15500

# ---------------------------------------------------------------------------
# ABR bandpass filter (clinical standard: 150–3000 Hz)
# ---------------------------------------------------------------------------
def _bandpass(signal, lo=150., hi=None, fs=None, order=4):
    """Zero-phase Butterworth high-pass (hi=None) or bandpass filter, row-wise."""
    if fs is None:
        fs = 1.0 / (DT * 1e-3)
    if hi is None:
        sos = scipy.signal.butter(order, lo, btype='high', fs=fs, output='sos')
    else:
        sos = scipy.signal.butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    if signal.ndim == 1:
        return scipy.signal.sosfiltfilt(sos, signal)
    return np.stack([scipy.signal.sosfiltfilt(sos, row) for row in signal])


# ---------------------------------------------------------------------------
# 4-sphere head model parameters
# All spatial units: µm  (lfpykit-native)
# ---------------------------------------------------------------------------
FOUR_SPHERE_RADII  = [79_000., 80_000., 85_000., 90_000.]  # µm: brain,CSF,skull,scalp
FOUR_SPHERE_SIGMAS = [0.33, 1.79, 0.008, 0.3]              # S/m

# MSO dipole source position in head-centred coordinates (µm).
#
# MNI152 centroids (from MRI):
#   R MSO MNI = [+5, -32, -35] mm,  L MSO MNI = [-5, -32, -35] mm
#
# MNI -> head-centred (rigid shift, no rotation):
#   head_centre_MNI ~= [0, -18.3, +5.5] mm  (Koessler et al. 2009 Cz anchor)
#   head_y = -32 + 18.3 = -13.7 mm,  head_z = -35 - 5.5 = -40.5 mm
# |r| ~= 43.0 mm < 79 mm brain sphere    Uncertainty: +-5 mm
# See ABR_reconstruction/plot_mso_position.py for visualisation.
#
# Previous estimate (rigid shift only):
# MSO_POS_UM = {
#     'R': np.array([ 5_000., -13_700., -40_500.]),
#     'L': np.array([-5_000., -13_700., -40_500.]),
# }
MSO_POS_UM = {
    'R': np.array([ 5_000., -18_700., -29_520.]),
    'L': np.array([-5_000., -18_700., -29_520.]),
}


_SCALP_R = 89_999.   # µm — electrodes must be strictly inside scalp (r < r_scalp)
# MUST be changes IF we change FOUR_SPHERE_RADII[3] (scalp radius)


def _on_scalp(v):
    """Project vector to just inside the scalp surface."""
    return v / np.linalg.norm(v) * _SCALP_R


# Scalp electrode positions (µm). FourSphereVolumeConductor requires r < r_scalp.
# Clinical 10-20 system: Cz = vertex, M1 = left mastoid, M2 = right mastoid.
# Standard ABR derivation: Cz−M1 (ipsilateral left) / Cz−M2 (ipsilateral right),
# vertex-positive upward. Ground electrode Fz is not modelled (no effect in 4-sphere).
ELECTRODE_POS = {
    'Cz': np.array([0., 0., _SCALP_R]),
    'M1': _on_scalp(np.array([-74_300., -42_200., -28_100.])),  # left mastoid
    'M2': _on_scalp(np.array([ 74_300., -42_200., -28_100.])),  # right mastoid
}

# ---------------------------------------------------------------------------
# Rotation matrices: model axes → head-centred axes
#
# MSO HOC (from LAYER_BOUNDARIES): medial dend z>0, lateral dend z<0.
# In head coords: x = left(−)/right(+), y = post(−)/ant(+), z = inf(−)/sup(+).
#
# head_x: medial-lateral axis
#   Right MSO: medial dend (+model_z) → toward midline → −head_x  (head_x = −model_z)
#   Left  MSO: medial dend (+model_z) → toward midline → +head_x  (head_x = +model_z)
#
# head_y: anterior-posterior axis (tonotopic: low CF posterior, high CF anterior)
#   Literature: lowest CF in posterior MSO, highest CF in anterior MSO.
#   model_x increases = higher CF = more anterior = higher head_y → head_y = +model_x
#   Same sign for both sides (tonotopy runs the same anatomical direction bilaterally).
#
# head_z: determined by right-hand rule (cross product of rows 0 and 1, det = +1)
#   R_R: [0,0,−1] × [+1,0,0] = (0,−1, 0)  → head_z = −model_y
#   R_L: [0,0,+1] × [+1,0,0] = (0,+1, 0)  → head_z = +model_y
# ---------------------------------------------------------------------------
ROTATION = {
    'R': np.array([[ 0., 0.,-1.],
                   [ 1., 0., 0.],
                   [ 0.,-1., 0.]]),
    'L': np.array([[ 0., 0., 1.],
                   [ 1., 0., 0.],
                   [ 0., 1., 0.]]),
}


def _side_condition(condition, side):
    """Return the per-MSO acoustic condition given the stimulated ear and MSO side.

    condition : 'binaural' | 'left_ear' | 'right_ear'
    side      : 'L' | 'R'
    returns   : 'binaural' | 'ipsilateral' | 'contralateral'
    """
    if condition == 'binaural':
        return 'binaural'
    if (condition == 'left_ear' and side == 'L') or \
       (condition == 'right_ear' and side == 'R'):
        return 'ipsilateral'
    return 'contralateral'


# ---------------------------------------------------------------------------
# Per-side simulation
# ---------------------------------------------------------------------------
def _run_one_side(side, args, meta):
    """
    Simulate MSO population for one brain side.
    Returns population dipole (3, T) nA·µm (only meaningful on rank 0)
    and the output directory path.
    """
    from lfpykit import CurrentDipoleMoment

    contra_side = 'R' if side == 'L' else 'L'
    X_pops = [f'SBC_{contra_side}', f'SBC_{side}',
               f'MNTBC_{side}', f'LNTBC_{side}']
    k_yxl_local = [
        [3, 0, 0, 0],   # medial dendrite:  3 × SBC_contra
        [0, 3, 0, 0],   # lateral dendrite: 3 × SBC_ipsi
        [0, 0, 2, 1],   # soma:             2 × MNTBC + 1 × LNTBC
    ]
    side_cond = _side_condition(args.condition, side)
    if side_cond == 'contralateral':
        k_yxl_local[1] = [0, 0, 0, 0]   # silence lateral dend (ipsi SBC off)
        k_yxl_local[2] = [0, 0, 2, 0]   # silence LNTBC (ipsi inhibition off)
    elif side_cond == 'ipsilateral':
        k_yxl_local[0] = [0, 0, 0, 0]   # silence medial dend (contra SBC off)
        k_yxl_local[2] = [0, 0, 0, 1]   # silence MNTBC (contra inhibition off)
    j_yx_local    = [0.055, 0.055, 0.025, 0.025]
    tau_yx_local  = [0.2,   0.2,   0.4,   0.4  ]
    syn_delay_loc = [2.0,   2.0,   1.0,   1.0  ]

    pic_file   = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                               'baseline_simulation.pic')
    stem       = _pic_stem(pic_file)
    spikes_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'spikes_{stem}_angle{args.angle}_{side}')
    cond_tag   = f'_{args.condition}' if args.condition != 'binaural' else ''
    output_dir = os.path.join(REPO_ROOT, 'RESULTS', 'abr_tmp',
                              f'output_{stem}_angle{args.angle}_{side}{cond_tag}')
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

    # CurrentDipoleMoment(None): cell=None is valid; hybridLFPy sets probe.cell
    # before calling get_transformation_matrix() inside cellsim().
    probe     = CurrentDipoleMoment(None)
    pop_label = f'MSO_{side}'

    pop = MSOPopulation(
        n_syn_per_pop=n_syn_per_pop,
        y=pop_label,
        cellParams={
            'morphology': HOC_FILE, 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP, 'nsegs_method': None,
        },
        rand_rot_axis=['z'],
        simulationParams={'rec_imem': True},
        populationParams={
            'number':             args.n_cells,
            'radius':             ELLIPSE_RADIUS_Y,
            'radius_x':           ELLIPSE_RADIUS_X,
            'radius_y':           ELLIPSE_RADIUS_Y,
            'z_min': -100.0, 'z_max': 100.0,
            'min_cell_interdist': 1.0,
            'min_r':              np.array([[0.], [0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=[probe],
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=42,
        X=X_pops,
        networkSim=networkSim,
        k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=syn_delay_loc,
        synDelayScale=[None] * 4,
        J_yX=j_yx_local,
        tau_yX=tau_yx_local,
    )

    pop.run()   # calls COMM.Barrier() internally
    COMM.Barrier()

    # Sum per-cell dipole moments on this rank.
    # pop.output[i]['CurrentDipoleMoment'] has shape (3, T) float32, nA·µm.
    # Must grab BEFORE any collect_data() call (not needed here, but for safety).
    local_dipole = None
    for i in pop.RANK_CELLINDICES:
        d = pop.output[i]['CurrentDipoleMoment'].astype(np.float64)  # (3, T)
        if local_dipole is None:
            local_dipole = d.copy()
        else:
            local_dipole += d

    if local_dipole is None:   # rank has no cells (edge case)
        n_t = int(round(TSTOP / DT)) + 1
        local_dipole = np.zeros((3, n_t), dtype=np.float64)

    # MPI reduce: sum across all ranks → rank 0 holds population dipole
    global_dipole = np.zeros_like(local_dipole)
    COMM.Reduce(local_dipole, global_dipole, op=MPI.SUM, root=0)
    COMM.Barrier()

    return global_dipole, output_dir   # global_dipole only valid on rank 0


# ---------------------------------------------------------------------------
# Head model projection + saving (rank 0 only)
# ---------------------------------------------------------------------------
def _project_and_save(side, p_model, output_dir):
    """
    Apply rotation and 4-sphere head model. Returns V_uV (n_e, T) and
    saves population_dipole.h5 and ABR.h5.
    """
    from lfpykit.eegmegcalc import FourSphereVolumeConductor

    # --- Rotate: model axes → head-centred axes ----
    R = ROTATION[side]
    p_head = R @ p_model    # (3, T), nA·µm, head-centred

    # --- Save population dipole ----
    dipole_path = os.path.join(output_dir, 'population_dipole.h5')
    srate = 1.0 / (DT * 1e-3)   # Hz
    with h5py.File(dipole_path, 'w') as f:
        f.create_dataset('data',  data=p_head)    # (3, T), nA·µm
        f.create_dataset('srate', data=srate)
        f.attrs['axes'] = 'x=mediolateral, y=anteroposterior, z=inferosuperior'
        f.attrs['units'] = 'nA·µm'
    print(f'Population dipole saved → {dipole_path}')

    return p_head, srate


def _apply_head_model(p_head_by_side, output_dir, electrode_names):
    """
    Project each population's dipole from its own anatomical position and sum
    the resulting scalp potentials (linear superposition):
        Φ_total = Σ_pop  L(r_pop) · p_pop
    Returns V_uV (n_e, T) in µV.
    """
    from lfpykit.eegmegcalc import FourSphereVolumeConductor

    # Electrode array (n_e, 3) µm — shared across all populations
    r_elec = np.stack([ELECTRODE_POS[e] for e in electrode_names])

    fsc = FourSphereVolumeConductor(
        r_electrodes=r_elec,
        radii=FOUR_SPHERE_RADII,
        sigmas=FOUR_SPHERE_SIGMAS,
    )

    # Project each population from its own MSO position and accumulate
    V_mV = None
    for side, p_head in p_head_by_side.items():
        r_dipole = MSO_POS_UM[side]          # anatomical position of this population
        V_side   = fsc.get_dipole_potential(p_head, r_dipole)   # (n_e, T) mV
        V_mV     = V_side if V_mV is None else V_mV + V_side

    V_uV = _bandpass(V_mV * 1e3, hi=3000., lo=150.)  # mV → µV, high-pass 150 Hz

    # for bandpass filter OFF:
    # V_uV = V_mV * 1e3

    srate = 1.0 / (DT * 1e-3)
    abr_path = os.path.join(output_dir, 'ABR.h5')
    with h5py.File(abr_path, 'w') as f:
        f.create_dataset('data',  data=V_uV)    # (n_e, T) µV
        f.create_dataset('srate', data=srate)
        f.create_dataset('electrode_names',
                         data=np.array(electrode_names, dtype='S'))
        f.attrs['units'] = 'µV'
    print(f'ABR saved → {abr_path}')

    return V_uV, srate


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_abr(output_dir, V_uV, electrode_names, srate, angle, side, n_cells,
              derivation='Cz-M1'):
    import matplotlib.gridspec as gridspec

    n_t  = V_uV.shape[1]
    tvec = np.arange(n_t) / srate * 1e3   # ms

    has_cz = 'Cz' in electrode_names
    has_m1 = 'M1' in electrode_names

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

    ax_cz  = fig.add_subplot(gs[0, 0])
    ax_m1  = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, :])

    def _plot_single(ax, label, color, title):
        if label in electrode_names:
            idx = electrode_names.index(label)
            ax.plot(tvec, V_uV[idx], color=color, lw=0.9, label=label)
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.set_ylabel('Potential (µV)')
        ax.set_xlabel('Time (ms)')
        ax.set_title(title)
        ax.legend(fontsize=9)

    _plot_single(ax_cz, 'Cz', 'steelblue',
                 f'Cz (vertex)  |  angle {angle}°  |  side {side}  |  N={n_cells}')
    _plot_single(ax_m1, 'M1', 'firebrick', 'M1 (left mastoid)')

    # Compute chosen differential
    cz = V_uV[electrode_names.index('Cz')]
    m1 = V_uV[electrode_names.index('M1')]
    m2 = V_uV[electrode_names.index('M2')]
    if derivation == 'Cz-M1':
        diff, deriv_label = cz - m1, 'Cz−M1'
    elif derivation == 'Cz-M2':
        diff, deriv_label = cz - m2, 'Cz−M2'
    else:
        diff, deriv_label = cz - (m1 + m2) / 2.0, 'Cz−(M1+M2)/2'

    ax_bot.plot(tvec, diff, color='darkorchid', lw=0.9, label=deriv_label)
    ax_bot.axhline(0, color='k', lw=0.4, ls=':')
    ax_bot.set_ylabel('Amplitude (µV)')
    ax_bot.set_xlabel('Time (ms)')
    ax_bot.set_title(f'{deriv_label}  (vertex-positive upward)')
    ax_bot.legend(fontsize=9)

    fig.suptitle('MSO ABR  |  vertex-positive upward',
                 fontsize=11, fontweight='bold')

    fig_path = os.path.join(output_dir, 'figures', 'mso_abr.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'ABR figure saved → {fig_path}')

    # Second figure: derivation only, more square aspect ratio
    fig2, ax2 = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax2.plot(tvec, diff, color='darkorchid', lw=0.9, label=deriv_label)
    ax2.axhline(0, color='k', lw=0.4, ls=':')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude (µV)')
    ax2.set_title(f'{deriv_label}  |  angle {angle}°  |  side {side}  |  N={n_cells}')
    ax2.legend(fontsize=9)
    fig2.suptitle('MSO ABR  |  vertex-positive upward',
                  fontsize=10, fontweight='bold')
    fig2_path = os.path.join(output_dir, 'figures', 'mso_abr_derivation.png')
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    print(f'Derivation figure saved → {fig2_path}')


def _plot_phase_cycle_abr(output_dir, V_uV, electrode_names, srate,
                           stim_freq, angle, side, n_cells, skip_ms=10.0):
    import scipy.signal

    if stim_freq is None:
        print('Stimulus frequency unknown; skipping phase-cycle ABR plot.')
        return

    dt_ms    = 1e3 / srate
    skip_idx = int(skip_ms / dt_ms)
    V_ss     = scipy.signal.detrend(V_uV[:, skip_idx:], axis=1)

    T_int    = max(1, int(round(1e3 / stim_freq / dt_ms)))
    n_cycles = V_ss.shape[1] // T_int
    if n_cycles < 1:
        print('< 1 complete cycle after ramp skip; skipping phase-cycle ABR plot.')
        return

    V_phase = (V_ss[:, :n_cycles * T_int]
               .reshape(V_ss.shape[0], n_cycles, T_int)
               .mean(axis=1))   # (n_e, T_int)
    phase   = np.linspace(0, 1, T_int, endpoint=False)

    colors = ['steelblue', 'firebrick', 'forestgreen']
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i, name in enumerate(electrode_names):
        ax.plot(phase, V_phase[i], label=name, color=colors[i % len(colors)])
    ax.set_xlabel('Cycle phase')
    ax.set_ylabel('ABR (µV)')
    ax.set_title(f'Phase-averaged ABR | {stim_freq:.0f} Hz | angle {angle}° | side {side}')
    ax.legend()
    ax.axhline(0, color='k', lw=0.5, ls='--')

    fig_path = os.path.join(output_dir, 'figures', 'mso_abr_phase_cycle.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Phase-cycle ABR figure saved → {fig_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='MSO ABR reconstruction')
    parser.add_argument('--pic-file',   type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',      type=int, default=0)
    parser.add_argument('--side',       type=str, default='L',
                        choices=['L', 'R', 'both'])
    parser.add_argument('--n-cells',    type=int, default=N_CELLS, dest='n_cells')
    parser.add_argument('--derivation', type=str, default='Cz-M1',
                        choices=['Cz-M1', 'Cz-M2', 'Cz-avg'],
                        help='Differential derivation to plot (Cz-avg = Cz − mean(M1,M2))')
    parser.add_argument('--condition', type=str, default='binaural',
                        choices=['binaural', 'left_ear', 'right_ear'])
    args = parser.parse_args()

    sides = ['L', 'R'] if args.side == 'both' else [args.side]

    p_head_by_side = {}   # side → (3, T) nA·µm in head coords (rank 0 only)
    output_dirs    = {}

    for side in sides:
        # Spike extraction on rank 0, broadcast metadata
        if RANK == 0:
            meta = _extract_spikes(args.angle, side, pic_file=args.pic_file)
        else:
            meta = None
        meta = COMM.bcast(meta, root=0)
        COMM.Barrier()

        global_dipole, output_dir = _run_one_side(side, args, meta)
        output_dirs[side] = output_dir

        if RANK == 0:
            p_head, _ = _project_and_save(side, global_dipole, output_dir)
            p_head_by_side[side] = p_head

    if RANK == 0:
        # For bilateral, create a joint output directory
        if len(sides) > 1:
            pic_file = args.pic_file or os.path.join(
                REPO_ROOT, 'RESULTS', 'baseline_simulation.pic')
            stem = _pic_stem(pic_file)
            cond_tag = f'_{args.condition}' if args.condition != 'binaural' else ''
            joint_dir = os.path.join(REPO_ROOT, 'RESULTS', 'abr_tmp',
                                     f'output_{stem}_angle{args.angle}_both{cond_tag}')
            os.makedirs(os.path.join(joint_dir, 'figures'), exist_ok=True)
            final_dir = joint_dir
        else:
            final_dir = output_dirs[sides[0]]

        electrode_names = ['Cz', 'M1', 'M2']
        V_uV, srate = _apply_head_model(p_head_by_side, final_dir, electrode_names)

        #stim_freq = meta.get('stim_freq_hz')
        _plot_abr(final_dir, V_uV, electrode_names, srate,
                  args.angle, args.side, args.n_cells, derivation=args.derivation)
        #_plot_phase_cycle_abr(final_dir, V_uV, electrode_names, srate,
                              #stim_freq, args.angle, args.side, args.n_cells)


if __name__ == '__main__':
    main()
