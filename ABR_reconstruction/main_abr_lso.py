#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
LSO ABR reconstruction via current-dipole moment + 4-sphere head model.

This is the SPIKING-output ABR generator the scalp BIC reflects (Tolnai et al.):
each LSO projection neuron is fired one AP per NEST-output spike and the AP travels
up the ~4 mm ascending-lateral-lemniscus active axon as a moving current dipole
(main_reconstruct_lso.LSOSpikingPopulation + MSO_models/lso_model_active_axon.hoc).
Contrast with main_abr.py (MSO, synaptic dipole) — the two are the computational
ablations compared in plot_bic_sweep.py.

Mirrors main_abr_avcn.py: own anatomical position (LSO_POS_UM) + rotation, the
whole-cell dipole (soma+dendrites+axon travelling wave) projected ONCE per cell
through the 4-sphere model (NOT split synaptic/axonal — see main_abr_avcn.py:98).

CLI (single or MPI):
  python ABR_reconstruction/main_abr_lso.py --pic-file RESULTS/x.pic --angle 0 --side both --n-cells 200
  mpiexec -n 4 python ABR_reconstruction/main_abr_lso.py --itd-us 500 --side both --n-cells 200

Condition selection (mutually exclusive): --angle (default) | --itd-us | --ild-db.

Outputs:
  RESULTS/abr_tmp/output_lso_{stem}_{cond}_{side}/population_dipole.h5   (per side)
  RESULTS/abr_tmp/output_lso_{stem}_{cond}_{sidespec}/ABR.h5            (Cz/M1/M2, µV)
  .../figures/lso_abr.png
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

# LSO spiking population + geometry (mechanisms loaded on import)
sys.path.insert(0, os.path.join(REPO_ROOT, 'LFP_reconstruction'))
from main_reconstruct_lso import (          # noqa: E402
    LSOPopulation, LSOSpikingPopulation, HOC_FILE, HOC_FILE_AXON,
    ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y, LAYER_BOUNDARIES,
    DT, TSTOP, V_INIT, N_LSO_TOTAL,
)
from main_reconstruct import _pic_stem, _extract_spikes   # noqa: E402

# Shared head-model constants + band-pass (reuse MSO ABR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_abr import (                        # noqa: E402
    _bandpass, FOUR_SPHERE_RADII, FOUR_SPHERE_SIGMAS, ELECTRODE_POS,
    save_dipole_record,
)

# ---------------------------------------------------------------------------
# LSO dipole source position in head-centred coordinates (µm).
#
# The human LSO sits in the superior olivary complex, LATERAL to the MSO
# (MSO_POS_UM ≈ ±5 mm x). First-pass estimate — more lateral (±9 mm), same
# rostrocaudal/vertical ballpark as the MSO; |r| stays inside the 79 mm brain
# sphere. Refine from MRI/atlas as was done for MSO_POS_UM. x = mediolateral (±),
# y = anteroposterior (ant +), z = inferosuperior.
# ---------------------------------------------------------------------------
# The LSO sits POSTERIOR and LATERAL to the MSO (MSO_POS_UM y=-18700). We place it
# ~1.8 mm posterior (more -head_y) and more lateral (x=+-9 mm vs MSO +-5). Exact
# rostrocaudal/vertical offset refinable from atlas. x=mediolat(+-), y=post(-)/ant(+),
# z=inf(-)/sup(+).
LSO_POS_UM = {
    'R': np.array([ 9_000., -20_500., -30_000.]),
    'L': np.array([-9_000., -20_500., -30_000.]),
}

# ---------------------------------------------------------------------------
# Rotation: model axes -> head-centred axes, PER SIDE.
#
# The extended LSO axon runs along AXON_DIR (rostro-dorsal, +x dorsal +y rostral;
# build_lso_axon.py); the ascending LL climbs toward the IC = +head_z (superior).
# Both sides map AXON_DIR -> +head_z (axons ascend on both sides). The mediolateral
# axis is mirrored per side so "medial" (+model_z) always points toward the midline:
#   Right LSO (at +head_x): medial -> -head_x   (R @ model_z = -head_x, s=-1)
#   Left  LSO (at -head_x): medial -> +head_x   (R @ model_z = +head_x, s=+1)
# Consequence: head_z (axon->IC) is IDENTICAL both sides; head_x (mediolateral) AND
# head_y (anteroposterior) mirror. This differs from the MSO (which mirrors head_x+
# head_z, keeps head_y) because here head_z is the axon axis and must not flip.
# This keeps tonotopy (mediolateral z) anatomically consistent bilaterally
# (lateral=low CF, medial=high CF). Cells are NOT randomly rotated (rand_rot_axis=[]).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(REPO_ROOT, 'MSO_models'))
from build_lso_axon import AXON_DIR    # noqa: E402  model-frame axon unit vector


def _lso_rotation(a, side):
    """Proper rotation (det=+1): R @ a = +head_z; R @ model_z = -/+head_x for R/L.

    ey = ez x ex with orthonormal {ex,ez} is right-handed -> det = +1 by construction.
    """
    ez = np.asarray(a, float); ez = ez / np.linalg.norm(ez)
    s  = -1.0 if side == 'R' else 1.0                     # mirror mediolateral per side
    ex = np.array([0., 0., 1.]) - np.array([0., 0., 1.]).dot(ez) * ez  # model z ⟂ ez
    if np.linalg.norm(ex) < 1e-6:                          # a ∥ z fallback
        ex = np.array([1., 0., 0.]) - np.array([1., 0., 0.]).dot(ez) * ez
    ex = s * ex / np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    return np.vstack([ex, ey, ez])


ROTATION = {'R': _lso_rotation(AXON_DIR, 'R'),
            'L': _lso_rotation(AXON_DIR, 'L')}


# ---------------------------------------------------------------------------
# Condition selection (angle | itd-us | ild-db) -> (.pic key, readable label)
# ---------------------------------------------------------------------------
def _condition(args):
    if getattr(args, 'ild_db', None) is not None:
        return float(args.ild_db), f'ild{args.ild_db:g}dB'
    if getattr(args, 'itd_us', None) is not None:
        return args.itd_us * 1e-6, f'itd{args.itd_us:g}us'
    return args.angle, f'angle{args.angle}'


# ---------------------------------------------------------------------------
# Per-side simulation -> population dipole (3, T) nA·µm (rank 0)
# ---------------------------------------------------------------------------
def _side_condition(condition, side):
    """left_ear/right_ear -> per-side {ipsilateral,contralateral} (as main_abr.py).
    For side L the ipsi ear is 'left'; for side R the ipsi ear is 'right'."""
    if condition == 'binaural':
        return 'binaural'
    if (condition == 'left_ear' and side == 'L') or \
       (condition == 'right_ear' and side == 'R'):
        return 'ipsilateral'
    return 'contralateral'


def _run_one_side(side, args, meta, cond_label):
    from lfpykit import CurrentDipoleMoment
    spiking = (args.drive == 'spiking')

    if spiking:
        pop_class     = LSOSpikingPopulation
        hoc_file      = HOC_FILE_AXON
        rot_axis      = []                   # coherent rostro-dorsal volley
        X_pops        = [f'LSO_{side}']
        k_yxl_local   = [[1]]                # single suprathreshold synapse -> AIS
        syn0          = LSOSpikingPopulation.PER_POP_SYN['LSO']
        j_yx_local, tau_yx_local = [syn0['weight']], [syn0['tau2']]
        syn_delay_loc, syn_delay_scale = [0.05], [None]
    else:
        # SYNAPTIC LSO scalp dipole (analog of the MSO ABR): integrate SBC/MNTBC
        # currents; monaural via ear-specific silencing (single-layer k_yXL).
        pop_class     = LSOPopulation
        hoc_file      = HOC_FILE
        rot_axis      = []
        X_pops        = [f'SBC_{side}', f'MNTBC_{side}']
        k_yxl_local   = [[40, 8]]            # SBC (dends) + MNTBC (soma), one layer
        sc = _side_condition(args.condition, side)
        if sc == 'ipsilateral':             # ipsi ear -> keep SBC, drop MNTBC
            k_yxl_local = [[40, 0]]
        elif sc == 'contralateral':         # contra ear -> drop SBC, keep MNTBC
            k_yxl_local = [[0, 8]]
        j_yx_local    = [0.040, 0.020]
        tau_yx_local  = [1.0,   0.7]
        syn_delay_loc, syn_delay_scale = [2.0, 0.78], [None, None]

    pic_file   = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                               'baseline_simulation.pic')
    stem       = _pic_stem(pic_file)
    cond, _    = _condition(args)
    spikes_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'spikes_{stem}_angle{cond}_{side}')
    drive_tag  = '' if spiking else '_syn'
    cond_tag   = '' if args.condition == 'binaural' else f'_{args.condition}'
    output_dir = os.path.join(REPO_ROOT, 'RESULTS', 'abr_tmp',
                              f'output_lso_{stem}_{cond_label}_{side}{drive_tag}{cond_tag}')
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    k_arr = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT,
        spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops,
    )

    probe     = CurrentDipoleMoment(None)   # cell set by hybridLFPy in cellsim()
    pop_label = f'LSO_{side}'

    pop = pop_class(
        n_syn_per_pop=n_syn_per_pop,
        y=pop_label,
        cellParams={
            'morphology': hoc_file, 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP, 'nsegs_method': None,
        },
        rand_rot_axis=rot_axis,
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   args.n_cells,
            'radius':   ELLIPSE_RADIUS_Y,
            'radius_x': ELLIPSE_RADIUS_X,
            'radius_y': ELLIPSE_RADIUS_Y,
            'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=[probe],
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=43,
        X=X_pops,
        networkSim=networkSim,
        k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=syn_delay_loc,
        synDelayScale=syn_delay_scale,
        J_yX=j_yx_local,
        tau_yX=tau_yx_local,
    )

    pop.run()
    COMM.Barrier()

    # Sum per-cell whole-cell dipole moments (3, T) nA·µm on this rank.
    local = None
    for i in pop.RANK_CELLINDICES:
        d = pop.output[i]['CurrentDipoleMoment'].astype(np.float64)
        local = d.copy() if local is None else local + d
    if local is None:
        n_t = int(round(TSTOP / DT)) + 1
        local = np.zeros((3, n_t), dtype=np.float64)
    glob = np.zeros_like(local)
    COMM.Reduce(local, glob, op=MPI.SUM, root=0)
    COMM.Barrier()
    return glob, output_dir


# The long (~4 mm) active axon makes finitialize's settling capacitive currents,
# times the large lever arm, a big SPURIOUS dipole at t≈0 that the zero-phase
# bandpass rings off. NEST LSO spikes only start >6 ms, so 0–SETTLE_MS is
# physiologically silent — blank it at the source before projection/filtering.
SETTLE_MS = 3.0


def _project_and_save(side, p_model, output_dir):
    """Rotate model->head coords, save population_dipole.h5. Returns p_head."""
    n_settle = int(round(SETTLE_MS / DT))
    p_model = p_model.copy()
    p_model[:, :n_settle] = 0.0
    p_head = ROTATION[side] @ p_model
    srate  = 1.0 / (DT * 1e-3)
    with h5py.File(os.path.join(output_dir, 'population_dipole.h5'), 'w') as f:
        f.create_dataset('data',  data=p_head)
        f.create_dataset('srate', data=srate)
        f.attrs['axes']  = 'x=mediolateral, y=anteroposterior, z=inferosuperior'
        f.attrs['units'] = 'nA·µm'
    return p_head


def _apply_head_model(p_head_by_side, output_dir, electrode_names):
    """Project each side's LSO dipole from its own position, SUM scalp potentials."""
    from lfpykit.eegmegcalc import FourSphereVolumeConductor

    r_elec = np.stack([ELECTRODE_POS[e] for e in electrode_names])
    fsc = FourSphereVolumeConductor(r_electrodes=r_elec,
                                    radii=FOUR_SPHERE_RADII,
                                    sigmas=FOUR_SPHERE_SIGMAS)
    V_mV = None
    for side, p_head in p_head_by_side.items():
        V_side = fsc.get_dipole_potential(p_head, LSO_POS_UM[side])   # (n_e, T) mV
        V_mV   = V_side if V_mV is None else V_mV + V_side

    V_uV  = _bandpass(V_mV * 1e3, hi=1500., lo=100.)   # Tolnai ABR band 100-1500 Hz
    srate = 1.0 / (DT * 1e-3)
    with h5py.File(os.path.join(output_dir, 'ABR.h5'), 'w') as f:
        f.create_dataset('data',  data=V_uV)
        f.create_dataset('srate', data=srate)
        f.create_dataset('electrode_names', data=np.array(electrode_names, dtype='S'))
        f.attrs['units'] = 'µV'
    print(f'ABR saved → {os.path.join(output_dir, "ABR.h5")}')
    return V_uV, srate


def _plot_abr(output_dir, V_uV, electrode_names, srate, cond_label, side, n_cells,
              derivation='Cz-M1'):
    n_t  = V_uV.shape[1]
    tvec = np.arange(n_t) / srate * 1e3
    idx  = {e: electrode_names.index(e) for e in electrode_names}
    cz, m1, m2 = V_uV[idx['Cz']], V_uV[idx['M1']], V_uV[idx['M2']]
    diff, lbl = (cz - m2, 'Cz−M2') if derivation == 'Cz-M2' else \
                (cz - (m1 + m2) / 2., 'Cz−(M1+M2)/2') if derivation == 'Cz-avg' else \
                (cz - m1, 'Cz−M1')

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    ax0.plot(tvec, cz, color='steelblue', lw=0.9, label='Cz')
    ax0.plot(tvec, m1, color='firebrick', lw=0.7, label='M1')
    ax0.axhline(0, color='k', lw=0.4, ls=':')
    ax0.set_ylabel('Potential (µV)')
    ax0.set_title(f'LSO ABR (spiking) | {cond_label} | side {side} | N={n_cells}')
    ax0.legend(fontsize=9)

    ax1.plot(tvec, diff, color='darkorchid', lw=1.0, label=lbl)
    ax1.axhline(0, color='k', lw=0.4, ls=':')
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'{lbl}  (vertex-positive upward)')
    ax1.legend(fontsize=9)

    path = os.path.join(output_dir, 'figures', 'lso_abr.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'ABR figure saved → {path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='LSO (spiking) ABR reconstruction')
    parser.add_argument('--pic-file',   type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',      type=int, default=0)
    parser.add_argument('--itd-us',     type=float, default=None, dest='itd_us',
                        help='Select an artificial-ITD condition (µs); key = µs*1e-6 s.')
    parser.add_argument('--ild-db',     type=float, default=None, dest='ild_db',
                        help='Select an artificial-ILD condition (dB); key = dB.')
    parser.add_argument('--side',       type=str, default='L',
                        choices=['L', 'R', 'both'])
    parser.add_argument('--n-cells',    type=int, default=200, dest='n_cells')
    parser.add_argument('--derivation', type=str, default='Cz-M1',
                        choices=['Cz-M1', 'Cz-M2', 'Cz-avg'])
    parser.add_argument('--drive', type=str, default='spiking',
                        choices=['spiking', 'synaptic'],
                        help='spiking=LSO output train up the LL axon (travelling-wave '
                             'dipole; monaural needs monaural pics); synaptic=SBC/MNTBC '
                             'synaptic dipole -> scalp (analog of the MSO ABR; monaural '
                             'via --condition, like main_abr.py)')
    parser.add_argument('--condition', type=str, default='binaural',
                        choices=['binaural', 'left_ear', 'right_ear'],
                        help='synaptic drive only: acoustic condition for the BIC '
                             '(binaural / left-ear-only / right-ear-only)')
    args = parser.parse_args()

    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    cond, cond_label = _condition(args)
    stem = _pic_stem(args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                                   'baseline_simulation.pic'))

    p_head_by_side, output_dirs = {}, {}
    for side in sides:
        if RANK == 0:
            meta = _extract_spikes(cond, side, pic_file=args.pic_file)
        else:
            meta = None
        meta = COMM.bcast(meta, root=0)
        COMM.Barrier()

        p_model, output_dir = _run_one_side(side, args, meta, cond_label)
        output_dirs[side] = output_dir
        if RANK == 0:
            p_head_by_side[side] = _project_and_save(side, p_model, output_dir)
            save_dipole_record(stem, args.angle, 'LSO', args.drive, side,
                               p_head_by_side[side], LSO_POS_UM[side],
                               N_LSO_TOTAL, args.n_cells, condition=args.condition)

    if RANK == 0:
        pic_file = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                                 'baseline_simulation.pic')
        stem = _pic_stem(pic_file)
        if len(sides) > 1:
            drive_tag = '' if args.drive == 'spiking' else '_syn'
            cond_tag  = '' if args.condition == 'binaural' else f'_{args.condition}'
            final_dir = os.path.join(REPO_ROOT, 'RESULTS', 'abr_tmp',
                                     f'output_lso_{stem}_{cond_label}_both{drive_tag}{cond_tag}')
            os.makedirs(os.path.join(final_dir, 'figures'), exist_ok=True)
        else:
            final_dir = output_dirs[sides[0]]

        electrode_names = ['Cz', 'M1', 'M2']
        V_uV, srate = _apply_head_model(p_head_by_side, final_dir, electrode_names)
        _plot_abr(final_dir, V_uV, electrode_names, srate,
                  cond_label, args.side, args.n_cells, derivation=args.derivation)


if __name__ == '__main__':
    main()
