#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
HybridLFPy LFP reconstruction for LSO population.

CLI usage (single or multi-process):
  python LFP_reconstruction/main_reconstruct_lso.py [options]
  mpiexec -n 4 python LFP_reconstruction/main_reconstruct_lso.py --angle 0 --side L --n-cells 200

Options:
  --pic-file FILE   Path to .pic simulation result (default: RESULTS/baseline_simulation.pic)
  --angle DEGREES   Sound azimuth angle (default: 0)
  --side  L|R       Brain side (default: L)
  --n-cells N       LSO cells to simulate (default: 100)
  --n-single N      Single-cell contribution plots to save (default: 5)

LSO connectivity (per BrainstemModel.py):
  SBC_{side}   -> LSO_{side}  excitatory, ipsilateral, 40 synapses/cell
  MNTBC_{side} -> LSO_{side}  inhibitory, ipsilateral,  8 synapses/cell

Reuses presynaptic GDF files produced by main_reconstruct.py for the same
pic/angle/side (spikes_{stem}_angle{ANGLE}_{SIDE}/).

Outputs saved to RESULTS/lfp_tmp/output_lso_{stem}_angle{ANGLE}_{SIDE}/figures/:
  lso_lfp_reconstruction.png
  lso_lfp_single_cells.png
"""

import os
import sys
import random

import numpy as np
import h5py

import neuron
import lfpykit.models as lfpykit_models
import hybridLFPy
from hybridLFPy.population import Population
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOC_FILE  = os.path.join(REPO_ROOT, 'MSO_models', 'lso_model_active.hoc')
# Extended-axon morphology for the SPIKING driver (--drive spiking): soma+dends
# of lso_model_active.hoc + ~4 mm ascending-LL active axon (build_lso_axon.py).
HOC_FILE_AXON = os.path.join(REPO_ROOT, 'MSO_models', 'lso_model_active_axon.hoc')
try:
    neuron.load_mechanisms(os.path.join(REPO_ROOT, 'MSO_models'))
except RuntimeError as _e:
    if 'already exists' not in str(_e):
        raise

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DT     = 0.026   # ms
TSTOP  = 50.0    # ms
V_INIT = -63.0   # mV  (E_L.LSO from params.py)

N_LSO_TOTAL   = 5600   # LSO neurons per side in NEST sim

# ---------------------------------------------------------------------------
# LSO nucleus geometry — HUMAN dims, anatomically reoriented (Addendum A).
# Model axes: x=dorsoventral, y=rostrocaudal, z=mediolateral.
#   HEIGHT along y (rostrocaudal) = 2800 um (the LSO long axis);
#   elliptic cross-section in the x-z plane:
#     x (dorsoventral) semi-axis 400 um,  z (mediolateral / TONOTOPIC) semi-axis 600 um.
# Tonotopy runs mediolaterally (z): lateral = low CF, medial = high CF.
# ---------------------------------------------------------------------------
HALF_HEIGHT_Y = 1400.0   # um  rostrocaudal half-height (full 2800 um)
RADIUS_X      = 400.0    # um  dorsoventral semi-axis
RADIUS_Z      = 600.0    # um  mediolateral / tonotopic semi-axis
# Back-compat aliases (populationParams keys only; draw_rand_pos below uses the
# constants above directly, so the passed radius_x/radius_y are not authoritative).
ELLIPSE_RADIUS_X = RADIUS_X
ELLIPSE_RADIUS_Y = RADIUS_Z

# ---------------------------------------------------------------------------
# Probe geometry  (same layout as MSO script)
# ---------------------------------------------------------------------------
N_CH    = 16
PROBE_Z = np.linspace(-500, 500, N_CH)   # um  (brackets the +-413.5 um dendrite tips)
PROBE_X = np.zeros(N_CH)
PROBE_Y = np.zeros(N_CH)
SIGMA   = 0.3   # S/m

# ---------------------------------------------------------------------------
# Layer boundaries — SINGLE all-encompassing layer.
#
# hybridLFPy assigns synapses by ABSOLUTE z (a cortical-depth model) AFTER the cell
# is positioned. Since the reoriented nucleus distributes somas along z (tonotopic,
# +-RADIUS_Z), a narrow per-structure z-layer would miss most cells (get_idx -> empty
# -> no synapse -> no drive). We don't need z-layers: insert_all_synapses places every
# synapse by SECTION NAME (SBC->dendrites, MNTBC->soma, spiking driver->AIS), so one
# layer spanning all z works for any soma position. k_yXL is therefore single-row.
# ---------------------------------------------------------------------------
LAYER_BOUNDARIES = [
    [-1.0e4, 1.0e4],
]


# ---------------------------------------------------------------------------
# LSOPopulation subclass
# ---------------------------------------------------------------------------
class LSOPopulation(Population):
    """Population subclass for LSO: tonotopic presynaptic assignment + Exp2Syn params."""

    PER_POP_SYN = {
        'SBC': {
            'syntype': 'Exp2Syn',
            'tau1':    0.2,    # ms  TAUS_EX_RISE.LSO
            'tau2':    0.5,    # ms  TAUS_EX_DECAY.LSO
            'e':       0.0,    # mV  EXC_REV.LSO
            'weight':  0.040,  # uS
        },
        'MNTBC': {
            'syntype': 'Exp2Syn',
            'tau1':    0.2,    # ms  TAUS_IN_RISE.LSO
            'tau2':    0.5,    # ms  TAUS_IN_DECAY.LSO
            'e':      -90.0,   # mV  INH_REV.LSO
            'weight':  0.020,  # uS
        },
    }

    def __init__(self, n_syn_per_pop=None, **kwargs):
        self.n_syn_per_pop = n_syn_per_pop or {}
        super().__init__(**kwargs)

    def get_all_SpCells(self):
        """Tonotopic x_to_one assignment mirroring NEST connector for LSO."""
        n_cells = self.POPULATION_SIZE
        SpCells  = {}

        for cellindex in self.RANK_CELLINDICES:
            lso_idx = (int(round(cellindex * (N_LSO_TOTAL - 1) / (n_cells - 1)))
                       if n_cells > 1 else 0)

            SpCells[cellindex] = {}
            for X in self.X:
                nodes  = self.networkSim.nodes[X]
                N_pre  = len(nodes)
                n_src  = self.n_syn_per_pop.get(X, 1)

                step      = (N_pre - n_src) / max(N_LSO_TOTAL - 1, 1)
                pre_start = min(int(round(lso_idx * step)), N_pre - n_src)
                window    = nodes[pre_start : pre_start + n_src]

                SpCell   = []
                src_used = 0
                for compartments in self.synIdx[cellindex][X]:
                    size = len(compartments)
                    if size > 0:
                        SpCell.append(
                            window[src_used : src_used + size].astype('int32'))
                        src_used += size
                    else:
                        SpCell.append(np.array([], dtype='int32'))
                SpCells[cellindex][X] = SpCell

        return SpCells

    def insert_all_synapses(self, cellindex, cell):
        dend_segs = np.concatenate([
            cell.get_idx('dend_A'),
            cell.get_idx('dend_B'),
            cell.get_idx('dend_C'),
        ])
        soma_segs = cell.get_idx('soma')

        for X in self.X:
            pop_type = X.rsplit('_', 1)[0]
            for j in range(len(self.synIdx[cellindex][X])):
                idx = self.synIdx[cellindex][X][j]
                synDelays = (self.synDelays[cellindex][X][j]
                             if self.synDelays is not None else None)

                if len(idx) == 0:
                    continue

                if pop_type == 'SBC' and len(dend_segs) > 0:
                    # Distance-weighted placement on dendrites (distal bias)
                    soma_mid = np.array([cell.x[soma_segs].mean(),
                                         cell.y[soma_segs].mean(),
                                         cell.z[soma_segs].mean()])
                    seg_mids = np.column_stack([
                        cell.x[dend_segs].mean(axis=1),
                        cell.y[dend_segs].mean(axis=1),
                        cell.z[dend_segs].mean(axis=1),
                    ])
                    dist = np.linalg.norm(seg_mids - soma_mid, axis=1)
                    total = dist.sum()
                    weights = dist / total if total > 0 else np.ones(len(dist)) / len(dist)
                    idx = np.random.choice(dend_segs, size=len(idx),
                                           p=weights, replace=True).astype('int32')

                elif pop_type == 'MNTBC' and len(soma_segs) > 0:
                    idx = np.random.choice(soma_segs, size=len(idx),
                                           replace=True).astype('int32')

                self.insert_synapses(
                    cell=cell,
                    cellindex=cellindex,
                    synParams=self.PER_POP_SYN[pop_type].copy(),
                    idx=idx,
                    X=X,
                    SpCell=self.SpCells[cellindex][X][j],
                    synDelays=synDelays,
                )

    def draw_rand_pos(self, min_cell_interdist=1.0, **kwargs):
        """Uniform sampling in the reoriented LSO nucleus (Addendum A): HEIGHT along
        y (rostrocaudal, +-HALF_HEIGHT_Y), elliptic cross-section in the x-z plane
        (semi-axes RADIUS_X dorsoventral, RADIUS_Z mediolateral). The passed
        radius_x/radius_y/z_min/z_max are ignored — geometry comes from the module
        constants. Somas are sorted mediolaterally (z) so the tonotopic assignment
        in get_all_SpCells runs lateral -> medial = low -> high CF."""
        N = self.POPULATION_SIZE

        def _sample(m):
            xx = (np.random.rand(m) - 0.5) * 2 * RADIUS_X
            zz = (np.random.rand(m) - 0.5) * 2 * RADIUS_Z
            yy = (np.random.rand(m) - 0.5) * 2 * HALF_HEIGHT_Y
            return xx, yy, zz

        def _outside(xx, zz):
            return np.where((xx / RADIUS_X)**2 + (zz / RADIUS_Z)**2 > 1)[0]

        x, y, z = _sample(N)
        out = _outside(x, z)
        while len(out):
            x[out], y[out], z[out] = _sample(len(out))
            out = _outside(x, z)

        too_close = np.where(self.calc_min_cell_interdist(x, y, z) < min_cell_interdist)[0]
        while len(too_close):
            x[too_close], y[too_close], z[too_close] = _sample(len(too_close))
            out = _outside(x, z)
            while len(out):
                x[out], y[out], z[out] = _sample(len(out))
                out = _outside(x, z)
            too_close = np.where(self.calc_min_cell_interdist(x, y, z) < min_cell_interdist)[0]

        soma_pos = [{'x': x[i], 'y': y[i], 'z': z[i]} for i in range(N)]
        soma_pos.sort(key=lambda p: p['z'])   # tonotopic along z (mediolateral)
        return soma_pos


# ---------------------------------------------------------------------------
# LSOSpikingPopulation — the SPIKING-output driver (Tolnai-BIC generator)
#
# Instead of integrating SBC/MNTBC synaptic currents, each cell is driven by its
# OWN NEST LSO output train (X = LSO_{side}) through a single SUPRATHRESHOLD
# synapse on the AIS, so it fires one AP per spike; the AP propagates up the
# ~4 mm ascending-LL active axon (lso_model_active_axon.hoc) as a travelling-wave
# current dipole — the integrated spiking output the scalp BIC reflects. Mirrors
# the calyx suprathreshold-drive pattern (main_reconstruct_calyx.CalyxPopulation).
# Reuses LSOPopulation.get_all_SpCells (tonotopic 1-source window) and draw_rand_pos.
# ---------------------------------------------------------------------------
class LSOSpikingPopulation(LSOPopulation):
    """LSO driven suprathreshold by its own output train -> axonal travelling wave."""

    PER_POP_SYN = {
        'LSO': {
            'syntype': 'Exp2Syn',
            'tau1':    0.1,    # ms  fast AMPA-like rise
            'tau2':    0.2,    # ms  fast decay -> a single AP per input
            'e':       0.0,    # mV
            'weight':  0.30,   # uS  suprathreshold (single-cell rheobase ~6 nA)
        },
    }

    def insert_all_synapses(self, cellindex, cell):
        # Single suprathreshold synapse on the AIS (falls back to soma). Firing the
        # AIS both initiates the AP and seeds the saltatory volley up the axon.
        drive_segs = cell.get_idx('ais')
        if len(drive_segs) == 0:
            drive_segs = cell.get_idx('soma')

        for X in self.X:
            pop_type = X.rsplit('_', 1)[0]   # 'LSO'
            for j in range(len(self.synIdx[cellindex][X])):
                idx = self.synIdx[cellindex][X][j]
                synDelays = (self.synDelays[cellindex][X][j]
                             if self.synDelays is not None else None)
                if len(idx) == 0:
                    continue
                if len(drive_segs) > 0:
                    idx = np.random.choice(drive_segs, size=len(idx),
                                           replace=True).astype('int32')
                self.insert_synapses(
                    cell=cell,
                    cellindex=cellindex,
                    synParams=self.PER_POP_SYN[pop_type].copy(),
                    idx=idx,
                    X=X,
                    SpCell=self.SpCells[cellindex][X][j],
                    synDelays=synDelays,
                )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='LSO LFP reconstruction')
    parser.add_argument('--pic-file', type=str,  default=None, dest='pic_file')
    parser.add_argument('--angle',    type=int,  default=0)
    parser.add_argument('--side',     type=str,  default='L', choices=['L', 'R'])
    parser.add_argument('--n-cells',  type=int,  default=N_LSO_TOTAL, dest='n_cells')
    parser.add_argument('--n-single', type=int,  default=5,  dest='n_single')
    parser.add_argument('--condition', type=str, default='binaural',
                        choices=['binaural', 'ipsilateral', 'contralateral'],
                        help='binaural=both; ipsilateral=silence MNTBC; contralateral=silence SBC')
    parser.add_argument('--drive', type=str, default='synaptic',
                        choices=['synaptic', 'spiking'],
                        help='synaptic=integrate SBC/MNTBC currents (near-field LFP, '
                             'default); spiking=drive the LSO output train up the '
                             'extended active axon (travelling-wave dipole, BIC generator)')
    parser.add_argument('--itd-us', type=float, default=None, dest='itd_us',
                        help='Select an artificial-ITD condition (µs). Overrides '
                             '--angle; pic key looked up in seconds (µs*1e-6).')
    parser.add_argument('--ild-db', type=float, default=None, dest='ild_db',
                        help='Select an artificial-ILD condition (dB). Overrides '
                             '--itd-us/--angle; pic key looked up in dB.')
    args = parser.parse_args()

    side = args.side

    pic_file = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                              'baseline_simulation.pic')

    # Stimulus-condition key: --ild-db > --itd-us > --angle (default 0). Same
    # convention as main_reconstruct.py / main_abr.py::_condition.
    if args.ild_db is not None:
        cond_val, cond_label = float(args.ild_db), f'ild{args.ild_db:g}dB'
    elif args.itd_us is not None:
        cond_val, cond_label = args.itd_us * 1e-6, f'itd{args.itd_us:g}us'
    else:
        cond_val, cond_label = args.angle, f'angle{args.angle}'

    # Reuse the same spikes directory as the MSO script (same pops extracted)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main_reconstruct import _pic_stem, _extract_spikes

    if RANK == 0:
        meta = _extract_spikes(cond_val, side, pic_file=pic_file)
    else:
        meta = None
    meta = COMM.bcast(meta, root=0)
    COMM.Barrier()

    if args.drive == 'spiking':
        # Drive each cell suprathreshold from its OWN LSO output train up the
        # extended active axon (travelling-wave dipole). One synapse on the AIS
        # (soma layer, k row 2); insert_all_synapses reassigns it to the AIS.
        pop_class     = LSOSpikingPopulation
        hoc_file      = HOC_FILE_AXON
        rot_axis      = []   # no rotation -> coherent rostro-dorsal volley (axon is
                             #                off-principal-axis, so any spin decoheres it)
        X_pops        = [f'LSO_{side}']
        k_yxl_local   = [[1]]              # 1 synapse (single layer) -> AIS via override
        syn0          = LSOSpikingPopulation.PER_POP_SYN['LSO']
        j_yx_local    = [syn0['weight']]
        tau_yx_local  = [syn0['tau2']]
        syn_delay_loc = [0.05]
        syn_delay_scale = [None]
    else:
        pop_class     = LSOPopulation
        hoc_file      = HOC_FILE
        rot_axis      = ['z']
        X_pops = [f'SBC_{side}', f'MNTBC_{side}']

        # Single layer: 40 SBC (-> dendrites) + 8 MNTBC (-> soma), placed by the
        # section-name override in LSOPopulation.insert_all_synapses.
        k_yxl_local = [[40, 8]]
        if args.condition == 'ipsilateral':
            k_yxl_local = [[40, 0]]        # silence MNTBC (contra inhibition)
        elif args.condition == 'contralateral':
            k_yxl_local = [[0, 8]]         # silence SBC (ipsi excitation)
        j_yx_local      = [0.040, 0.020]
        tau_yx_local    = [1.0,   0.7  ]
        # params.py SYN_DELAYS: SBCs2LSO=2.0, MNTBCs2LSO=0.78
        syn_delay_loc   = [2.0,   0.78 ]
        syn_delay_scale = [None,  None  ]

    stem       = _pic_stem(pic_file)
    spikes_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'spikes_{stem}_angle{cond_val}_{side}')
    cond_tag   = '' if args.condition == 'binaural' else f'_cond_{args.condition}'
    drive_tag  = '_spiking' if args.drive == 'spiking' else ''
    output_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'output_lso_{stem}_{cond_label}_{side}{cond_tag}{drive_tag}')
    for sub in ('cells', 'figures', 'populations'):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    k_arr         = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT,
        spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops,
    )
    probe = lfpykit_models.PointSourcePotential(
        cell=None, x=PROBE_X, y=PROBE_Y, z=PROBE_Z, sigma=SIGMA,
    )
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

    n_available = len(pop.RANK_CELLINDICES)
    n_grab = min(args.n_single, n_available)
    cell_indices = random.sample(list(pop.RANK_CELLINDICES), n_grab)

    single_contribs = np.stack(
        [pop.output[i]['PointSourcePotential'] * 1e3 for i in cell_indices],
        axis=0,
    )   # (n_grab, n_ch, n_t)  uV
    soma_pos = np.array([[pop.pop_soma_pos[i]['x'],
                          pop.pop_soma_pos[i]['y'],
                          pop.pop_soma_pos[i]['z']] for i in cell_indices])

    pop.collect_data()
    COMM.Barrier()

    postproc = hybridLFPy.PostProcess(
        y=[pop_label], dt_output=DT,
        mapping_Yy=[(pop_label, pop_label)],
        savelist=['somapos'], probes=[probe],
        savefolder=output_dir,
    )
    if RANK == 0:
        postproc.run()
    COMM.Barrier()

    if RANK == 0:
        tvec = np.arange(single_contribs.shape[2]) * DT
        _plot_lfp(output_dir, N_CH, PROBE_Z, side, args.angle, args.n_cells)
        _plot_single_cells(output_dir, single_contribs, tvec,
                           PROBE_Z, soma_pos, PROBE_X, PROBE_Y,
                           cell_gids=cell_indices, total_sim_cells=args.n_cells)
        _plot_phase_cycle(output_dir, meta.get('stim_freq_hz'), PROBE_Z,
                          side, args.angle, args.n_cells)


# ---------------------------------------------------------------------------
# Plotting: phase-cycle / depth-profile figure
# ---------------------------------------------------------------------------
def _plot_phase_cycle(output_dir, stimulus_freq, probe_z, side, angle, n_cells,
                      skip_ms=10.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy.signal

    if stimulus_freq is None:
        print('Warning: stimulus frequency unknown; skipping phase-cycle plot.')
        return

    h5_path = os.path.join(output_dir, 'PointSourcePotential_sum.h5')
    with h5py.File(h5_path, 'r') as f:
        lfp   = f['data'][()] * 1e3
        srate = float(f['srate'][()])

    dt_ms    = 1e3 / srate
    skip_idx = int(skip_ms / dt_ms)
    lfp_ss   = lfp[:, skip_idx:]
    lfp_ss   = scipy.signal.detrend(lfp_ss, axis=1)

    T_int    = max(1, int(round(1e3 / stimulus_freq / dt_ms)))
    n_cycles = lfp_ss.shape[1] // T_int
    if n_cycles < 1:
        print('Warning: <1 complete cycle after ramp skip; skipping phase plot.')
        return

    lfp_phase = (lfp_ss[:, :n_cycles * T_int]
                 .reshape(lfp_ss.shape[0], n_cycles, T_int)
                 .mean(axis=1))

    phase   = np.linspace(0, 1, T_int, endpoint=False)
    n_ch    = lfp_phase.shape[0]
    spacing = (probe_z[-1] - probe_z[0]) / max(n_ch - 1, 1)
    half    = T_int // 2
    lfp_d1  = lfp_phase - lfp_phase.mean(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
    panels = [
        (axes[0], lfp_phase,
         f'Side {side} | {angle}° | N={n_cells} | {stimulus_freq:.0f} Hz'),
        (axes[1], lfp_d1,
         f'Side {side} | {angle}° | N={n_cells} | {stimulus_freq:.0f} Hz'),
    ]

    for ax, data, title in panels:
        peak = np.abs(data).max() or 1e-9
        sc   = peak / (spacing * 1)
        for ch in range(n_ch):
            offset = probe_z[ch]
            trace  = data[ch] / sc
            ax.plot(phase, trace + offset, color='gray', lw=0.9)
            ax.plot(phase[half], trace[half] + offset,
                    'o', color='steelblue', ms=4, zorder=3)
            ax.plot(phase[-1],   trace[-1]   + offset,
                    'o', color='firebrick', ms=4, zorder=3)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Probe z (μm)')
        ax.set_title(title)
        ax.set_xlim(0, 1)

    ax3 = axes[2]
    for t in range(T_int):
        if t == half:
            ax3.plot(probe_z, lfp_d1[:, t], color='steelblue', lw=1.8, zorder=3)
        elif t == T_int - 1:
            ax3.plot(probe_z, lfp_d1[:, t], color='firebrick',  lw=1.8, zorder=3)
        else:
            ax3.plot(probe_z, lfp_d1[:, t], color='lightgray',  lw=0.6, zorder=1)
    ax3.axhline(0, color='k', lw=0.5, ls='--', zorder=2)
    ax3.set_xlabel('Probe z (μm)')
    ax3.set_ylabel('LFP (µV)')
    ax3.set_title('Depth profile (D1)')

    fig_path = os.path.join(output_dir, 'figures', 'lso_lfp_phase_cycle.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Phase-cycle figure saved -> {fig_path}')


# ---------------------------------------------------------------------------
# Plotting: compound LFP
# ---------------------------------------------------------------------------
def _plot_lfp(output_dir, n_ch, probe_z, side, angle, n_cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h5_path = os.path.join(output_dir, 'PointSourcePotential_sum.h5')
    with h5py.File(h5_path, 'r') as f:
        lfp   = f['data'][()] * 1e3        # mV -> uV
        srate = float(f['srate'][()])

    tvec    = np.arange(lfp.shape[1]) / srate * 1e3
    spacing = (probe_z[-1] - probe_z[0]) / max(n_ch - 1, 1)
    vmax    = float(np.percentile(np.abs(lfp), 99)) or 1e-9
    per_ch_scale = np.percentile(np.abs(lfp), 99, axis=1, keepdims=True)
    per_ch_scale = np.where(per_ch_scale < 1e-9, 1e-9, per_ch_scale)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)

    ax = axes[0]
    for i in range(n_ch):
        ax.plot(tvec, lfp[i] / per_ch_scale[i] * spacing * 0.100 + probe_z[i],
                color='k', lw=0.6)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Probe z (um)')
    ax.set_title('LSO compound LFP — stacked traces')

    ax2 = axes[1]
    im = ax2.imshow(lfp, aspect='auto', origin='lower',
                    extent=[tvec[0], tvec[-1], probe_z[0], probe_z[-1]],
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax2, label='LFP (uV)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Probe z (um)')
    ax2.set_title(f'LSO | Side {side} | angle {angle} deg | N={n_cells}')

    fig_path = os.path.join(output_dir, 'figures', 'lso_lfp_reconstruction.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Compound LFP figure saved -> {fig_path}')


# ---------------------------------------------------------------------------
# Plotting: single-cell contributions
# ---------------------------------------------------------------------------
def _plot_single_cells(output_dir, single_contribs, tvec, probe_z,
                       soma_pos, probe_x, probe_y, cell_gids, total_sim_cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_cells = single_contribs.shape[0]

    def _min_dist(sx, sy, sz):
        return float(np.min(np.sqrt(
            (sx - probe_x)**2 + (sy - probe_y)**2 + (sz - probe_z)**2)))

    fig = plt.figure(figsize=(14, 2.8 * n_cells))
    gs  = gridspec.GridSpec(n_cells, 2, figure=fig,
                            left=0.07, right=0.97, hspace=0.5, wspace=0.35)

    for i in range(n_cells):
        gid = cell_gids[i]
        lso_idx = int(round(gid * (N_LSO_TOTAL - 1) / (total_sim_cells - 1))) \
                  if total_sim_cells > 1 else 0

        sx, sy, sz = soma_pos[i]
        d_min = _min_dist(sx, sy, sz)

        ax_map = fig.add_subplot(gs[i, 0])
        vmax = np.abs(single_contribs[i]).max() or 1e-9
        ax_map.imshow(single_contribs[i], aspect='auto', origin='lower',
                      extent=[tvec[0], tvec[-1], probe_z[0], probe_z[-1]],
                      cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax_map.set_ylabel('z (um)')
        ax_map.set_title(
            f'Sim ID {gid} (LSO idx {lso_idx}) | soma ({sx:.0f}, {sy:.0f}, {sz:.0f}) um'
            f' | d_min = {d_min:.0f} um'
        )
        if i == n_cells - 1:
            ax_map.set_xlabel('Time (ms)')

        best_ch = int(np.argmax(np.abs(single_contribs[i]).max(axis=1)))
        ax_tr = fig.add_subplot(gs[i, 1])
        ax_tr.plot(tvec, single_contribs[i, best_ch], color='steelblue', lw=0.8)
        ax_tr.set_ylabel('LFP (uV)')
        ax_tr.set_title(
            f'Sim ID {gid} ch {best_ch} (z = {probe_z[best_ch]:.0f} um)'
            f' | d_min = {d_min:.0f} um'
        )
        if i == n_cells - 1:
            ax_tr.set_xlabel('Time (ms)')

    fig.suptitle('LSO single-cell LFP contributions', y=1.01)
    fig_path = os.path.join(output_dir, 'figures', 'lso_lfp_single_cells.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Single-cell figure saved -> {fig_path}')


if __name__ == '__main__':
    main()
