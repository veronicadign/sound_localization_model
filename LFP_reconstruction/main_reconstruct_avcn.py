#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
HybridLFPy LFP reconstruction for the AVCN globular bushy cell (GBC) population.

Unlike the MSO/LSO scripts (hand-drawn stick morphologies + Exp2Syn), the GBC is
a morphologically detailed cell decorated with ported cnmodel channels
(klt/kht/ihvcn/leak/nacncoop, XM13_nacncoop mouse Type-II) and driven by
auditory-nerve endbulbs of Held.

Scaffold morphology: cnmodel bushy_stick.hoc (AVCN_models/morphology/); to be
swapped for a Dryad EM reconstruction later. Biophysics is applied at cell build
time via LFPy custom_fun = gbc_biophysics.decorate_gbc (adapts to any morphology).

Presynaptic drive: ANF_{side} only (ipsilateral; cochlear nucleus is ipsilateral
to the ear). 20 endbulbs/GBC, matching the NEST ANFs2GBCs convergence, placed
with biophysically realistic weighting: 70% soma / 20% proximal dendrite+hubs /
10% axon hillock+AIS (gbc_biophysics.weighted_endbulb_idx).

CLI (single or MPI):
  python LFP_reconstruction/main_reconstruct_avcn.py --pic-file RESULTS/x.pic \
      --angle 0 --side L --n-cells 100
  mpiexec -n 4 python LFP_reconstruction/main_reconstruct_avcn.py ... --n-cells 3600

Outputs -> RESULTS/lfp_tmp/output_avcn_{stem}_angle{A}_{S}/figures/:
  avcn_lfp_reconstruction.png
  avcn_lfp_single_cells.png
"""

import os
import sys
import random

import numpy as np
import h5py

import LFPy
import neuron
import lfpykit.models as lfpykit_models
import hybridLFPy
from hybridLFPy.population import Population
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# ---------------------------------------------------------------------------
# Paths + mechanisms (AVCN_models only — SUFFIX collides with MSO_models)
# ---------------------------------------------------------------------------
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AVCN_DIR   = os.path.join(REPO_ROOT, 'AVCN_models')
STICK_HOC  = os.path.join(AVCN_DIR, 'morphology', 'bushy_stick.hoc')
# Default: real Dryad EM reconstruction (mesh-inflated, accurate surface areas).
# Use --hoc-file to pick another VCN_c* cell or STICK_HOC for a fast smoke test.
HOC_FILE   = os.path.join(AVCN_DIR, 'morphology', 'dryad',
                          'VCN_c09_Full_MeshInflate.hoc')
sys.path.insert(0, AVCN_DIR)
import gbc_biophysics  # noqa: E402

try:
    neuron.load_mechanisms(AVCN_DIR)
except RuntimeError as _e:
    if 'already exists' not in str(_e):
        raise


def _decorate(cell):
    """LFPy custom_fun: apply cnmodel GBC channel densities. nseg set by LFPy."""
    gbc_biophysics.decorate_gbc(cell, set_nseg=False)


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DT     = 0.026   # ms
TSTOP  = 50.0    # ms
V_INIT = -65.0   # mV  (leak erev, XM13_nacncoop)

N_CELLS       = 100      # default representative count
N_GBC_TOTAL   = 3600     # GBC per side in NEST sim (params.py n_GBCs)
N_ANF_TOTAL   = 35000    # ANF per side (cochlea output)
N_ENDBULBS    = 20       # ANF endbulbs per GBC (params.py ANFs2GBCs)

# ---------------------------------------------------------------------------
# AVCN elliptic-cylinder geometry (x = tonotopic/dorsoventral, y = rostrocaudal)
# Human AVCN is a few hundred µm across; values approximate, refine with the
# EM/anatomy once the real morphology is in.
# ---------------------------------------------------------------------------
ELLIPSE_RADIUS_X = 400.0   # µm  tonotopic half-axis
ELLIPSE_RADIUS_Y = 600.0   # µm  rostrocaudal half-axis

# ---------------------------------------------------------------------------
# Probe geometry (reuse 16-ch linear layout)
# ---------------------------------------------------------------------------
N_CH    = 16
PROBE_Z = np.linspace(-300, 300, N_CH)   # µm (GBC is smaller than MSO/LSO)
PROBE_X = np.zeros(N_CH)
PROBE_Y = np.zeros(N_CH)
SIGMA   = 0.3   # S/m

# ---------------------------------------------------------------------------
# Target myelinated-axon direction in model coords (x=dorsoventral, ventral=-x;
# y=rostrocaudal; z=mediolateral). Pure ventromedial, medial mirrored per side
# (left toward -z, right toward +z). Every GBC's axon is rotated onto this so
# axonal axial currents summate coherently into a large-scale dipole. Editable:
# flip a sign to change ventral/medial orientation.
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)

# Fixed axon orientation for both sides: axon exits ventromedially (-x, -z).
# Input laterality is handled by X_pops = [ANF_{side}], not by flipping geometry.
AXON_TARGET = _unit([-1., 0., -1.])

# ---------------------------------------------------------------------------
# Synapse layer structure. Placement is overridden per section in
# insert_all_synapses; a single wide z-band just fixes the per-cell count (20).
# ---------------------------------------------------------------------------
LAYER_BOUNDARIES = [[-100.0, 100.0]]
K_YXL            = [[N_ENDBULBS]]          # 20 ANF endbulbs / GBC
SYN_DELAY_LOC    = [0.5]                   # ms (params.py ANFs2GBCs delay)
SYN_DELAY_SCALE  = [None]


# ---------------------------------------------------------------------------
# AVCNPopulation subclass
# ---------------------------------------------------------------------------
class AVCNPopulation(Population):
    """GBC population: tonotopic ANF endbulb assignment + somatic AMPA endbulbs."""

    PER_POP_SYN = {
        'ANF': {
            'syntype': 'Exp2Syn',
            'tau1':    0.2,     # ms from TAUS_EX_RISE.GBC OR # 0.1 -> fast endbulb AMPA rise
            'tau2':    0.5,     # ms from TAUS_EX_DECAY.GBC # 0.3 -> fast endbulb AMPA decay
            'e':       0.0,     # mV  excitatory reversal
            'weight':  0.005,   # µS  (endbulb conductance; tunable, scales LFP)
        },
    }

    def __init__(self, n_syn_per_pop=None, axon_target=None,
                 n_post_total=N_GBC_TOTAL, n_endbulbs=N_ENDBULBS,
                 endbulb_weights=None, per_pop_syn=None, **kwargs):
        self.n_syn_per_pop = n_syn_per_pop or {}
        self.axon_target = axon_target        # unit vector in model coords, or None
        # Population-specific knobs (defaults reproduce the GBC pipeline exactly):
        self.n_post_total    = n_post_total   # total cells/side in NEST (GBC 3600)
        self.n_endbulbs      = n_endbulbs     # ANF endbulbs per cell   (GBC 20)
        self.endbulb_weights = endbulb_weights  # compartment split, or None=GBC 70/20/5/5
        self.per_pop_syn     = per_pop_syn if per_pop_syn is not None else self.PER_POP_SYN
        super().__init__(**kwargs)

    def set_rotations(self):
        """Deterministic per-cell rotation aligning the native myelinated axon to
        ``axon_target`` (same for every cell → coherent ventromedial axons).

        Replaces hybridLFPy's random-axis rotation. With ``rand_rot_axis=[]`` and
        ``axon_target=None`` this falls back to the parent (identity) behaviour.
        """
        from time import time
        if self.axon_target is None:
            return super().set_rotations()
        tic = time()
        if RANK == 0:
            trial_params = {k: v for k, v in self.cellParams.items()
                            if k not in ('custom_fun', 'custom_fun_args')}
            cell = LFPy.Cell(**trial_params)
            native = gbc_biophysics.native_axon_direction(cell)
            if native is None:
                print('[AVCN] no axon compartment found; skipping axon alignment')
                rot = {}
            else:
                rot = gbc_biophysics.lfpy_align_angles(native, self.axon_target)
                cell.set_rotation(**rot)
                achieved = gbc_biophysics.native_axon_direction(cell)
                print(f'[AVCN] axon aligned: native={np.round(native, 2)} -> '
                      f'{np.round(achieved, 2)} (target {np.round(self.axon_target, 2)})')
            rotations = [dict(rot) for _ in range(self.POPULATION_SIZE)]
            print('found cell rotations in %.2f s' % (time() - tic))
        else:
            rotations = None
        return COMM.bcast(rotations, root=0)

    def get_all_SpCells(self):
        """Tonotopic ANF -> cell assignment mirroring the NEST x_to_one connector."""
        n_cells    = self.POPULATION_SIZE
        n_post_tot = self.n_post_total
        SpCells = {}
        for cellindex in self.RANK_CELLINDICES:
            post_idx = (int(round(cellindex * (n_post_tot - 1) / (n_cells - 1)))
                        if n_cells > 1 else 0)

            SpCells[cellindex] = {}
            for X in self.X:
                nodes = self.networkSim.nodes[X]
                N_pre = len(nodes)
                n_src = self.n_syn_per_pop.get(X, self.n_endbulbs)

                step      = (N_pre - n_src) / max(n_post_tot - 1, 1)
                pre_start = min(int(round(post_idx * step)), N_pre - n_src)
                window    = nodes[pre_start: pre_start + n_src]

                SpCell   = []
                src_used = 0
                for compartments in self.synIdx[cellindex][X]:
                    size = len(compartments)
                    if size > 0:
                        SpCell.append(window[src_used: src_used + size].astype('int32'))
                        src_used += size
                    else:
                        SpCell.append(np.array([], dtype='int32'))
                SpCells[cellindex][X] = SpCell
        return SpCells

    def insert_all_synapses(self, cellindex, cell):
        for X in self.X:
            pop_type = X.rsplit('_', 1)[0]
            for j in range(len(self.synIdx[cellindex][X])):
                idx = self.synIdx[cellindex][X][j]
                if len(idx) == 0:
                    continue
                synDelays = (self.synDelays[cellindex][X][j]
                             if self.synDelays is not None else None)

                idx = gbc_biophysics.weighted_endbulb_idx(
                    cell, len(idx), weights=self.endbulb_weights)

                self.insert_synapses(
                    cell=cell,
                    cellindex=cellindex,
                    synParams=self.per_pop_syn[pop_type].copy(),
                    idx=idx,
                    X=X,
                    SpCell=self.SpCells[cellindex][X][j],
                    synDelays=synDelays,
                )

    def draw_rand_pos(self, radius_x=ELLIPSE_RADIUS_X, radius_y=ELLIPSE_RADIUS_Y,
                      z_min=0.0, z_max=0.0, min_cell_interdist=1.0, **kwargs):
        """Uniform sampling inside the AVCN elliptic cylinder, sorted tonotopically."""
        N = self.POPULATION_SIZE

        def _sample(n):
            xi = (np.random.rand(n) - 0.5) * 2 * radius_x
            yi = (np.random.rand(n) - 0.5) * 2 * radius_y
            zi = np.random.rand(n) * (z_max - z_min) + z_min
            return xi, yi, zi

        x, y, z = _sample(N)
        bad = np.where((x / radius_x) ** 2 + (y / radius_y) ** 2 > 1)[0]
        while len(bad):
            x[bad], y[bad], z[bad] = _sample(len(bad))
            bad = np.where((x / radius_x) ** 2 + (y / radius_y) ** 2 > 1)[0]

        too_close = np.where(self.calc_min_cell_interdist(x, y, z) < min_cell_interdist)[0]
        while len(too_close):
            xr, yr, zr = _sample(len(too_close))
            x[too_close], y[too_close], z[too_close] = xr, yr, zr
            bad = np.where((x / radius_x) ** 2 + (y / radius_y) ** 2 > 1)[0]
            while len(bad):
                x[bad], y[bad], z[bad] = _sample(len(bad))
                bad = np.where((x / radius_x) ** 2 + (y / radius_y) ** 2 > 1)[0]
            too_close = np.where(self.calc_min_cell_interdist(x, y, z) < min_cell_interdist)[0]

        soma_pos = [{'x': x[i], 'y': y[i], 'z': z[i]} for i in range(N)]
        soma_pos.sort(key=lambda p: p['x'])
        return soma_pos


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='AVCN (GBC) LFP reconstruction')
    parser.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',    type=int, default=0)
    parser.add_argument('--side',     type=str, default='L', choices=['L', 'R'])
    parser.add_argument('--n-cells',  type=int, default=N_CELLS, dest='n_cells')
    parser.add_argument('--n-single', type=int, default=5, dest='n_single')
    parser.add_argument('--hoc-file', type=str, default=HOC_FILE, dest='hoc_file',
                        help='GBC morphology hoc (default: VCN_c09 EM reconstruction)')
    args = parser.parse_args()

    side     = args.side
    pic_file = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                             'baseline_simulation.pic')

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main_reconstruct import _pic_stem, _extract_spikes

    if RANK == 0:
        meta = _extract_spikes(args.angle, side, pic_file=pic_file)
    else:
        meta = None
    meta = COMM.bcast(meta, root=0)
    COMM.Barrier()

    X_pops        = [f'ANF_{side}']
    k_yxl_local   = K_YXL
    j_yx_local    = [AVCNPopulation.PER_POP_SYN['ANF']['weight']]
    tau_yx_local  = [AVCNPopulation.PER_POP_SYN['ANF']['tau2']]

    stem       = _pic_stem(pic_file)
    spikes_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'spikes_{stem}_angle{args.angle}_{side}')
    output_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'output_avcn_{stem}_angle{args.angle}_{side}')
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
    pop_label = f'AVCN_{side}'
    pop = AVCNPopulation(
        n_syn_per_pop=n_syn_per_pop,
        axon_target=AXON_TARGET,   # fixed ventromedial; laterality set by X_pops
        y=pop_label,
        cellParams={
            'morphology': args.hoc_file, 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP,
            'nsegs_method': 'lambda_f', 'lambda_f': 100,
            'custom_fun': [_decorate], 'custom_fun_args': [{}],
        },
        rand_rot_axis=[],   # deterministic ventromedial axon orientation instead
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
        POPULATIONSEED=44,
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

    n_grab = min(args.n_single, len(pop.RANK_CELLINDICES))
    cell_indices = random.sample(list(pop.RANK_CELLINDICES), n_grab) if n_grab else []
    single_contribs = (np.stack([pop.output[i]['PointSourcePotential'] * 1e3
                                 for i in cell_indices], axis=0)
                       if cell_indices else None)
    soma_pos = (np.array([[pop.pop_soma_pos[i]['x'], pop.pop_soma_pos[i]['y'],
                           pop.pop_soma_pos[i]['z']] for i in cell_indices])
                if cell_indices else None)

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
        _plot_lfp(output_dir, N_CH, PROBE_Z, side, args.angle, args.n_cells)
        _plot_phase_cycle(output_dir, meta.get('stim_freq_hz'), PROBE_Z,
                          side, args.angle, args.n_cells)
        if single_contribs is not None:
            tvec = np.arange(single_contribs.shape[2]) * DT
            _plot_single_cells(output_dir, single_contribs, tvec, PROBE_Z,
                               soma_pos, PROBE_X, PROBE_Y, cell_indices, args.n_cells)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_lfp(output_dir, n_ch, probe_z, side, angle, n_cells,
              title_prefix='AVCN (GBC)'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with h5py.File(os.path.join(output_dir, 'PointSourcePotential_sum.h5'), 'r') as f:
        lfp   = f['data'][()] * 1e3
        srate = float(f['srate'][()])
    tvec = np.arange(lfp.shape[1]) / srate * 1e3
    # Blank the t=0 capacitive initialisation transient (a huge single-sample
    # spike that otherwise dominates the colour/trace scaling and blanks the plot).
    n0 = max(1, int(round(0.2 / (1e3 / srate))))   # ~first 0.2 ms
    lfp[:, :n0] = 0.0
    scale = max(np.abs(lfp).max() * 2, 1e-9)
    vmax  = float(np.abs(lfp).max()) or 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    for i in range(n_ch):
        axes[0].plot(tvec, lfp[i] / scale * 60 + probe_z[i], color='k', lw=0.6)
    axes[0].set_xlabel('Time (ms)'); axes[0].set_ylabel('Probe z (µm)')
    axes[0].set_title(f'{title_prefix} compound LFP — stacked traces')
    im = axes[1].imshow(lfp, aspect='auto', origin='lower',
                        extent=[tvec[0], tvec[-1], probe_z[0], probe_z[-1]],
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=axes[1], label='LFP (µV)')
    axes[1].set_xlabel('Time (ms)'); axes[1].set_ylabel('Probe z (µm)')
    axes[1].set_title(f'Side {side} | angle {angle}° | N={n_cells}')
    path = os.path.join(output_dir, 'figures', 'avcn_lfp_reconstruction.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Compound LFP figure saved -> {path}')


def _plot_phase_cycle(output_dir, stimulus_freq, probe_z, side, angle, n_cells,
                      skip_ms=10.0):
    """Phase-averaged LFP over one stimulus cycle (mirrors the MSO/LSO plot).

    Reads the saved PointSourcePotential_sum.h5, so it can be regenerated from a
    finished run without re-simulating (see plot_avcn_phase_cycle.py).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy.signal

    if stimulus_freq is None:
        print('Warning: stimulus frequency unknown; skipping phase-cycle plot.')
        return

    with h5py.File(os.path.join(output_dir, 'PointSourcePotential_sum.h5'), 'r') as f:
        lfp   = f['data'][()] * 1e3    # mV -> µV  (n_ch, n_t)
        srate = float(f['srate'][()])

    dt_ms    = 1e3 / srate
    skip_idx = int(skip_ms / dt_ms)
    lfp_ss   = scipy.signal.detrend(lfp[:, skip_idx:], axis=1)

    T_int    = max(1, int(round(1e3 / stimulus_freq / dt_ms)))
    n_cycles = lfp_ss.shape[1] // T_int
    if n_cycles < 1:
        print('Warning: <1 complete cycle after ramp skip; skipping phase plot.')
        return

    lfp_phase = (lfp_ss[:, :n_cycles * T_int]
                 .reshape(lfp_ss.shape[0], n_cycles, T_int).mean(axis=1))
    phase   = np.linspace(0, 1, T_int, endpoint=False)
    n_ch    = lfp_phase.shape[0]
    spacing = (probe_z[-1] - probe_z[0]) / max(n_ch - 1, 1)
    half    = T_int // 2
    lfp_d1  = lfp_phase - lfp_phase.mean(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
    title = f'Side {side} | {angle}° | N={n_cells} | {stimulus_freq:.0f} Hz'
    for ax, data in ((axes[0], lfp_phase), (axes[1], lfp_d1)):
        sc = (np.abs(data).max() or 1e-9) / (spacing * 1)
        for ch in range(n_ch):
            trace = data[ch] / sc
            ax.plot(phase, trace + probe_z[ch], color='gray', lw=0.9)
            ax.plot(phase[half], trace[half] + probe_z[ch], 'o', color='steelblue', ms=4, zorder=3)
            ax.plot(phase[-1],   trace[-1]   + probe_z[ch], 'o', color='firebrick', ms=4, zorder=3)
        ax.set_xlabel('Cycle'); ax.set_ylabel('Probe z (µm)'); ax.set_title(title); ax.set_xlim(0, 1)

    ax3 = axes[2]
    for t in range(T_int):
        if t == half:
            ax3.plot(probe_z, lfp_d1[:, t], color='steelblue', lw=1.8, zorder=3)
        elif t == T_int - 1:
            ax3.plot(probe_z, lfp_d1[:, t], color='firebrick', lw=1.8, zorder=3)
        else:
            ax3.plot(probe_z, lfp_d1[:, t], color='lightgray', lw=0.6, zorder=1)
    ax3.axhline(0, color='k', lw=0.5, ls='--', zorder=2)
    ax3.set_xlabel('Probe z (µm)'); ax3.set_ylabel('LFP (µV)'); ax3.set_title('Depth profile (D1)')

    path = os.path.join(output_dir, 'figures', 'avcn_lfp_phase_cycle.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Phase-cycle figure saved -> {path}')


def _plot_single_cells(output_dir, single_contribs, tvec, probe_z, soma_pos,
                       probe_x, probe_y, cell_gids, total_sim_cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Blank the t=0 capacitive initialisation transient (same artifact as the
    # compound plot; per-cell it otherwise dominates each colourmap/trace scale).
    single_contribs = single_contribs.copy()
    if len(tvec) > 1:
        n0 = max(1, int(round(0.2 / (tvec[1] - tvec[0]))))
        single_contribs[:, :, :n0] = 0.0

    n = single_contribs.shape[0]
    fig = plt.figure(figsize=(14, 2.8 * n))
    gs  = gridspec.GridSpec(n, 2, figure=fig, left=0.07, right=0.97,
                            hspace=0.5, wspace=0.35)
    for i in range(n):
        sx, sy, sz = soma_pos[i]
        d_min = float(np.min(np.sqrt((sx - probe_x) ** 2 + (sy - probe_y) ** 2 +
                                     (sz - probe_z) ** 2)))
        vmax = np.abs(single_contribs[i]).max() or 1e-9
        ax_map = fig.add_subplot(gs[i, 0])
        ax_map.imshow(single_contribs[i], aspect='auto', origin='lower',
                      extent=[tvec[0], tvec[-1], probe_z[0], probe_z[-1]],
                      cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax_map.set_ylabel('z (µm)')
        ax_map.set_title(f'Sim ID {cell_gids[i]} | soma ({sx:.0f},{sy:.0f},{sz:.0f}) '
                         f'µm | d_min={d_min:.0f} µm')
        best = int(np.argmax(np.abs(single_contribs[i]).max(axis=1)))
        ax_tr = fig.add_subplot(gs[i, 1])
        ax_tr.plot(tvec, single_contribs[i, best], color='steelblue', lw=0.8)
        ax_tr.set_ylabel('LFP (µV)')
        ax_tr.set_title(f'ch {best} (z={probe_z[best]:.0f} µm)')
        if i == n - 1:
            ax_map.set_xlabel('Time (ms)'); ax_tr.set_xlabel('Time (ms)')
    path = os.path.join(output_dir, 'figures', 'avcn_lfp_single_cells.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'Single-cell figure saved -> {path}')


if __name__ == '__main__':
    main()
