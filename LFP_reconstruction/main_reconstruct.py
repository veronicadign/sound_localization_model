#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
HybridLFPy LFP reconstruction for MSO population.

CLI usage (single or multi-process):
  python LFP_reconstruction/main_reconstruct.py [options]
  mpiexec -n 4 python LFP_reconstruction/main_reconstruct.py --angle 45 --side R --n-cells 200

Options:
  --pic-file FILE   Path to .pic simulation result (default: RESULTS/baseline_simulation.pic)
  --angle DEGREES   Sound azimuth angle (default: 0)
  --side  L|R       Brain side (default: L)
  --n-cells N       MSO cells to simulate (default: 100)
  --n-single N      Single-cell contribution plots to save (default: 5)

Outputs saved to RESULTS/lfp_tmp/output_angle{ANGLE}_{SIDE}/figures/:
  mso_lfp_reconstruction.png   — compound LFP (stacked traces + colourmap)
  mso_lfp_single_cells.png     — per-cell LFP colourmap + best-channel trace
                                  annotated with min distance soma→probe
"""

import os
import re
import sys

import numpy as np
import h5py

import LFPy
import lfpykit.models as lfpykit_models
import hybridLFPy
from hybridLFPy.population import Population
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# ---------------------------------------------------------------------------
# Paths  (module-level so convergence_test.py can import them)
# ---------------------------------------------------------------------------
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOC_FILE   = os.path.join(REPO_ROOT, 'MSO_models', 'mso_model.hoc')
SPIKES_DIR = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp', 'spikes')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp', 'output')
META_FILE  = os.path.join(SPIKES_DIR, 'metadata.json')

# ---------------------------------------------------------------------------
# Simulation parameters  (module-level for convergence_test.py)
# ---------------------------------------------------------------------------
DT           = 0.026   # ms #0.0625
TSTOP        = 50.0     # ms
V_INIT       = -57.0    # mV  (E_L.MSO)
N_CELLS           = 3000    # default representative cell count
N_MSO_TOTAL       = 15500   # MSO neurons per side in NEST sim (params.py POP_NUM.MSO)
MSO_DENSITY_MM3   = 13049.0 # human MSO packing density (neurons/mm³)

# ---------------------------------------------------------------------------
# Probe geometry  (module-level for convergence_test.py)
# ---------------------------------------------------------------------------
N_CH    = 16
PROBE_Z = np.linspace(-400, 400, N_CH)   # μm
PROBE_X = np.zeros(N_CH)
PROBE_Y = np.zeros(N_CH)
SIGMA   = 0.3                            # S/m

# ---------------------------------------------------------------------------
# Connectivity  (module-level for convergence_test.py)
# ---------------------------------------------------------------------------
LAYER_BOUNDARIES = [
    [ 10.,  160.],   # medial dendrite
    [-160., -10.],   # lateral dendrite
    [ -10.,  10.],   # soma
]
K_YXL = [
    [2, 0, 0],
    [1, 0, 0],
    [0, 2, 1],
]
SYN_DELAY_LOC   = [2.0, 1.0, 1.0]
SYN_DELAY_SCALE = [None, None, None]


# ---------------------------------------------------------------------------
# MSOPopulation subclass
# ---------------------------------------------------------------------------
class MSOPopulation(Population):
    """Population subclass: tonotopic presynaptic assignment + heterogeneous Exp2Syn params."""

    PER_POP_SYN = {
        'SBC': {
            'syntype': 'Exp2Syn',
            'tau1':    0.15,    # ms  (TAUS_EX_RISE.MSO)
            'tau2':    0.3,     #0.3 ms  (TAUS_EX_DECAY.MSO) try 0.2
            'e':       0.0,     # mV  excitatory reversal
            'weight':  0.012,   # μS = 12 nS
        },
        'MNTBC': {
            'syntype': 'Exp2Syn',
            'tau1':    0.15,
            'tau2':    0.7,     #0.7 ms  (TAUS_IN_DECAY.MSO) try 0.4
            'e':      -75.0,    # mV  inhibitory reversal
            'weight':  0.010,   # μS = 10 nS 0.010
        },
        'LNTBC': {
            'syntype': 'Exp2Syn',
            'tau1':    0.15,
            'tau2':    0.4,
            'e':      -75.0,
            'weight':  0.000,
        },
    }

    def __init__(self, n_syn_per_pop=None, **kwargs):
        self.n_syn_per_pop = n_syn_per_pop or {}
        super().__init__(**kwargs)

    def get_all_SpCells(self):
        """Tonotopic x_to_one assignment (mirrors NEST custom connector)."""
        n_cells  = self.POPULATION_SIZE   # number being simulated
        N_mso    = N_MSO_TOTAL            # 15500 per side
        SpCells  = {}

        for cellindex in self.RANK_CELLINDICES:
            # Map simulated index → global MSO tonotopic index (uniform sample)
            mso_idx = (int(round(cellindex * (N_mso - 1) / (n_cells - 1)))
                       if n_cells > 1 else 0)

            SpCells[cellindex] = {}
            for X in self.X: # loops through each presynaptic population
                nodes  = self.networkSim.nodes[X]   # 1-based contiguous GIDs
                N_pre  = len(nodes)
                n_src  = self.n_syn_per_pop.get(X, 1)

                # x_to_one window: same formula as NEST custom connector
                step      = (N_pre - n_src) / max(N_mso - 1, 1)
                pre_start = min(int(round(mso_idx * step)), N_pre - n_src)
                window    = nodes[pre_start : pre_start + n_src]

                # Assign window GIDs layer-by-layer (respecting synIdx structure)
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
        for X in self.X:
            for j in range(len(self.synIdx[cellindex][X])):
                synDelays = (self.synDelays[cellindex][X][j]
                             if self.synDelays is not None else None)
                self.insert_synapses(
                    cell=cell,
                    cellindex=cellindex,
                    synParams=self.PER_POP_SYN[X.rsplit('_', 1)[0]].copy(),
                    idx=self.synIdx[cellindex][X][j],
                    X=X,
                    SpCell=self.SpCells[cellindex][X][j],
                    synDelays=synDelays,
                )

    def draw_rand_pos(self, **kwargs):
        soma_pos = super().draw_rand_pos(**kwargs)
        # Cell index = tonotopic rank → cell 0 (lowest CF) at x_min, cell n-1 (highest CF) at x_max.
        # (Fischl et al. 2016: "tonotopy along rostrocaudal axis" -> axis perpendicular to probe/z).
        soma_pos.sort(key=lambda p: p['x']) # sort by x coordinate in the cylinder
        return soma_pos


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    import random
    parser = argparse.ArgumentParser(description='MSO LFP reconstruction')
    parser.add_argument('--pic-file', type=str,  default=None,
                        dest='pic_file',
                        help='Path to .pic file (default: RESULTS/baseline_simulation.pic)')
    parser.add_argument('--angle',    type=int,  default=0,
                        help='Sound angle in degrees (default: 0)')
    parser.add_argument('--side',     type=str,  default='L', choices=['L', 'R'],
                        help='Side (default: L)')
    parser.add_argument('--n-cells',  type=int,  default=N_CELLS, dest='n_cells',
                        help=f'Number of MSO cells (default: {N_CELLS})')
    parser.add_argument('--n-single', type=int,  default=5, dest='n_single',
                        help='Number of single-cell contribution plots (default: 5)')
    parser.add_argument('--monaural', action='store_true', default=False,
                        help='Monaural condition: remove contralateral SBC (silences medial dendrite)')
    parser.add_argument('--hoc-file', type=str, default=None, dest='hoc_file',
                        help='HOC morphology file (default: MSO_models/mso_model.hoc)')
    args = parser.parse_args()

    if args.hoc_file is None:
        args.hoc_file = HOC_FILE

    side        = args.side
    contra_side = 'R' if side == 'L' else 'L'

    # Spike extraction on rank 0 only; broadcast metadata to all ranks
    if RANK == 0:
        meta = _extract_spikes(args.angle, side, pic_file=args.pic_file)
    else:
        meta = None
    meta = COMM.bcast(meta, root=0)
    COMM.Barrier()

    # MSO anatomy: medial dendrite ← contra SBC, lateral dendrite ← ipsi SBC
    # (Cant & Hyson 1992; Joris et al. 1998)
    # Monaural condition removes contra SBC (zeroes medial dendrite row).
    X_pops = [f'SBC_{contra_side}', f'SBC_{side}',
              f'MNTBC_{side}', f'LNTBC_{side}']
    k_yxl_local = [
        [3, 0, 0, 0],   # medial dendrite:  1×SBC_contra
        [0, 3, 0, 0],   # lateral dendrite: 2×SBC_ipsi
        [0, 0, 2, 1],   # soma:             2×MNTBC + 1×LNTBC
    ]
    if args.monaural:
        k_yxl_local[1] = [0, 0, 0, 0]   # no contra SBC → silence medial dendrite
    j_yx_local      = [0.012, 0.012, 0.010, 0.010]
    tau_yx_local    = [0.2,   0.2,   0.4,   0.4  ]
    syn_delay_loc   = [2.0,   2.0,   1.0,   1.0  ]
    syn_delay_scale = [None,  None,  None,  None  ]

    pic_file   = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                               'baseline_simulation.pic')
    stem       = _pic_stem(pic_file)
    spikes_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'spikes_{stem}_angle{args.angle}_{side}')
    cond_tag   = '_monaural' if args.monaural else ''
    hoc_tag    = '_active' if args.hoc_file != HOC_FILE else ''
    output_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'output_{stem}_angle{args.angle}_{side}{cond_tag}{hoc_tag}')
    for sub in ('cells', 'figures', 'populations'):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # Synapses per presynaptic pop per MSO cell (column sums of k_yxl_local)
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
    pop_label = f'MSO_{side}'
    # Cylinder sized to match human MSO packing density (13049 neurons/mm³).
    # Aspect ratio h = 2r (isotropic): V = 2π r³ → r = (V/2π)^(1/3).
    _V_um3 = (args.n_cells / MSO_DENSITY_MM3) * 1e9
    _r_um  = (_V_um3 / (2 * np.pi)) ** (1 / 3)
    pop = MSOPopulation(
        n_syn_per_pop=n_syn_per_pop,
        y=pop_label,
        cellParams={
            'morphology': args.hoc_file, 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP, 'nsegs_method': None,
        },
        rand_rot_axis=['z'],
        simulationParams={'rec_imem': True},
        populationParams={
            'number': args.n_cells,
            'radius': 61.5,  # μm  (OR _r_um for density-based sizing)
            'z_min': -50.0, 'z_max': 50.0, 'min_cell_interdist': 1.5,
            'min_r': np.array([[0.], [0.]]),
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
        synDelayScale=syn_delay_scale,
        J_yX=j_yx_local,
        tau_yX=tau_yx_local,
    )

    pop.run()
    COMM.Barrier()

    # Grab single-cell outputs before collect_data() clears pop.output
    
    # Ensure we don't try to sample more cells than exist on this specific MPI rank
    n_available = len(pop.RANK_CELLINDICES)
    n_grab = min(args.n_single, n_available)
    
    # Select cells randomly instead of taking the first contiguous block
    cell_indices = random.sample(list(pop.RANK_CELLINDICES), n_grab)

    single_contribs = np.stack(
        [pop.output[i]['PointSourcePotential'] * 1e3 for i in cell_indices],
        axis=0,
    )   # (n_grab, n_ch, n_t)  µV
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
        tvec = np.arange(single_contribs.shape[2]) * DT   # ms
        _plot_lfp(output_dir, N_CH, PROBE_Z, side, args.angle, args.n_cells)
        _plot_phase_cycle(output_dir, meta.get('stim_freq_hz'), PROBE_Z,
                          side, args.angle, args.n_cells)
        _plot_single_cells(output_dir, single_contribs, tvec,
                           PROBE_Z, soma_pos, PROBE_X, PROBE_Y, 
                           cell_gids=cell_indices, total_sim_cells=args.n_cells)


# ---------------------------------------------------------------------------
# Helper: extract spikes (runs extract_spikes.py automatically if needed)
# ---------------------------------------------------------------------------
def _pic_stem(pic_file):
    """Sanitised basename of a .pic path, safe for use in directory names."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_',
                  os.path.splitext(os.path.basename(pic_file))[0])


def _extract_spikes(angle, side, pic_file=None):
    """Load cached metadata or run extraction from .pic file."""
    import json
    if pic_file is None:
        pic_file = os.path.join(REPO_ROOT, 'RESULTS', 'baseline_simulation.pic')
    stem       = _pic_stem(pic_file)
    spikes_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'spikes_{stem}_angle{angle}_{side}')
    meta_path  = os.path.join(spikes_dir, 'metadata.json')
    contra_side = 'R' if side == 'L' else 'L'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        # Accept cache only if contralateral SBC is present (new format)
        if f'SBC_{contra_side}' in meta:
            return meta
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import extract_spikes as _es
    return _es.extract_and_save(pic_file, angle, side, spikes_dir)


# ---------------------------------------------------------------------------
# Plotting: compound LFP
# ---------------------------------------------------------------------------
def _plot_lfp(output_dir, n_ch, probe_z, side, angle, n_cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h5_path = os.path.join(output_dir, 'PointSourcePotential_sum.h5')
    with h5py.File(h5_path, 'r') as f:
        lfp   = f['data'][()] * 1e3        # mV → µV
        srate = float(f['srate'][()])

    tvec  = np.arange(lfp.shape[1]) / srate * 1e3
    scale = max(np.abs(lfp).max() * 2, 1e-9)
    vmax  = float(np.abs(lfp).max()) or 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)

    ax = axes[0]
    for i in range(n_ch):
        ax.plot(tvec, lfp[i] / scale * 70 + probe_z[i], color='k', lw=0.6)
        #ax.plot(tvec, lfp[i] + probe_z[i], color='k', lw=0.6)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Probe z (μm)')
    ax.set_title('MSO compound LFP — stacked traces')

    ax2 = axes[1]
    im = ax2.imshow(lfp, aspect='auto', origin='lower',
                    extent=[tvec[0], tvec[-1], probe_z[0], probe_z[-1]],
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax2, label='LFP (μV)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Probe z (μm)')
    ax2.set_title(f'Side {side} | angle {angle}° | N={n_cells}')

    fig_path = os.path.join(output_dir, 'figures', 'mso_lfp_reconstruction.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Compound LFP figure saved -> {fig_path}')


# ---------------------------------------------------------------------------
# Plotting: phase-cycle figure (C1 / D1)
# ---------------------------------------------------------------------------
def _plot_phase_cycle(output_dir, stimulus_freq, probe_z, side, angle, n_cells,
                      skip_ms=10.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if stimulus_freq is None:
        print('Warning: stimulus frequency unknown; skipping phase-cycle plot.')
        return

    h5_path = os.path.join(output_dir, 'PointSourcePotential_sum.h5')
    with h5py.File(h5_path, 'r') as f:
        lfp   = f['data'][()] * 1e3    # mV → µV  (n_ch, n_t)
        srate = float(f['srate'][()])

    dt_ms    = 1e3 / srate
    skip_idx = int(skip_ms / dt_ms)
    lfp_ss   = lfp[:, skip_idx:]
    import scipy.signal
    lfp_ss   = scipy.signal.detrend(lfp_ss, axis=1)

    T_int    = max(1, int(round(1e3 / stimulus_freq / dt_ms)))
    n_cycles = lfp_ss.shape[1] // T_int
    if n_cycles < 1:
        print('Warning: <1 complete cycle after ramp skip; skipping phase plot.')
        return

    lfp_phase = (lfp_ss[:, :n_cycles * T_int]
                 .reshape(lfp_ss.shape[0], n_cycles, T_int)
                 .mean(axis=1))          # (n_ch, T_int)

    phase   = np.linspace(0, 1, T_int, endpoint=False)
    n_ch    = lfp_phase.shape[0]
    spacing = (probe_z[-1] - probe_z[0]) / max(n_ch - 1, 1)
    half    = T_int // 2
    lfp_d1  = lfp_phase - lfp_phase.mean(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
    panels = [
        (axes[0], lfp_phase,
         f'Side {side} | {angle}° | N={n_cells} | {stimulus_freq:.0f} Hz'),
        (axes[1], lfp_d1, f'Side {side} | {angle}° | N={n_cells} | {stimulus_freq:.0f} Hz'),
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

    # Depth profile: LFP vs probe depth for each phase time-point
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

    fig_path = os.path.join(output_dir, 'figures', 'mso_lfp_phase_cycle.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Phase-cycle figure saved -> {fig_path}')


# ---------------------------------------------------------------------------
# Plotting: single-cell contributions
# ---------------------------------------------------------------------------
def _plot_single_cells(output_dir, single_contribs, tvec, probe_z,
                       soma_pos, probe_x, probe_y, cell_gids, total_sim_cells):
    """
    single_contribs : (n_cells, n_ch, n_t)  µV
    soma_pos        : (n_cells, 3)          µm  [x, y, z]
    cell_gids       : (n_cells,)            Simulated Global IDs
    """
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
        
        # Calculate the true biological tonotopic index
        N_mso = 15500 
        mso_idx = int(round(gid * (N_mso - 1) / (total_sim_cells - 1))) if total_sim_cells > 1 else 0

        sx, sy, sz = soma_pos[i]
        d_min = _min_dist(sx, sy, sz)

        # Left: all-channel colour map
        ax_map = fig.add_subplot(gs[i, 0])
        vmax = np.abs(single_contribs[i]).max() or 1e-9
        ax_map.imshow(single_contribs[i], aspect='auto', origin='lower',
                      extent=[tvec[0], tvec[-1], probe_z[0], probe_z[-1]],
                      cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax_map.set_ylabel('z (μm)')
        ax_map.set_title(
            f'Sim ID {gid} | soma ({sx:.0f}, {sy:.0f}, {sz:.0f}) μm | d_min = {d_min:.0f} μm'
            # we can also add (Bio ID {mso_idx})
        )
        if i == n_cells - 1:
            ax_map.set_xlabel('Time (ms)')

        # Right: best-channel trace
        best_ch = int(np.argmax(np.abs(single_contribs[i]).max(axis=1)))
        ax_tr = fig.add_subplot(gs[i, 1])
        ax_tr.plot(tvec, single_contribs[i, best_ch], color='steelblue', lw=0.8)
        ax_tr.set_ylabel('LFP (μV)')
        ax_tr.set_title(
            f'Sim ID {gid} ch {best_ch} (z = {probe_z[best_ch]:.0f} μm) | d_min = {d_min:.0f} μm'
        )
        if i == n_cells - 1:
            ax_tr.set_xlabel('Time (ms)')

    fig.suptitle('Single-cell LFP contributions', y=1.01)
    fig_path = os.path.join(output_dir, 'figures', 'mso_lfp_single_cells.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Single-cell figure saved -> {fig_path}')


if __name__ == '__main__':
    main()