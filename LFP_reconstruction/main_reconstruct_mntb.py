#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
HybridLFPy LFP reconstruction for the MNTB (medial nucleus of the trapezoid body).

The MNTB principal cell is driven by a single giant calyx of Held from the
CONTRALATERAL globular bushy cell (GBC). This script reconstructs the MNTB
POSTSYNAPTIC field (the calyx EPSC somatic sink + passive spread). The calyx
PRESYNAPTIC current ("prespike") + cleft potential are a SEPARATE co-located
generator added by --with-calyx (see calyx_model.hoc / Phase 3).

  MNTB drive (per BrainstemModel.py):
    GBC_{contra} -> MNTB_{side}   excitatory calyx, 1 per cell, decussating

Non-double-counting: the GBC axon crossing the midline (the dominant wave-III
generator) is modelled in the AVCN GBC pipeline (VCN_c09_extended_axon.hoc), NOT
here. This stage is the MNTB's own somatodendritic + calyx-terminal field.

CLI:
  python LFP_reconstruction/main_reconstruct_mntb.py --pic-file RESULTS/<f>.pic \
      --angle 0 --side L --n-cells 100 [--with-calyx]
  mpiexec -n 4 python LFP_reconstruction/main_reconstruct_mntb.py ... --n-cells 3600

Outputs -> RESULTS/lfp_tmp/output_mntb_{stem}_angle{A}_{S}/figures/
           mntb_lfp_reconstruction.png, mntb_lfp_single_cells.png,
           mntb_lfp_phase_cycle.png  (+ output_mntb_calyx_* and a combined
           figure when --with-calyx)
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
# Paths + mechanisms
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNTB_DIR  = os.path.join(REPO_ROOT, 'MNTB_models')
HOC_FILE  = os.path.join(MNTB_DIR, 'mntb_model_active.hoc')

# namntb lives in MNTB_models; klt/kht/ih in MSO_models (also auto-loaded from CWD).
for _d in (MNTB_DIR, os.path.join(REPO_ROOT, 'MSO_models')):
    try:
        neuron.load_mechanisms(_d)
    except RuntimeError as _e:
        if 'already exists' not in str(_e):
            raise

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DT     = 0.026   # ms
TSTOP  = 50.0    # ms
V_INIT = -70.0   # mV  (E_L.MNTBC from params.py)

N_MNTB_TOTAL = 3600   # MNTB principal neurons per side (params.py POP_NUM.n_MNTBCs)

# ---------------------------------------------------------------------------
# MNTB elliptic-cylinder geometry (human; Kulesza 2015, Karadas 2021)
#   ~4 mm rostrocaudal, coextensive with MSO; tonotopy medial->lateral.
#   x = dorsoventral/tonotopic half-axis, y = rostrocaudal half-axis.
# ---------------------------------------------------------------------------
ELLIPSE_RADIUS_X = 200.0    # um  tonotopic half-axis (medial-lateral extent)
ELLIPSE_RADIUS_Y = 2000.0   # um  rostrocaudal half-axis (~4 mm span)

# ---------------------------------------------------------------------------
# Probe geometry (z = dendritic / probe-depth axis)
# ---------------------------------------------------------------------------
N_CH    = 16
PROBE_Z = np.linspace(-300, 300, N_CH)   # um  (brackets the +-67 um dendrite tips)
PROBE_X = np.zeros(N_CH)
PROBE_Y = np.zeros(N_CH)
SIGMA   = 0.3   # S/m

# ---------------------------------------------------------------------------
# Layer boundaries (z; must match section extents in mntb_model_active.hoc)
#   Layer 0: dend_A (+z, 10..67)
#   Layer 1: dend_B (-z, -67..-10)
#   Layer 2: soma   (-10..10)  <- the calyx (axosomatic) lands here
# ---------------------------------------------------------------------------
LAYER_BOUNDARIES = [
    [  10.0,  67.0],   # dend_A (+ branches)
    [ -67.0, -10.0],   # dend_B (+ branches)
    [ -10.0,  10.0],   # soma
]

# Calyx-of-Held EPSC onto the MNTB soma (params.py: TAUS_EX_RISE/DECAY.MNTBC =
# 0.1/0.17 ms — the fastest synapse in the model; GBCs2MNTBCs delay 0.5 ms).
CALYX_SYN = {
    'GBC': {
        'syntype': 'Exp2Syn',
        'tau1':    0.1,     # ms  TAUS_EX_RISE.MNTBC
        'tau2':    0.17,    # ms  TAUS_EX_DECAY.MNTBC
        'e':       0.0,     # mV
        'weight':  0.050,   # uS  (tunable; suprathreshold calyx)
    },
}


# ---------------------------------------------------------------------------
# MNTBPopulation subclass
# ---------------------------------------------------------------------------
class MNTBPopulation(Population):
    """MNTB principal cell: tonotopic calyx (GBC_contra) assignment onto the soma."""

    PER_POP_SYN = CALYX_SYN

    def __init__(self, n_syn_per_pop=None, **kwargs):
        self.n_syn_per_pop = n_syn_per_pop or {}
        super().__init__(**kwargs)

    def get_all_SpCells(self):
        """Tonotopic x_to_one assignment: MNTB cell -> contralateral GBC (1:1)."""
        n_cells = self.POPULATION_SIZE
        SpCells = {}
        for cellindex in self.RANK_CELLINDICES:
            mntb_idx = (int(round(cellindex * (N_MNTB_TOTAL - 1) / (n_cells - 1)))
                        if n_cells > 1 else 0)
            SpCells[cellindex] = {}
            for X in self.X:
                nodes = self.networkSim.nodes[X]
                N_pre = len(nodes)
                n_src = self.n_syn_per_pop.get(X, 1)
                step      = (N_pre - n_src) / max(N_MNTB_TOTAL - 1, 1)
                pre_start = min(int(round(mntb_idx * step)), N_pre - n_src)
                window    = nodes[pre_start: pre_start + n_src]

                SpCell, src_used = [], 0
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
        soma_segs = cell.get_idx('soma')
        for X in self.X:
            pop_type = X.rsplit('_', 1)[0]
            for j in range(len(self.synIdx[cellindex][X])):
                idx = self.synIdx[cellindex][X][j]
                synDelays = (self.synDelays[cellindex][X][j]
                             if self.synDelays is not None else None)
                if len(idx) == 0:
                    continue
                # Calyx of Held is axosomatic -> place uniformly on the soma.
                if pop_type == 'GBC' and len(soma_segs) > 0:
                    idx = np.random.choice(soma_segs, size=len(idx),
                                           replace=True).astype('int32')
                self.insert_synapses(
                    cell=cell, cellindex=cellindex,
                    synParams=self.PER_POP_SYN[pop_type].copy(),
                    idx=idx, X=X, SpCell=self.SpCells[cellindex][X][j],
                    synDelays=synDelays,
                )

    def draw_rand_pos(self, radius_x=ELLIPSE_RADIUS_X, radius_y=ELLIPSE_RADIUS_Y,
                      z_min=0.0, z_max=0.0, min_cell_interdist=1.0, **kwargs):
        """Uniform sampling inside the MNTB elliptic cylinder (z collapsed)."""
        N = self.POPULATION_SIZE
        def _fill(n):
            return ((np.random.rand(n) - 0.5) * 2 * radius_x,
                    (np.random.rand(n) - 0.5) * 2 * radius_y,
                    np.random.rand(n) * (z_max - z_min) + z_min)
        x, y, z = _fill(N)
        outside = np.where((x / radius_x)**2 + (y / radius_y)**2 > 1)[0]
        while len(outside):
            x[outside], y[outside], z[outside] = _fill(len(outside))
            outside = np.where((x / radius_x)**2 + (y / radius_y)**2 > 1)[0]
        too_close = np.where(self.calc_min_cell_interdist(x, y, z) < min_cell_interdist)[0]
        while len(too_close):
            x[too_close], y[too_close], z[too_close] = _fill(len(too_close))
            outside = np.where((x / radius_x)**2 + (y / radius_y)**2 > 1)[0]
            while len(outside):
                x[outside], y[outside], z[outside] = _fill(len(outside))
                outside = np.where((x / radius_x)**2 + (y / radius_y)**2 > 1)[0]
            too_close = np.where(
                self.calc_min_cell_interdist(x, y, z) < min_cell_interdist)[0]
        soma_pos = [{'x': x[i], 'y': y[i], 'z': z[i]} for i in range(N)]
        soma_pos.sort(key=lambda p: p['x'])   # tonotopic order (x)
        return soma_pos


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='MNTB LFP reconstruction')
    parser.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',    type=int, default=0)
    parser.add_argument('--side',     type=str, default='L', choices=['L', 'R'])
    parser.add_argument('--n-cells',  type=int, default=100, dest='n_cells')
    parser.add_argument('--n-single', type=int, default=5, dest='n_single')
    parser.add_argument('--with-calyx', action='store_true', dest='with_calyx',
                        help='also simulate the co-located calyx-of-Held prespike '
                             'generator and sum its LFP (Phase 3)')
    args = parser.parse_args()

    side        = args.side
    contra_side = 'R' if side == 'L' else 'L'
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

    # MNTB is driven by the CONTRALATERAL GBC (decussating calyx).
    X_pops      = [f'GBC_{contra_side}']
    k_yxl_local = [[0], [0], [1]]        # dend_A, dend_B, soma <- 1 calyx on soma
    j_yx_local  = [0.050]
    tau_yx_local = [0.17]
    syn_delay_loc   = [0.5]              # params.py SYN_DELAYS.GBCs2MNTBCs
    syn_delay_scale = [None]

    stem       = _pic_stem(pic_file)
    spikes_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'spikes_{stem}_angle{args.angle}_{side}')
    output_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'output_mntb_{stem}_angle{args.angle}_{side}')
    for sub in ('cells', 'figures', 'populations'):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    k_arr         = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    _run_population(
        MNTBPopulation, HOC_FILE, X_pops, meta, spikes_dir, output_dir,
        k_yxl_local, j_yx_local, tau_yx_local, syn_delay_loc, syn_delay_scale,
        n_syn_per_pop, args, seed=46, title='MNTB', fig_prefix='mntb',
        per_pop_syn=None, v_init=V_INIT)

    # ----- Phase 3: co-located calyx prespike generator -----
    if args.with_calyx:
        import main_reconstruct_calyx as calyx_mod
        calyx_mod.run_calyx(args, meta, spikes_dir, stem, contra_side)
        if RANK == 0:
            _plot_combined(output_dir, args, stem)


def _run_population(PopClass, hoc_file, X_pops, meta, spikes_dir, output_dir,
                    k_yxl_local, j_yx_local, tau_yx_local, syn_delay_loc,
                    syn_delay_scale, n_syn_per_pop, args, seed, title, fig_prefix,
                    per_pop_syn=None, v_init=V_INIT):
    """Shared hybridLFPy run + plotting for a hand-written-hoc population."""
    from main_reconstruct_lso import (_plot_lfp, _plot_single_cells,
                                       _plot_phase_cycle)

    side = args.side
    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT, spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops,
    )
    probe = lfpykit_models.PointSourcePotential(
        cell=None, x=PROBE_X, y=PROBE_Y, z=PROBE_Z, sigma=SIGMA)

    pop_label = f'{title}_{side}'
    pop = PopClass(
        n_syn_per_pop=n_syn_per_pop, y=pop_label,
        cellParams={'morphology': hoc_file, 'passive': False, 'v_init': v_init,
                    'dt': DT, 'tstart': 0., 'tstop': TSTOP, 'nsegs_method': None},
        rand_rot_axis=['z'],
        simulationParams={'rec_imem': True},
        populationParams={'number': args.n_cells,
                          'radius': ELLIPSE_RADIUS_Y,
                          'radius_x': ELLIPSE_RADIUS_X, 'radius_y': ELLIPSE_RADIUS_Y,
                          'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
                          'min_r': np.array([[0.], [0.]])},
        layerBoundaries=LAYER_BOUNDARIES, probes=[probe], savelist=['somapos'],
        savefolder=output_dir, dt_output=DT, POPULATIONSEED=seed, X=X_pops,
        networkSim=networkSim, k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=syn_delay_loc, synDelayScale=syn_delay_scale,
        J_yX=j_yx_local, tau_yX=tau_yx_local,
    )
    if per_pop_syn is not None:
        pop.PER_POP_SYN = per_pop_syn

    pop.run()
    COMM.Barrier()

    n_grab = min(args.n_single, len(pop.RANK_CELLINDICES))
    cell_indices = random.sample(list(pop.RANK_CELLINDICES), n_grab)
    single_contribs = np.stack(
        [pop.output[i]['PointSourcePotential'] * 1e3 for i in cell_indices], axis=0)
    soma_pos = np.array([[pop.pop_soma_pos[i]['x'], pop.pop_soma_pos[i]['y'],
                          pop.pop_soma_pos[i]['z']] for i in cell_indices])

    pop.collect_data()
    COMM.Barrier()

    postproc = hybridLFPy.PostProcess(
        y=[pop_label], dt_output=DT, mapping_Yy=[(pop_label, pop_label)],
        savelist=['somapos'], probes=[probe], savefolder=output_dir)
    if RANK == 0:
        postproc.run()
    COMM.Barrier()

    if RANK == 0:
        tvec = np.arange(single_contribs.shape[2]) * DT
        _plot_lfp(output_dir, N_CH, PROBE_Z, side, args.angle, args.n_cells)
        _plot_single_cells(output_dir, single_contribs, tvec, PROBE_Z, soma_pos,
                           PROBE_X, PROBE_Y, cell_gids=cell_indices,
                           total_sim_cells=args.n_cells)
        _plot_phase_cycle(output_dir, meta.get('stim_freq_hz'), PROBE_Z,
                          side, args.angle, args.n_cells)
        # rename lso_*.png -> {fig_prefix}_*.png
        figdir = os.path.join(output_dir, 'figures')
        for a, b in (('lso_lfp_reconstruction.png', f'{fig_prefix}_lfp_reconstruction.png'),
                     ('lso_lfp_phase_cycle.png',    f'{fig_prefix}_lfp_phase_cycle.png'),
                     ('lso_lfp_single_cells.png',   f'{fig_prefix}_lfp_single_cells.png')):
            src = os.path.join(figdir, a)
            if os.path.exists(src):
                os.replace(src, os.path.join(figdir, b))


def _plot_combined(output_dir, args, stem):
    """Sum principal + calyx PointSourcePotential and plot the composite LFP."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    calyx_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                             f'output_mntb_calyx_{stem}_angle{args.angle}_{args.side}')
    p_princ = os.path.join(output_dir, 'PointSourcePotential_sum.h5')
    p_calyx = os.path.join(calyx_dir, 'PointSourcePotential_sum.h5')
    if not (os.path.exists(p_princ) and os.path.exists(p_calyx)):
        print('Warning: missing principal or calyx LFP; skipping combined plot.')
        return
    with h5py.File(p_princ, 'r') as f:
        princ = f['data'][()] * 1e3; srate = float(f['srate'][()])
    with h5py.File(p_calyx, 'r') as f:
        calyx = f['data'][()] * 1e3
    comp = princ + calyx
    tvec = np.arange(comp.shape[1]) / srate * 1e3

    best = int(np.argmax(np.abs(comp).max(axis=1)))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(tvec, princ[best], color='steelblue', lw=1.0, label='principal (postsynaptic)')
    ax.plot(tvec, calyx[best], color='darkorange', lw=1.0, label='calyx (prespike)')
    ax.plot(tvec, comp[best],  color='k', lw=1.3, label='composite')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('LFP (uV)')
    ax.set_title(f'MNTB composite LFP (probe z={PROBE_Z[best]:.0f} um) | '
                 f'side {args.side} | angle {args.angle}')
    ax.legend(fontsize=8)
    out = os.path.join(output_dir, 'figures', 'mntb_lfp_composite.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f'Combined LFP figure saved -> {out}')


if __name__ == '__main__':
    main()
