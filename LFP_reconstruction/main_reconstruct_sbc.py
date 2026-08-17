#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
HybridLFPy LFP reconstruction for the AVCN *spherical* bushy cell (SBC) population.

Parallel to main_reconstruct_avcn.py (the globular bushy cell / GBC pipeline),
reusing the same AVCNPopulation subclass and plotting.  Only the SBC-specific
deltas are defined here:

  morphology    real SBC EM reconstruction SBC_S113.hoc (NeuroMorpho Atoh7+,
                converted by AVCN_models/swc_to_hoc.py) vs GBC's VCN_c09
  channels      cnmodel XM13_nacncoop Type II-I (less KLT/Na -> higher input
                resistance) vs GBC's Type II
  convergence   3 ANF endbulbs/SBC (params.py ANFs2SBCs) vs 20/GBC
  endbulbs      few, large, axosomatic (85% soma / 15% proximal dendrite) vs
                GBC's 70/20/5/5 modified endbulbs
  population    28,000 SBC/side (params.py n_SBCs) vs 3,600 GBC
  geometry      rostral spherical-cell area ~1.5-2.0 mm rostrocaudal ->
                ELLIPSE_RADIUS_Y = 875 µm vs GBC's 600

Presynaptic drive, taus, coherent ventromedial axon orientation, probe and
tonotopy are identical to the GBC pipeline.

CLI (single or MPI):
  python LFP_reconstruction/main_reconstruct_sbc.py --pic-file RESULTS/x.pic \
      --angle 0 --side L --n-cells 100
  mpiexec -n 4 python LFP_reconstruction/main_reconstruct_sbc.py ... --n-cells 3600

Outputs -> RESULTS/lfp_tmp/output_sbc_{stem}_angle{A}_{S}/figures/:
  sbc_lfp_reconstruction.png, sbc_lfp_phase_cycle.png, sbc_lfp_single_cells.png
"""

import os
import sys
import random

import numpy as np
import h5py

import LFPy
import lfpykit.models as lfpykit_models
import hybridLFPy
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'LFP_reconstruction'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'AVCN_models'))

# Reuse the whole GBC infrastructure (mechanisms are loaded on import of the
# AVCN module).  We import the class, the plotting, and the shared constants.
import gbc_biophysics                                    # noqa: E402
from main_reconstruct_avcn import (                       # noqa: E402
    AVCNPopulation, _plot_lfp, _plot_phase_cycle, _plot_single_cells,
    DT, TSTOP, V_INIT, N_CH, PROBE_X, PROBE_Y, PROBE_Z, SIGMA,
    ELLIPSE_RADIUS_X, AXON_TARGET, LAYER_BOUNDARIES,
    SYN_DELAY_LOC, SYN_DELAY_SCALE, AVCN_DIR,
)

# ---------------------------------------------------------------------------
# SBC-specific parameters
# ---------------------------------------------------------------------------
HOC_FILE_SBC = os.path.join(AVCN_DIR, 'morphology', 'neuromorpho', 'SBC_S113.hoc')

N_CELLS       = 100
N_SBC_TOTAL   = 28000    # SBC per side in NEST sim (params.py n_SBCs)
N_ENDBULBS    = 3        # ANF endbulbs per SBC (params.py ANFs2SBCs)

ELLIPSE_RADIUS_Y = 875.0   # µm  rostrocaudal half-axis (spherical-cell area spans
                           #     the rostral ~1.5-2.0 mm of VCN -> half-extent ~875)

# Few, large, axosomatic endbulbs of Held: place mostly on the soma.
ENDBULB_WEIGHTS_SBC = {'soma': 0.85, 'primarydendrite': 0.15}

# Endbulb synapse: same fast AMPA kinetics as GBC (TAUS_EX_RISE/DECAY.SBC =
# 0.2/0.5 ms in params.py, identical to GBC), larger unit conductance because an
# SBC has only 3 (large) endbulbs vs 20 modified endbulbs on a GBC.
PER_POP_SYN_SBC = {
    'ANF': {
        'syntype': 'Exp2Syn',
        'tau1':    0.2,     # ms  TAUS_EX_RISE.SBC
        'tau2':    0.5,     # ms  TAUS_EX_DECAY.SBC
        'e':       0.0,     # mV  excitatory reversal
        'weight':  0.030,   # µS  large axosomatic endbulb (tunable; scales LFP)
    },
}

K_YXL = [[N_ENDBULBS]]     # 3 ANF endbulbs / SBC


def _decorate_sbc(cell):
    """LFPy custom_fun: cnmodel Type II-I (SBC) channel densities."""
    gbc_biophysics.decorate_gbc(cell, set_nseg=False,
                                ref_ns=gbc_biophysics.REF_NS_II_I)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='AVCN (SBC) LFP reconstruction')
    parser.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',    type=int, default=0)
    parser.add_argument('--side',     type=str, default='L', choices=['L', 'R'])
    parser.add_argument('--n-cells',  type=int, default=N_CELLS, dest='n_cells')
    parser.add_argument('--n-single', type=int, default=5, dest='n_single')
    parser.add_argument('--hoc-file', type=str, default=HOC_FILE_SBC, dest='hoc_file',
                        help='SBC morphology hoc (default: SBC_S113 reconstruction)')
    args = parser.parse_args()

    side     = args.side
    pic_file = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                             'baseline_simulation.pic')

    from main_reconstruct import _pic_stem, _extract_spikes

    if RANK == 0:
        meta = _extract_spikes(args.angle, side, pic_file=pic_file)
    else:
        meta = None
    meta = COMM.bcast(meta, root=0)
    COMM.Barrier()

    X_pops       = [f'ANF_{side}']
    j_yx_local   = [PER_POP_SYN_SBC['ANF']['weight']]
    tau_yx_local = [PER_POP_SYN_SBC['ANF']['tau2']]

    stem       = _pic_stem(pic_file)
    spikes_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'spikes_{stem}_angle{args.angle}_{side}')
    output_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'output_sbc_{stem}_angle{args.angle}_{side}')
    for sub in ('cells', 'figures', 'populations'):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    k_arr         = np.array(K_YXL)
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
    pop_label = f'SBC_{side}'
    pop = AVCNPopulation(
        n_syn_per_pop=n_syn_per_pop,
        axon_target=AXON_TARGET,
        n_post_total=N_SBC_TOTAL,
        n_endbulbs=N_ENDBULBS,
        endbulb_weights=ENDBULB_WEIGHTS_SBC,
        per_pop_syn=PER_POP_SYN_SBC,
        y=pop_label,
        cellParams={
            'morphology': args.hoc_file, 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP,
            'nsegs_method': 'lambda_f', 'lambda_f': 100,
            'custom_fun': [_decorate_sbc], 'custom_fun_args': [{}],
        },
        rand_rot_axis=[],
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
        POPULATIONSEED=45,      # distinct from GBC (44)
        X=X_pops,
        networkSim=networkSim,
        k_yXL=K_YXL,
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
        _plot_lfp(output_dir, N_CH, PROBE_Z, side, args.angle, args.n_cells,
                  title_prefix='AVCN (SBC)')
        _plot_phase_cycle(output_dir, meta.get('stim_freq_hz'), PROBE_Z,
                          side, args.angle, args.n_cells)
        if single_contribs is not None:
            tvec = np.arange(single_contribs.shape[2]) * DT
            #_plot_single_cells(output_dir, single_contribs, tvec, PROBE_Z,
                               #soma_pos, PROBE_X, PROBE_Y, cell_indices, args.n_cells)
        # rename the GBC-titled figures to sbc_* for clarity
        figdir = os.path.join(output_dir, 'figures')
        for a, b in (('avcn_lfp_reconstruction.png', 'sbc_lfp_reconstruction.png'),
                     ('avcn_lfp_phase_cycle.png',    'sbc_lfp_phase_cycle.png'),
                     ('avcn_lfp_single_cells.png',   'sbc_lfp_single_cells.png')):
            src = os.path.join(figdir, a)
            if os.path.exists(src):
                os.replace(src, os.path.join(figdir, b))


if __name__ == '__main__':
    main()
