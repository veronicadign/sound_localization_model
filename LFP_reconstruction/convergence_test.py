#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
Convergence test: run MSOPopulation for N = 100, 500, 1000, 2000 cells.
For each N, compute the compound LFP, isolate the max-amplitude channel,
and overlay normalised traces on a single plot.

Output: RESULTS/lfp_tmp/convergence/convergence_test.png
"""

import os
import sys
import json

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- bring in shared setup from main_reconstruct --------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_reconstruct import (
    MSOPopulation,
    HOC_FILE, SPIKES_DIR, META_FILE,
    DT, TSTOP, V_INIT,
    N_CH, PROBE_X, PROBE_Y, PROBE_Z, SIGMA,
    LAYER_BOUNDARIES, K_YXL,
    SYN_DELAY_LOC, SYN_DELAY_SCALE,
)

import hybridLFPy
import lfpykit.models as lfpykit_models
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONV_DIR    = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp', 'convergence')
N_VALUES    = [100, 500, 1000, 2000]

os.makedirs(CONV_DIR, exist_ok=True)


def run_for_n(n_cells, networkSim, meta):
    """Run one Population simulation for n_cells; return compound LFP (n_ch, n_t)."""
    out_dir = os.path.join(CONV_DIR, f'N{n_cells}')
    os.makedirs(out_dir, exist_ok=True)
    # sub-dirs expected by PostProcess
    for sub in ('cells', 'figures', 'populations'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    probe = lfpykit_models.PointSourcePotential(
        cell=None, x=PROBE_X, y=PROBE_Y, z=PROBE_Z, sigma=SIGMA,
    )

    X_pops = ['SBC_L', 'MNTBC_L', 'LNTBC_L']

    pop = MSOPopulation(
        y='MSO_L',
        cellParams={
            'morphology':   HOC_FILE,
            'passive':      False,
            'v_init':       V_INIT,
            'dt':           DT,
            'tstart':       0.,
            'tstop':        TSTOP,
            'nsegs_method': None,
        },
        rand_rot_axis=['z'],
        simulationParams={'rec_imem': True},
        populationParams={
            'number':            n_cells,
            'radius':            100.,
            'z_min':             -50.,
            'z_max':              50.,
            'min_cell_interdist':  5.,
            'min_r':             np.array([[-50., 50.], [0., 0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=[probe],
        savelist=['somapos'],
        savefolder=out_dir,
        dt_output=DT,
        POPULATIONSEED=42,
        X=X_pops,
        networkSim=networkSim,
        k_yXL=K_YXL,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=SYN_DELAY_LOC,
        synDelayScale=SYN_DELAY_SCALE,
        J_yX=[0.012, 0.010, 0.010],
        tau_yX=[0.3, 0.7, 0.7],
    )

    pop.run()
    COMM.Barrier()
    pop.collect_data()
    COMM.Barrier()

    postproc = hybridLFPy.PostProcess(
        y=['MSO_L'], dt_output=DT,
        mapping_Yy=[('MSO_L', 'MSO_L')],
        savelist=['somapos'], probes=[probe],
        savefolder=out_dir,
    )
    if RANK == 0:
        postproc.run()
    COMM.Barrier()

    with h5py.File(os.path.join(out_dir, 'PointSourcePotential_sum.h5'), 'r') as f:
        lfp = f['data'][()]   # (n_ch, n_t)  mV

    if RANK == 0:
        print(f'  N={n_cells:5d}: peak RMS = {np.sqrt((lfp**2).mean(axis=1)).max()*1e3:.3f} uV')
    return lfp


def main():
    with open(META_FILE) as f:
        meta = json.load(f)

    X_pops = ['SBC_L', 'MNTBC_L', 'LNTBC_L']

    # Build CachedNetwork once — shared across all N runs
    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT,
        spike_output_path=SPIKES_DIR,
        label='spikes', ext='gdf',
        GIDs={
            'SBC_L':   [meta['SBC_L']['first_gid'],   meta['SBC_L']['n_neurons']],
            'MNTBC_L': [meta['MNTBC_L']['first_gid'], meta['MNTBC_L']['n_neurons']],
            'LNTBC_L': [meta['LNTBC_L']['first_gid'], meta['LNTBC_L']['n_neurons']],
        },
        X=X_pops,
    )

    results = {}
    for n in N_VALUES:
        if RANK == 0:
            print(f'\n--- N = {n} ---')
        results[n] = run_for_n(n, networkSim, meta)

    if RANK == 0:
        _plot(results)


def _plot(results):
    # Find the channel with max RMS across the largest N run
    lfp_ref   = results[N_VALUES[-1]]
    best_ch   = int(np.argmax(np.sqrt((lfp_ref**2).mean(axis=1))))
    best_z    = PROBE_Z[best_ch]
    srate     = 1e3 / DT               # Hz
    tvec      = np.arange(lfp_ref.shape[1]) / srate * 1e3   # ms

    colors = ['#b0b0b0', '#7ec8e3', '#2176ae', '#d62828']   # light -> dark

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ----- Panel 1: normalised traces (shape convergence) -----
    ax = axes[0]
    for color, n in zip(colors, N_VALUES):
        trace = results[n][best_ch] * 1e3   # mV -> uV
        # normalise by N so traces are comparable in shape
        trace_norm = trace / n
        ax.plot(tvec, trace_norm, color=color, lw=1.2, label=f'N = {n}')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('LFP / N  (μV per cell)')
    ax.set_title(f'Shape convergence  —  ch {best_ch} (z = {best_z:.0f} μm)')
    ax.legend(frameon=False)

    # ----- Panel 2: peak-to-peak amplitude vs N (linearity check) -----
    ax2 = axes[1]
    ptp = [np.ptp(results[n][best_ch]) * 1e3 for n in N_VALUES]   # uV
    ax2.plot(N_VALUES, ptp, 'o-', color='#2176ae', lw=1.5, ms=6)
    # fit a line to check linearity
    coeffs  = np.polyfit(N_VALUES, ptp, 1)
    ptp_fit = np.polyval(coeffs, N_VALUES)
    ax2.plot(N_VALUES, ptp_fit, '--', color='#d62828', lw=1, label=f'linear fit  (R²={np.corrcoef(N_VALUES, ptp)[0,1]**2:.4f})')
    ax2.set_xlabel('N cells')
    ax2.set_ylabel('Peak-to-peak LFP (μV)')
    ax2.set_title('Amplitude scaling')
    ax2.legend(frameon=False)

    fig.suptitle('MSO LFP convergence test  (angle=0, side=L, 0.5 kHz tone)', y=1.01)
    fig.tight_layout()
    out = os.path.join(CONV_DIR, 'convergence_test.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved -> {out}')


if __name__ == '__main__':
    main()
