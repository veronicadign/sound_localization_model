#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
HybridLFPy LFP reconstruction for MSO population (angle=0, side=L).

Pipeline:
  CachedNetwork (SBC_L + MNTBC_L + LNTBC_L spikes from GDF)
  -> MSOPopulation (100 multicompartment cells driven by presynaptic Exp2Syn)
  -> PostProcess  (sum contributions across cells)
  -> PointSourcePotential_sum.h5 + figures/mso_lfp_reconstruction.png

Run as:
  python LFP_reconstruction/main_reconstruct.py
  mpiexec -n 4 python LFP_reconstruction/main_reconstruct.py
"""

import os
import sys
import json

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import LFPy
import lfpykit.models as lfpykit_models
import hybridLFPy
from hybridLFPy.population import Population
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOC_FILE   = os.path.join(REPO_ROOT, 'MSO_models', 'mso_model.hoc')
SPIKES_DIR = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp', 'spikes')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp', 'output')
META_FILE  = os.path.join(SPIKES_DIR, 'metadata.json')

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DT      = 0.0625   # ms
TSTOP   = 50.0     # ms  (matches actual spike data duration)
V_INIT  = -57.0    # mV  (E_L.MSO)
N_CELLS = 100      # representative MSO cells (full population has 15500)

# ---------------------------------------------------------------------------
# Probe: 16-channel linear array along z-axis
# ---------------------------------------------------------------------------
N_CH    = 16
PROBE_Z = np.linspace(-400, 400, N_CH)   # μm
PROBE_X = np.zeros(N_CH)
PROBE_Y = np.zeros(N_CH)
SIGMA   = 0.3                            # S/m, extracellular conductivity

# ---------------------------------------------------------------------------
# Connectivity: layerBoundaries and k_yXL
# Layers:   [medial_dend z∈[10,160]], [lateral_dend z∈[-160,-10]], [soma z∈[-10,10]]
# Pops:     [SBC_L, MNTBC_L, LNTBC_L]
# From params.py: SBCs2MSOs=3, MNTBCs2MSOs=2, LNTBCs2MSOs=1
# ---------------------------------------------------------------------------
LAYER_BOUNDARIES = [
    [ 10.,  160.],   # medial dendrite
    [-160., -10.],   # lateral dendrite
    [ -10.,  10.],   # soma
]

K_YXL = [
    [2, 0, 0],   # medial dend:   2 SBC,  0 MNTBC, 0 LNTBC
    [1, 0, 0],   # lateral dend:  1 SBC,  0 MNTBC, 0 LNTBC
    [0, 2, 1],   # soma:          0 SBC,  2 MNTBC, 1 LNTBC
]

# Synapse delays [ms]: from params.py SYN_DELAYS
SYN_DELAY_LOC   = [2.0, 1.0, 1.0]   # [SBC_L, MNTBC_L, LNTBC_L]
SYN_DELAY_SCALE = [None, None, None]  # fixed delays (no variance)


# ---------------------------------------------------------------------------
# MSOPopulation: subclass to support per-population Exp2Syn params
# The base insert_all_synapses only updates weight/tau (not tau1/tau2/e),
# so we override it with per-population synapse dicts.
# ---------------------------------------------------------------------------
class MSOPopulation(Population):
    """Population subclass with heterogeneous Exp2Syn params per presynaptic pop."""

    PER_POP_SYN = {
        'SBC_L': {
            'syntype': 'Exp2Syn',
            'tau1':    0.15,    # ms, rise  (TAUS_EX_RISE.MSO)
            'tau2':    0.3,     # ms, decay (TAUS_EX_DECAY.MSO)
            'e':       0.0,     # mV, excitatory reversal
            'weight':  0.012,   # μS = 12 nS (SYN_WEIGHTS.SBCs2MSO)
        },
        'MNTBC_L': {
            'syntype': 'Exp2Syn',
            'tau1':    0.15,    # ms (TAUS_IN_RISE.MSO)
            'tau2':    0.7,     # ms (TAUS_IN_DECAY.MSO)
            'e':      -75.0,    # mV, inhibitory reversal (INH_REV.MSO)
            'weight':  0.010,   # μS = 10 nS (SYN_WEIGHTS.MNTBCs2MSO magnitude)
        },
        'LNTBC_L': {
            'syntype': 'Exp2Syn',
            'tau1':    0.15,
            'tau2':    0.7,
            'e':      -75.0,
            'weight':  0.010,
        },
    }

    def insert_all_synapses(self, cellindex, cell):
        for X in self.X:
            for j in range(len(self.synIdx[cellindex][X])):
                synDelays = (self.synDelays[cellindex][X][j]
                             if self.synDelays is not None else None)
                self.insert_synapses(
                    cell=cell,
                    cellindex=cellindex,
                    synParams=self.PER_POP_SYN[X].copy(),
                    idx=self.synIdx[cellindex][X][j],
                    X=X,
                    SpCell=self.SpCells[cellindex][X][j],
                    synDelays=synDelays,
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Read GID metadata
    with open(META_FILE) as f:
        meta = json.load(f)

    X_pops = ['SBC_L', 'MNTBC_L', 'LNTBC_L']

    # --- CachedNetwork ---
    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP,
        dt=DT,
        spike_output_path=SPIKES_DIR,
        label='spikes',
        ext='gdf',
        GIDs={
            'SBC_L':   [meta['SBC_L']['first_gid'],   meta['SBC_L']['n_neurons']],
            'MNTBC_L': [meta['MNTBC_L']['first_gid'], meta['MNTBC_L']['n_neurons']],
            'LNTBC_L': [meta['LNTBC_L']['first_gid'], meta['LNTBC_L']['n_neurons']],
        },
        X=X_pops,
        autocollect=True,
        skiprows=0,
    )

    # --- Electrode probe (created once, cell set per-simulation by Population) ---
    probe = lfpykit_models.PointSourcePotential(
        cell=None,
        x=PROBE_X,
        y=PROBE_Y,
        z=PROBE_Z,
        sigma=SIGMA,
    )

    # --- Cell params ---
    cellParams = {
        'morphology':    HOC_FILE,
        'passive':       False,   # passive already inserted in HOC
        'v_init':        V_INIT,
        'dt':            DT,
        'tstart':        0.,
        'tstop':         TSTOP,
        'nsegs_method':  None,    # use nseg as defined in HOC
    }

    # --- Population ---
    pop = MSOPopulation(
        y='MSO_L',
        cellParams=cellParams,
        rand_rot_axis=['z'],
        simulationParams={'rec_imem': True},
        populationParams={
            'number':            N_CELLS,
            'radius':            100.,    # μm
            'z_min':             -50.,    # μm
            'z_max':              50.,    # μm
            'min_cell_interdist':  5.,
            # min_r: 2D array [[z_coords], [min_radii]], set to 0 = no inner exclusion zone
            'min_r':             np.array([[-50., 50.], [0., 0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=[probe],
        savelist=['somapos'],
        savefolder=OUTPUT_DIR,
        dt_output=DT,
        POPULATIONSEED=42,
        X=X_pops,
        networkSim=networkSim,
        k_yXL=K_YXL,
        # synParams 'section' key is the only field used by fetchSynIdxCell;
        # actual synapse params are set per-population in MSOPopulation.PER_POP_SYN
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=SYN_DELAY_LOC,
        synDelayScale=SYN_DELAY_SCALE,
        J_yX=[0.012, 0.010, 0.010],   # μS (unused by subclass, harmless)
        tau_yX=[0.3, 0.7, 0.7],
    )

    pop.run()
    COMM.Barrier()
    pop.collect_data()
    COMM.Barrier()

    # --- PostProcess: sum cell contributions -> compound LFP ---
    postproc = hybridLFPy.PostProcess(
        y=['MSO_L'],
        dt_output=DT,
        mapping_Yy=[('MSO_L', 'MSO_L')],
        savelist=['somapos'],
        probes=[probe],
        savefolder=OUTPUT_DIR,
        compound_file='{}_sum.h5',
        output_file='{}_population_{}',
    )
    if RANK == 0:
        postproc.run()
    COMM.Barrier()

    # --- Plot ---
    if RANK == 0:
        _plot_lfp(OUTPUT_DIR, N_CH, PROBE_Z)


def _plot_lfp(output_dir, n_ch, probe_z):
    h5_path = os.path.join(output_dir, 'PointSourcePotential_sum.h5')
    with h5py.File(h5_path, 'r') as f:
        lfp   = f['data'][()]        # (n_ch, n_t)  mV
        srate = float(f['srate'][()]) # Hz

    lfp_uv = lfp * 1e3               # convert to μV
    tvec   = np.arange(lfp.shape[1]) / srate * 1e3   # ms

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    # Panel 1: trace per channel, offset by z-position
    ax = axes[0]
    scale = max(np.abs(lfp_uv).max() * 2, 1e-9)
    for i in range(n_ch):
        ax.plot(tvec, lfp_uv[i] / scale * 50 + probe_z[i],
                color='k', lw=0.6, alpha=0.9)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Probe z (μm)')
    ax.set_title('MSO LFP traces (angle=0, L)')

    # Panel 2: colour map
    ax2 = axes[1]
    im = ax2.imshow(
        lfp_uv,
        aspect='auto', origin='lower',
        extent=[tvec[0], tvec[-1], probe_z[0], probe_z[-1]],
        cmap='RdBu_r',
    )
    plt.colorbar(im, ax=ax2, label='LFP (μV)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Probe z (μm)')
    ax2.set_title('LFP colour map')

    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'figures', 'mso_lfp_reconstruction.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Figure saved -> {fig_path}')


if __name__ == '__main__':
    main()
