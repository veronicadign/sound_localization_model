#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
Co-located calyx-of-Held PRESPIKE population (Phase 3), summed with the MNTB
principal-cell LFP by linear superposition. Invoked by main_reconstruct_mntb.py
--with-calyx; not usually run standalone.

Each calyx is the presynaptic terminal of a contralateral GBC axon. Driven by the
GBC_{contra} spike train through a SUPRATHRESHOLD somatic-axon synapse so it fires
one presynaptic AP per input; its transmembrane current is the extracellular
prespike. Same probe + nucleus geometry as the principal cells, so the two LFPs
add exactly (output_mntb_calyx_* -> summed in main_reconstruct_mntb._plot_combined).

Cleft leak conductance g_cl ~ 1 uS is characterised single-cell in
MNTB_models/validate_calyx.py (V_cleft ~ several mV). The population prespike LFP
here is the calyx transmembrane current (the dominant term); cross-cell ephaptic
feedback onto the apposed MNTB soma is NOT modelled (LFPy cells are independent) —
a documented limitation.
"""
import os

import numpy as np

import main_reconstruct_mntb as mntb
from main_reconstruct_mntb import MNTBPopulation, _run_population, REPO_ROOT, COMM, RANK

CALYX_HOC = os.path.join(REPO_ROOT, 'MNTB_models', 'calyx_model.hoc')


class CalyxPopulation(MNTBPopulation):
    """Presynaptic calyx terminal: suprathreshold GBC drive onto the pre-calyx axon."""

    PER_POP_SYN = {
        'GBC': {
            'syntype': 'Exp2Syn',
            'tau1':    0.1,
            'tau2':    0.17,
            'e':       0.0,
            'weight':  0.150,   # uS  suprathreshold (fires the terminal once/input)
        },
    }

    def insert_all_synapses(self, cellindex, cell):
        pre_segs = cell.get_idx('precalyx_axon')
        for X in self.X:
            pop_type = X.rsplit('_', 1)[0]
            for j in range(len(self.synIdx[cellindex][X])):
                idx = self.synIdx[cellindex][X][j]
                synDelays = (self.synDelays[cellindex][X][j]
                             if self.synDelays is not None else None)
                if len(idx) == 0:
                    continue
                if pop_type == 'GBC' and len(pre_segs) > 0:
                    idx = np.random.choice(pre_segs, size=len(idx),
                                           replace=True).astype('int32')
                self.insert_synapses(
                    cell=cell, cellindex=cellindex,
                    synParams=self.PER_POP_SYN[pop_type].copy(),
                    idx=idx, X=X, SpCell=self.SpCells[cellindex][X][j],
                    synDelays=synDelays)


def run_calyx(args, meta, spikes_dir, stem, contra_side):
    X_pops      = [f'GBC_{contra_side}']
    k_yxl_local = [[0], [0], [1]]     # place the single synapse via override (soma layer)
    j_yx_local  = [0.150]
    tau_yx_local = [0.17]
    # The calyx AP occurs essentially AT the GBC spike (presynaptic), whereas the
    # postsynaptic MNTB EPSC carries the full GBCs2MNTBCs = 0.5 ms synaptic delay.
    # So the prespike LEADS the postsynaptic sink by ~0.5 ms (the in-vivo prespike).
    syn_delay_loc   = [0.05]
    syn_delay_scale = [None]

    output_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp',
                              f'output_mntb_calyx_{stem}_angle{args.angle}_{args.side}')
    for sub in ('cells', 'figures', 'populations'):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    k_arr = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    _run_population(
        CalyxPopulation, CALYX_HOC, X_pops, meta, spikes_dir, output_dir,
        k_yxl_local, j_yx_local, tau_yx_local, syn_delay_loc, syn_delay_scale,
        n_syn_per_pop, args, seed=47, title='CALYX', fig_prefix='calyx',
        v_init=-70.0)
