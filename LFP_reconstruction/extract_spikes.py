"""
Extract MSO presynaptic spike trains from a .pic file -> GDF files.

Output (in spikes_dir/):
  spikes-SBC_{side}-0.gdf    tab-separated: neuron_id  spike_time_ms
  spikes-MNTBC_{side}-0.gdf
  spikes-LNTBC_{side}-0.gdf
  metadata.json               GID info for CachedNetwork setup
"""

import os
import sys
import json

import dill
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'simulate'))

POPS = ['SBC', 'MNTBC', 'LNTBC']


def _write_pop(data_side, pop, label, spikes_dir, metadata):
    """Extract one population from data_side dict, write GDF, update metadata."""
    pop_data   = data_side[pop]
    times      = np.array(pop_data['times'],      dtype=float)
    senders    = np.array(pop_data['senders'],    dtype=int)
    global_ids = np.array(pop_data['global_ids'], dtype=int)

    sorted_ids = np.sort(global_ids)
    id_map     = {gid: i + 1 for i, gid in enumerate(sorted_ids)}
    remapped   = np.array([id_map[s] for s in senders], dtype=int)

    gdf_path = os.path.join(spikes_dir, f'spikes-{label}-0.gdf')
    arr = np.column_stack([remapped, times])
    np.savetxt(gdf_path, arr, fmt=['%d', '%.4f'], delimiter='\t')
    print(f'  {label}: {len(arr):,} spikes -> {gdf_path}')

    metadata[label] = {
        'first_gid': 1,
        'n_neurons':  int(len(global_ids)),
        'n_spikes':   int(len(times)),
        't_min_ms':   float(times.min()) if len(times) > 0 else None,
        't_max_ms':   float(times.max()) if len(times) > 0 else None,
    }


def extract_and_save(pic_file, angle, side, spikes_dir):
    """
    Load pic_file, extract presynaptic spikes for the given angle/side,
    write GDF files and metadata.json to spikes_dir.

    Extracts 4 populations:
      SBC_{side}        — ipsilateral SBC (medial dendrite input)
      SBC_{contra_side} — contralateral SBC (lateral dendrite input)
      MNTBC_{side}      — ipsilateral MNTBC (soma inhibition)
      LNTBC_{side}      — ipsilateral LNTBC (soma inhibition)

    Returns
    -------
    metadata : dict  {pop_label: {first_gid, n_neurons, n_spikes, t_min_ms, t_max_ms}}
    """
    os.makedirs(spikes_dir, exist_ok=True)
    contra_side = 'R' if side == 'L' else 'L'

    print(f'Loading {pic_file} ...', flush=True)
    with open(pic_file, 'rb') as f:
        result = dill.load(f, ignore=True)

    data_ipsi   = result['angle_to_rate'][angle][side]
    data_contra = result['angle_to_rate'][angle][contra_side]
    metadata    = {}

    # Ipsilateral SBC + inhibitory pops
    for pop in POPS:
        _write_pop(data_ipsi, pop, f'{pop}_{side}', spikes_dir, metadata)

    # Contralateral SBC (lateral dendrite excitation)
    _write_pop(data_contra, 'SBC', f'SBC_{contra_side}', spikes_dir, metadata)

    meta_path = os.path.join(spikes_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Metadata -> {meta_path}')

    return metadata


if __name__ == '__main__':
    pic_file   = os.path.join(REPO_ROOT, 'RESULTS', 'baseline_simulation.pic')
    spikes_dir = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp', 'spikes')
    extract_and_save(pic_file, angle=0, side='L', spikes_dir=spikes_dir)