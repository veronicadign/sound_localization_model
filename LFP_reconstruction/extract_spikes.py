#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
Extract MSO presynaptic spike trains from baseline_simulation.pic -> GDF files.

Output (in RESULTS/lfp_tmp/spikes/):
  spikes-SBC_L-0.gdf    tab-separated: neuron_id  spike_time_ms
  spikes-MNTBC_L-0.gdf
  spikes-LNTBC_L-0.gdf
  metadata.json          GID info for CachedNetwork setup
"""

import os
import sys
import json

import dill
import numpy as np

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'simulate'))

PIC_FILE   = os.path.join(REPO_ROOT, 'RESULTS', 'baseline_simulation.pic')
SPIKES_DIR = os.path.join(REPO_ROOT, 'RESULTS', 'lfp_tmp', 'spikes')

ANGLE = 0
SIDE  = 'L'
POPS  = ['SBC', 'MNTBC', 'LNTBC']

os.makedirs(SPIKES_DIR, exist_ok=True)

print(f'Loading {PIC_FILE} ...', flush=True)
with open(PIC_FILE, 'rb') as f:
    result = dill.load(f, ignore=True)

data = result['angle_to_rate'][ANGLE][SIDE]
metadata = {}

for pop in POPS:
    pop_data   = data[pop]
    times      = np.array(pop_data['times'],      dtype=float)
    senders    = np.array(pop_data['senders'],    dtype=int)
    global_ids = np.array(pop_data['global_ids'], dtype=int)

    # Remap NEST global IDs -> 1-based contiguous integers
    sorted_ids = np.sort(global_ids)
    id_map     = {gid: i + 1 for i, gid in enumerate(sorted_ids)}
    remapped   = np.array([id_map[s] for s in senders], dtype=int)

    pop_label = f'{pop}_{SIDE}'
    gdf_path  = os.path.join(SPIKES_DIR, f'spikes-{pop_label}-0.gdf')

    arr = np.column_stack([remapped, times])
    np.savetxt(gdf_path, arr, fmt=['%d', '%.4f'], delimiter='\t')
    print(f'  {pop_label}: {len(arr):,} spikes -> {gdf_path}')

    metadata[pop_label] = {
        'first_gid': 1,
        'n_neurons':  int(len(global_ids)),
        'n_spikes':   int(len(times)),
        't_min_ms':   float(times.min()) if len(times) > 0 else None,
        't_max_ms':   float(times.max()) if len(times) > 0 else None,
    }

meta_path = os.path.join(SPIKES_DIR, 'metadata.json')
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'\nMetadata -> {meta_path}')
