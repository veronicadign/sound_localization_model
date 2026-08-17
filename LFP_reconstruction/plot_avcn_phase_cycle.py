#!/home/verodige/miniforge3/envs/sl_env/bin/python
"""
Regenerate the AVCN phase-cycle LFP plot from a FINISHED run — no re-simulation.

Reads output_dir/PointSourcePotential_sum.h5 (saved by main_reconstruct_avcn.py)
and the stimulus frequency (auto-detected from the matching spikes metadata.json,
or passed with --stim-freq), and writes figures/avcn_lfp_phase_cycle.png.

Usage:
  python LFP_reconstruction/plot_avcn_phase_cycle.py \
      --output-dir RESULTS/lfp_tmp/output_avcn_tone_1_2kHz_59dB_angle0_L
  # optional overrides: --stim-freq 1200 --n-cells 3600 --skip-ms 10
"""
import os
import re
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_reconstruct_avcn import _plot_phase_cycle, _plot_lfp, PROBE_Z, N_CH


def _detect_stim_freq(output_dir):
    """Read stim_freq_hz from the spikes metadata.json matching this output dir."""
    base       = os.path.basename(os.path.normpath(output_dir))
    parent     = os.path.dirname(os.path.normpath(output_dir))
    spikes_dir = base.replace('output_avcn_', 'spikes_', 1)
    meta_path  = os.path.join(parent, spikes_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get('stim_freq_hz')
    return None


def _parse_angle_side(output_dir):
    m = re.search(r'_angle(-?\d+)_([LR])', os.path.basename(os.path.normpath(output_dir)))
    return (int(m.group(1)), m.group(2)) if m else (0, 'L')


def main():
    ap = argparse.ArgumentParser(description='Regenerate AVCN phase-cycle LFP plot')
    ap.add_argument('--output-dir', required=True, dest='output_dir',
                    help='RESULTS/lfp_tmp/output_avcn_.../ directory of a finished run')
    ap.add_argument('--stim-freq', type=float, default=None, dest='stim_freq',
                    help='stimulus frequency in Hz (default: auto-detect from spikes metadata)')
    ap.add_argument('--n-cells', type=int, default=None, dest='n_cells',
                    help='cell count (title only)')
    ap.add_argument('--skip-ms', type=float, default=10.0, dest='skip_ms',
                    help='onset ramp to skip before phase-averaging (ms)')
    args = ap.parse_args()

    h5 = os.path.join(args.output_dir, 'PointSourcePotential_sum.h5')
    if not os.path.exists(h5):
        sys.exit(f'error: {h5} not found — is this a finished AVCN LFP output dir?')

    freq = args.stim_freq if args.stim_freq is not None else _detect_stim_freq(args.output_dir)
    if freq is None:
        sys.exit('error: stimulus frequency unknown; pass --stim-freq')
    angle, side = _parse_angle_side(args.output_dir)
    n_cells = args.n_cells if args.n_cells is not None else '?'

    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    # Regenerate both figures from the saved h5 (compound LFP has the fixed
    # t=0-transient scaling; phase cycle is the per-cycle average).
    _plot_lfp(args.output_dir, N_CH, PROBE_Z, side, angle, n_cells)
    _plot_phase_cycle(args.output_dir, freq, PROBE_Z, side, angle, n_cells,
                      skip_ms=args.skip_ms)


if __name__ == '__main__':
    main()
