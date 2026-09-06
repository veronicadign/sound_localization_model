#!/usr/bin/env python3
"""
Near-field LFP figures, shared by every nucleus.

One implementation of each of the three plots the LFP pipelines produce:

`plot_compound_lfp`   stacked probe traces + depth/time colour map
`plot_phase_cycle`    cycle-averaged response and its depth profile (tonal stimuli)
`plot_single_cells`   per-cell colour map + best-channel trace

These existed as three copies each (MSO, LSO, AVCN), and MNTB/SBC reached them by
calling a sibling's plotter and then RENAMING the PNGs it wrote.  Everything that
actually differed between the copies is now a `FigureStyle` field, so a new
nucleus supplies a style rather than another copy.
"""

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.signal

from recon_core import io_utils
from recon_core.signal_utils import time_axis


@dataclass(frozen=True)
class FigureStyle:
    """Everything that differs between the nuclei's LFP figures.

    name          human-readable title prefix, e.g. 'AVCN (GBC)'
    file_prefix   filename stem, e.g. 'avcn' -> figures/avcn_lfp_*.png
    trace_scale   'global'      one scale for all channels (MSO, AVCN): keeps the
                                relative channel amplitudes visible
                  'per_channel' each channel normalised to its own 99th percentile
                                (LSO): its axonal signal spans orders of magnitude
                                across channels, so a global scale flattens most
                                of the probe into a line
    trace_gain    stacked-trace height as a fraction of the channel spacing
    blank_onset_ms  zero the first N ms before scaling.  NEURON's finitialize
                    leaves a one-sample capacitive transient that otherwise sets
                    the colour scale and blanks the real signal; only the
                    morphologically detailed bushy cells are affected.
    index_label   optional callable(gid, n_total) -> str, annotating each
                  single-cell panel with the tonotopic index that cell stands for
    """
    name: str
    file_prefix: str
    trace_scale: str = 'global'
    trace_gain: float = 70.0
    blank_onset_ms: float = 0.0
    index_label: object = None


def _figure_path(output_dir, file_prefix, kind):
    return os.path.join(output_dir, 'figures', f'{file_prefix}_lfp_{kind}.png')


def _save(fig, path, what):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'{what} figure saved -> {path}')


def _load(output_dir, style):
    lfp, srate = io_utils.read_lfp_sum(output_dir)
    if style.blank_onset_ms:
        lfp[:, :max(1, int(round(style.blank_onset_ms * srate * 1e-3)))] = 0.0
    return lfp, srate


def _stack_offsets(lfp, probe_z, style):
    """Per-channel divisor turning µV into probe-z units for the stacked plot."""
    if style.trace_scale == 'per_channel':
        scale = np.percentile(np.abs(lfp), 99, axis=1, keepdims=True)
        spacing = (probe_z[-1] - probe_z[0]) / max(len(probe_z) - 1, 1)
        return np.maximum(scale, 1e-9) / (spacing * style.trace_gain)
    return max(np.abs(lfp).max() * 2, 1e-9) / style.trace_gain


# ---------------------------------------------------------------------------
def plot_compound_lfp(output_dir, probe_z, side, angle, n_cells, style):
    """Stacked probe traces beside the depth/time colour map."""
    lfp, srate = _load(output_dir, style)
    tvec = time_axis(lfp.shape[1], srate)
    divisor = _stack_offsets(lfp, probe_z, style)
    vmax = (float(np.percentile(np.abs(lfp), 99))
            if style.trace_scale == 'per_channel'
            else float(np.abs(lfp).max())) or 1e-9

    fig, (ax_traces, ax_map) = plt.subplots(1, 2, figsize=(12, 7),
                                            constrained_layout=True)
    for ch in range(len(probe_z)):
        offset = divisor[ch] if np.ndim(divisor) else divisor
        ax_traces.plot(tvec, lfp[ch] / offset + probe_z[ch], color='k', lw=0.6)
    ax_traces.set_xlabel('Time (ms)')
    ax_traces.set_ylabel('Probe z (µm)')
    ax_traces.set_title(f'{style.name} compound LFP — stacked traces')

    im = ax_map.imshow(lfp, aspect='auto', origin='lower',
                       extent=[tvec[0], tvec[-1], probe_z[0], probe_z[-1]],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax_map, label='LFP (µV)')
    ax_map.set_xlabel('Time (ms)')
    ax_map.set_ylabel('Probe z (µm)')
    ax_map.set_title(f'{style.name} | side {side} | angle {angle}° | N={n_cells}')

    _save(fig, _figure_path(output_dir, style.file_prefix, 'reconstruction'),
          'Compound LFP')


# ---------------------------------------------------------------------------
def plot_phase_cycle(output_dir, stimulus_freq, probe_z, side, angle, n_cells,
                     style, skip_ms=10.0):
    """Cycle-averaged LFP and its depth profile — the neurophonic view.

    Only meaningful for a periodic stimulus: the response is folded onto one
    stimulus cycle, which averages away everything not phase-locked to it.  The
    first `skip_ms` are dropped so the onset ramp does not contaminate the
    average.  Silently skipped for clicks and noise (no stimulus frequency).
    """
    if stimulus_freq is None:
        print('Stimulus frequency unknown; skipping the phase-cycle figure.')
        return

    lfp, srate = _load(output_dir, style)
    dt_ms = 1e3 / srate
    steady = scipy.signal.detrend(lfp[:, int(skip_ms / dt_ms):], axis=1)

    samples_per_cycle = max(1, int(round(1e3 / stimulus_freq / dt_ms)))
    n_cycles = steady.shape[1] // samples_per_cycle
    if n_cycles < 1:
        print('Less than one full cycle after the ramp; skipping the phase-cycle figure.')
        return

    folded = (steady[:, :n_cycles * samples_per_cycle]
              .reshape(steady.shape[0], n_cycles, samples_per_cycle)
              .mean(axis=1))                       # (n_ch, samples_per_cycle)
    centred = folded - folded.mean(axis=1, keepdims=True)

    phase = np.linspace(0, 1, samples_per_cycle, endpoint=False)
    n_ch = folded.shape[0]
    spacing = (probe_z[-1] - probe_z[0]) / max(n_ch - 1, 1)
    half = samples_per_cycle // 2
    subtitle = f'side {side} | {angle}° | N={n_cells} | {stimulus_freq:.0f} Hz'

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
    for ax, data, title in ((axes[0], folded, f'{style.name} raw | {subtitle}'),
                            (axes[1], centred, f'{style.name} mean-removed | {subtitle}')):
        divisor = (np.abs(data).max() or 1e-9) / spacing
        for ch in range(n_ch):
            trace = data[ch] / divisor + probe_z[ch]
            ax.plot(phase, trace, color='gray', lw=0.9)
            # mark the half-cycle and end-of-cycle points, so a phase shift
            # across depth is visible as a tilt in the marker column
            ax.plot(phase[half], trace[half], 'o', color='steelblue', ms=4, zorder=3)
            ax.plot(phase[-1], trace[-1], 'o', color='firebrick', ms=4, zorder=3)
        ax.set_xlabel('Cycle phase')
        ax.set_ylabel('Probe z (µm)')
        ax.set_title(title)
        ax.set_xlim(0, 1)

    ax_depth = axes[2]
    for t in range(samples_per_cycle):
        colour, lw, z = ('steelblue', 1.8, 3) if t == half else \
                        ('firebrick', 1.8, 3) if t == samples_per_cycle - 1 else \
                        ('lightgray', 0.6, 1)
        ax_depth.plot(probe_z, centred[:, t], color=colour, lw=lw, zorder=z)
    ax_depth.axhline(0, color='k', lw=0.5, ls='--', zorder=2)
    ax_depth.set_xlabel('Probe z (µm)')
    ax_depth.set_ylabel('LFP (µV)')
    ax_depth.set_title('Depth profile')

    _save(fig, _figure_path(output_dir, style.file_prefix, 'phase_cycle'),
          'Phase-cycle')


# ---------------------------------------------------------------------------
def plot_single_cells(output_dir, single_contribs, tvec, probe_z, soma_pos,
                      probe_x, probe_y, cell_gids, total_sim_cells, style):
    """One row per sampled cell: full colour map + its strongest channel.

    single_contribs : (n_cells, n_ch, n_t) µV
    soma_pos        : (n_cells, 3) µm
    The soma→probe distance is annotated because it dominates the amplitude —
    it is the check that near cells, not a numerical artefact, drive the LFP.
    """
    n_cells = single_contribs.shape[0]

    def min_distance(sx, sy, sz):
        return float(np.min(np.sqrt((sx - probe_x) ** 2 + (sy - probe_y) ** 2
                                    + (sz - probe_z) ** 2)))

    fig = plt.figure(figsize=(14, 2.8 * n_cells))
    grid = gridspec.GridSpec(n_cells, 2, figure=fig, left=0.07, right=0.97,
                             hspace=0.5, wspace=0.35)

    for i in range(n_cells):
        gid = cell_gids[i]
        sx, sy, sz = soma_pos[i]
        d_min = min_distance(sx, sy, sz)
        tag = ''
        if style.index_label is not None:
            tag = f' ({style.index_label(gid, total_sim_cells)})'

        ax_map = fig.add_subplot(grid[i, 0])
        vmax = np.abs(single_contribs[i]).max() or 1e-9
        ax_map.imshow(single_contribs[i], aspect='auto', origin='lower',
                      extent=[tvec[0], tvec[-1], probe_z[0], probe_z[-1]],
                      cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax_map.set_ylabel('z (µm)')
        ax_map.set_title(f'Sim ID {gid}{tag} | soma ({sx:.0f}, {sy:.0f}, {sz:.0f}) µm'
                         f' | d_min = {d_min:.0f} µm')

        best_ch = int(np.argmax(np.abs(single_contribs[i]).max(axis=1)))
        ax_trace = fig.add_subplot(grid[i, 1])
        ax_trace.plot(tvec, single_contribs[i, best_ch], color='steelblue', lw=0.8)
        ax_trace.set_ylabel('LFP (µV)')
        ax_trace.set_title(f'Sim ID {gid} ch {best_ch} '
                           f'(z = {probe_z[best_ch]:.0f} µm) | d_min = {d_min:.0f} µm')
        if i == n_cells - 1:
            ax_map.set_xlabel('Time (ms)')
            ax_trace.set_xlabel('Time (ms)')

    fig.suptitle(f'{style.name} single-cell LFP contributions', y=1.01)
    path = _figure_path(output_dir, style.file_prefix, 'single_cells')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Single-cell figure saved -> {path}')


def plot_all(output_dir, probe_z, probe_x, probe_y, side, angle, n_cells, style,
             stimulus_freq=None, single_contribs=None, soma_pos=None,
             cell_gids=None, dt_ms=None):
    """Every figure a finished LFP run produces, in one call."""
    plot_compound_lfp(output_dir, probe_z, side, angle, n_cells, style)
    plot_phase_cycle(output_dir, stimulus_freq, probe_z, side, angle, n_cells, style)
    if single_contribs is not None and len(single_contribs):
        tvec = np.arange(single_contribs.shape[2]) * dt_ms
        plot_single_cells(output_dir, single_contribs, tvec, probe_z, soma_pos,
                          probe_x, probe_y, cell_gids, n_cells, style)
