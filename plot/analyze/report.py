import logging
from bisect import bisect_left
from collections import defaultdict
from contextlib import ExitStack
import math
from math import ceil
from pathlib import PurePath
from typing import Iterable, List

import brian2 as b2
from brian2 import Hz
import brian2hears as b2h
from brian2hears import erbspace
import dill
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sorcery import dict_of
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
    

# from analyze import sound_analysis as SA
# from cochleas.hrtf_utils import run_hrtf

from cochleas.consts import CFMAX, CFMIN
from utils.custom_sounds import Tone, ToneBurst

plt.rcParams["axes.grid"] = True
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight']= 'bold'
plt.rcParams['axes.spines.top']= False
plt.rcParams['axes.labelsize'] = 14 
plt.rcParams['xtick.labelsize'] = 12   # Size of x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12   # Size of y-axis tick labels
plt.rcParams['legend.fontsize'] = 14   # Size of the legend text
# Make axis labels bold
plt.rcParams['axes.labelweight'] = 'bold'  # Makes x and y axis labels bold

def create_xax_time_sound(res):
    x_times = np.linspace(0, res['simulation_time'], int((res['basesound'].sound.samplerate / b2.kHz)*res['simulation_time']))
    return x_times

def flatten(items):
    """Yield items from any nested iterable.
    from https://stackoverflow.com/a/40857703
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return (myList[0], 0)
    if pos == len(myList):
        return (myList[-1], len(myList))
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return (after, pos)
    else:
        return (before, pos - 1)

def avg_fire_rate_actv_neurons(x):
    active_neurons = set(x["senders"])
    return (len(x["times"]) / len(active_neurons)) if len(active_neurons) > 0 else 0

def firing_neurons_distribution(x):
    "returns {neuron_id: num_spikes}.keys()"
    n2s = {id: 0 for id in x["global_ids"]}
    for sender in x["senders"]:
        n2s[sender] += 1
    return n2s.values()

def shift_senders(x, hist_logscale=False):
    "returns list of 'senders' with ids shifted to [0,num_neurons]. optionally ids are CFs"
    if hist_logscale:
        cf = b2h.erbspace(CFMIN, CFMAX, len(x["global_ids"])) / b2.Hz
        old2newid = {oldid: cf[i] for i, oldid in enumerate(x["global_ids"])}
    else:
        old2newid = {oldid: i for i, oldid in enumerate(x["global_ids"])}
    return [old2newid[i] for i in x["senders"]]

def draw_hist(
    ax,
    senders_renamed,
    angles,
    num_neurons,
    max_spikes_single_neuron,
    logscale=True,
    freq=None,
):
    """draws a low opacity horizontal histogram for each angle position

    includes a secondary y-axis, optionally logarithmic.
    if logscale, expects senders to be renamed to CFs
    if freq, include a horizontal line at corresponding frequency
    """
    max_histogram_height = 0.25
    bin_count = 50
    alpha = 0.5
    freqlinestyle = {
        "color": "black",
        "linestyle": ":",
        "label": "freq_in",
        "alpha": 0.2,
    }
    if logscale:
        bins = b2h.erbspace(CFMIN, CFMAX, bin_count) / b2.Hz

        for j, angle in enumerate(angles):
            left_data = senders_renamed["L"][j]
            right_data = senders_renamed["R"][j]

            left_hist, _ = np.histogram(left_data, bins=bins)
            right_hist, _ = np.histogram(right_data, bins=bins)
            max_value = max(max(left_hist), max(right_hist))
            left_hist_normalized = left_hist / (max_value * max_histogram_height)
            right_hist_normalized = right_hist / (max_value * max_histogram_height)

            ax.barh(
                bins[:-1],
                -left_hist_normalized,
                height=np.diff(bins),  # bins have different sizes
                left=angle,
                color="m",
                alpha=alpha,
                align="edge",
            )
            ax.barh(
                bins[:-1],
                right_hist_normalized,
                height=np.diff(bins),
                left=angle,
                color="g",
                alpha=alpha,
                align="edge",
            )
        ax.set_yscale("log")
        ax.set_ylim(CFMIN, CFMAX)
        yticks = [125, 1000, 10000, 20000]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{freq} Hz" for freq in yticks])
        if freq is not None:
            ax.axhline(y=freq / b2.Hz, **freqlinestyle)
        ax.set_ylabel("CF [Hz]")
    else:
        bins = np.linspace(0, num_neurons, bin_count)

        for j, angle in enumerate(angles):
            left_data = senders_renamed["L"][j]
            right_data = senders_renamed["R"][j]

            left_hist, _ = np.histogram(left_data, bins=bins)
            right_hist, _ = np.histogram(right_data, bins=bins)
            left_hist_normalized = (
                left_hist / max_spikes_single_neuron * max_histogram_height
            )
            right_hist_normalized = (
                right_hist / max_spikes_single_neuron * max_histogram_height
            )

            ax.barh(
                bins[:-1],
                -left_hist_normalized,
                height=num_neurons / bin_count,
                left=angle,
                color="C0",
                alpha=alpha,
                align="edge",
            )
            ax.barh(
                bins[:-1],
                right_hist_normalized,
                height=num_neurons / bin_count,
                left=angle,
                color="C1",
                alpha=alpha,
                align="edge",
            )
        ax.set_ylabel("neuron id")
        ax.set_ylim(0, num_neurons)

        if freq is not None:
            cf = b2h.erbspace(CFMIN, CFMAX, num_neurons)
            freq, neur_n = take_closest(cf, freq)
            ax.axhline(y=neur_n)
    ax.yaxis.set_minor_locator(plt.NullLocator())  # remove minor ticks

def draw_single_angle_histogram(data, angle, population="SBC", fontsize=16, alpha=0.8):
    """
    Draw horizontal histograms of spike distributions across frequencies for a single angle,
    with left population growing downward and right population growing upward from a central axis.

    Parameters:
    -----------
    data : dict
        The full dataset containing angle_to_rate information
    angle : float
        The specific angle to visualize
    population : str
        Name of the neural population to visualize
    fontsize : int
        Base fontsize for the plot. Other elements will scale relative to this.

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Constants
    bin_count = 50

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 2.42))

    # Get data for this angle and population
    pop_data = {
        "L": data["angle_to_rate"][angle]["L"][population],
        "R": data["angle_to_rate"][angle]["R"][population],
    }

    # Create logarithmic bins for frequency
    bins = b2h.erbspace(CFMIN, CFMAX, bin_count) / b2.Hz

    # Process data for histograms
    senders_renamed = {
        side: shift_senders(pop_data[side], True)  # True for logscale
        for side in ["L", "R"]
    }

    # Create histograms
    left_hist, _ = np.histogram(senders_renamed["L"], bins=bins)
    right_hist, _ = np.histogram(senders_renamed["R"], bins=bins)

    # Normalize histograms
    max_value = max(max(left_hist), max(right_hist))
    if max_value > 0:  # Avoid division by zero
        left_hist = left_hist / max_value
        right_hist = right_hist / max_value

    # Plot histograms - note the negative values for left histogram
    ax.bar(
        bins[:-1],
        -left_hist,
        width=np.diff(bins),
        color="C0",
        alpha=alpha,
        label="Left",
        align="edge",
    )
    ax.bar(
        bins[:-1],
        right_hist,
        width=np.diff(bins),
        color="C1",
        alpha=alpha,
        label="Right",
        align="edge",
    )

    # Configure axes
    ax.set_xscale("log")
    ax.set_xlim(CFMIN, CFMAX)
    xticks = [20, 100, 500, 1000, 5000, 20000]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{freq} Hz" for freq in xticks], fontsize=fontsize)

    # Set y-axis limits symmetrically around zero
    ylim = 1.1  # Slightly larger than 1 to give some padding
    ax.set_ylim(-ylim, ylim)
    ax.tick_params(axis="y", labelsize=fontsize)  # Set y-tick font size

    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Set font sizes for labels and title
    ax.set_xlabel("Characteristic Frequency (Hz)", fontsize=fontsize)
    ax.set_ylabel("Normalized spikes", fontsize=fontsize)
    # ax.legend(fontsize=fontsize)

    # plt.title(
    #     f"{population} population response at {angle}° azimuth\n"
    #     f'Sound: {data["conf"]["sound_key"]}',
    #     fontsize=fontsize * 1.2,
    # )  # Title slightly larger

    plt.tight_layout()

    return fig

def synthetic_angle_to_itd(angle, w_head: int = 22, v_sound: int = 33000):
    delta_x = w_head * np.sin(np.deg2rad(angle))
    return round(1000 * delta_x / v_sound, 2)

def get_spike_phases(spike_times: np.ndarray, frequency: float) -> np.ndarray:
    times_sec = spike_times
    return 2 * np.pi * frequency * (times_sec % (1 / frequency))

def calculate_vector_strength(spike_times: np.ndarray, frequency: float) -> float:
    if len(spike_times) == 0:
        return 0
    phases = get_spike_phases(spike_times, frequency)
    x = np.mean(np.cos(phases))
    y = np.mean(np.sin(phases))
    return np.sqrt(x**2 + y**2)
    
def range_around_center(center, radius, min_val=0, max_val=np.iinfo(np.int64).max):
    start = max(min_val, center - radius)
    end = min(max_val + 1, center + radius + 1)
    return np.arange(start, end)

def calculate_vector_strength_from_result(
        # result file (loaded)
        res,
        angle,
        side,
        pop,
        freq = None, # if None: freq = res['basesound'].frequency
        color = None,
        cf_target = None,
        bandwidth=0,
        n_bins = 7,
        figsize = (7,5),
        display=False, # if True also return fig, show() in caller function
        x_ax = "phase",  # can be "phase" or "time"
        ylim = None,    # Added ylim parameter
        center_at_peak = False  # Center histogram so that the peak bin is at zero
        ):
    
    spikes = res["angle_to_rate"][angle][side][pop]
    sender2times = defaultdict(list)
    for sender, time in zip(spikes["senders"], spikes["times"]):
        if time <= 1000:
            sender2times[sender].append(time)
    sender2times = {k: np.array(v) / 1000 for k, v in sender2times.items()}
    num_neurons = len(spikes["global_ids"])
    cf = erbspace(CFMIN, CFMAX, num_neurons)

    if(freq == None):
        if(type(res['basesound'])  in (Tone,ToneBurst)):
            freq = res['basesound'].frequency
        else:
            print("Frequency needs to be specified for non-Tone sounds")
    else:
        freq = freq * Hz

    if(cf_target == None):    
        _, center_neuron_for_freq = take_closest(cf, freq)
    else:
        _, center_neuron_for_freq = take_closest(cf, cf_target *Hz)

    old2newid = {oldid: i for i, oldid in enumerate(spikes["global_ids"])}
    new2oldid = {v: k for k, v in old2newid.items()}

    relevant_neurons = range_around_center(
        center_neuron_for_freq, radius=bandwidth, max_val=num_neurons - 1
    )
    relevant_neurons_ids = [new2oldid[i] for i in relevant_neurons]

    spike_times_list = [sender2times[i] for i in relevant_neurons_ids]  
    spike_times_array = np.concatenate(spike_times_list)  # Flatten into a single array

    phases = get_spike_phases(
        spike_times= spike_times_array, frequency=freq / Hz
    )
    vs = calculate_vector_strength(
        spike_times=spike_times_array, frequency=freq / Hz
    )

    if not display:
        return (vs)
    if color == None:
        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'
        else: color = 'k'
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get total number of spikes for percentage calculation
    total_spikes = len(spike_times_array)
    
    if x_ax == "phase":
        # Initial binning to find the peak
        orig_bins = np.linspace(0, 2 * np.pi, n_bins + 1)
        hist_values, _ = np.histogram(phases, bins=orig_bins)
        peak_bin_idx = np.argmax(hist_values)
        
        if center_at_peak:
            # Calculate the center of the peak bin
            bin_centers = (orig_bins[:-1] + orig_bins[1:]) / 2
            peak_center = bin_centers[peak_bin_idx]
            
            # Create a shift that will center the peak at 0
            shift = peak_center - np.pi  # Shift to make peak at π, then will offset by π
            
            # Create bins centered around the peak
            bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            shifted_phases = np.mod(phases - shift, 2 * np.pi) - np.pi
            
            # Plot the shifted histogram as percentages
            hist1, _ = np.histogram(shifted_phases, bins=bins)
            hist_percent = (hist1 / total_spikes) * 100  # Convert to percentage
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(bin_centers, hist_percent, width=2 * np.pi / n_bins, alpha=0.7, color=color)
            
            # Set x-ticks in terms of π
            pi_ticks = np.array([-1, -0.5, 0, 0.5, 1]) * np.pi
            pi_labels = ['-0.5', '', '0', '', '0.5']
        else:
            # Original histogram from 0 to 2π
            bins = np.linspace(0, 2 * np.pi, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            hist1, _ = np.histogram(phases, bins=bins)
            hist_percent = (hist1 / total_spikes) * 100  # Convert to percentage
            
            ax.bar(bin_centers, hist_percent, width=2 * np.pi / n_bins, alpha=0.7, color=color)
            
            # Set x-ticks in terms of π
            pi_ticks = np.array([0, 0.5, 1, 1.5, 2]) * np.pi
            pi_labels = ['0', '', '0.5', '', '1']
        
        ax.set_xticks(pi_ticks)
        ax.set_xticklabels(pi_labels)
        ax.set_xlabel("Phase (cycles)")
        
    elif x_ax == "time":
        # Convert phases to time in milliseconds
        period_ms = 1000 / (freq / Hz)  # Period in milliseconds
        time_values = (phases / (2 * np.pi)) * period_ms
        
        # Initial binning to find the peak
        orig_time_bins = np.linspace(0, period_ms, n_bins + 1)
        hist_values, _ = np.histogram(time_values, bins=orig_time_bins)
        peak_bin_idx = np.argmax(hist_values)
        
        if center_at_peak:
            # Calculate the center of the peak bin
            bin_centers = (orig_time_bins[:-1] + orig_time_bins[1:]) / 2
            peak_center = bin_centers[peak_bin_idx]
            
            # Create a shift that will center the peak at 0
            shift = peak_center - period_ms/2  # Shift to make peak at period/2, then will offset
            
            # Create bins centered around the peak
            time_bins = np.linspace(-period_ms/2, period_ms/2, n_bins + 1)
            shifted_time_values = np.mod(time_values - shift, period_ms) - period_ms/2
            
            # Plot the shifted histogram as percentages
            hist1, _ = np.histogram(shifted_time_values, bins=time_bins)
            hist_percent = (hist1 / total_spikes) * 100  # Convert to percentage
            
            time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
            ax.bar(time_bin_centers, hist_percent, width=period_ms / n_bins, alpha=0.7, color=color)
        else:
            # Original time from 0 to period
            time_bins = np.linspace(0, period_ms, n_bins + 1)
            time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
            
            hist1, _ = np.histogram(time_values, bins=time_bins)
            hist_percent = (hist1 / total_spikes) * 100  # Convert to percentage
            
            ax.bar(time_bin_centers, hist_percent, width=period_ms / n_bins, alpha=0.7, color=color)
        
        ax.set_xlabel("Time [ms]")
    
    # Set y-axis label to percentage
    ax.set_ylabel("Spikes/bin (% of total)")
    
    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_title(f"R={vs:.3f}")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    return (vs, fig)

def calculate_vector_strength_from_result_polar(
        res,
        angle,
        side,
        pop,
        freq=None,  # if None: freq = res['basesound'].frequency
        cf_target=None,
        bandwidth=0,
        n_bins=7,
        display=False,
        color = None,
        figsize = [7,5]  # if True also return fig, show() in caller function
        ):
    
    # Get spikes and organize times per sender
    spikes = res["angle_to_rate"][angle][side][pop] 
    print(spikes)
    sender2times = defaultdict(list)
    for sender, time in zip(spikes["senders"], spikes["times"]):
        sender2times[sender].append(time)
    sender2times = {k: np.array(v) / 1000 for k, v in sender2times.items()}
    num_neurons = len(spikes["global_ids"])
    cf = erbspace(CFMIN, CFMAX, num_neurons)
    
    # Determine the frequency to use
    if freq is None:
        if type(res['basesound']) in (Tone, ToneBurst):
            freq = res['basesound'].frequency
        else:
            print("Frequency needs to be specified for non-Tone sounds")
    else:
        freq = freq * Hz
    
    # Determine the closest characteristic frequency (CF) neuron
    if cf_target is None:    
        cf_neuron, center_neuron_for_freq = take_closest(cf, freq)
    else:
        cf_neuron, center_neuron_for_freq = take_closest(cf, cf_target * Hz)
    
    # Map between old and new neuron IDs
    old2newid = {oldid: i for i, oldid in enumerate(spikes["global_ids"])}
    new2oldid = {v: k for k, v in old2newid.items()}
    
    # Choose relevant neurons based on the center neuron and bandwidth
    relevant_neurons = range_around_center(
        center_neuron_for_freq, radius=bandwidth, max_val=num_neurons - 1
    )
    relevant_neurons_ids = [new2oldid[i] for i in relevant_neurons]
    
    # Concatenate the spike times from the relevant neurons
    spike_times_list = [sender2times[i] for i in relevant_neurons_ids]
    spike_times_array = np.concatenate(spike_times_list)  # Flatten into a single array
    
    # Compute phases and vector strength
    phases = get_spike_phases(spike_times=spike_times_array, frequency=freq / Hz)
    vs = calculate_vector_strength(spike_times=spike_times_array, frequency=freq / Hz)
    
    if not display:
        return (vs, None)

    if color == None:
        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'
        else: color = 'k'

    
    # Plot phases in polar coordinates
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Create a polar subplot
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=figsize)
    hist1, _ = np.histogram(phases, bins=bins, density = True)
    ax.bar(bin_centers, hist1, width=2 * np.pi / n_bins, alpha=0.7, color = color)
    
    if bandwidth == 0:
        ax.set_title(f"Neuron {relevant_neurons_ids[0]} (CF: {cf_neuron:.1f} Hz)\nVS={vs:.3f}")
    else:
        ax.set_title(f"Neurons {relevant_neurons_ids[0]} : {relevant_neurons_ids[-1]} (center CF: {cf_neuron:.1f} Hz)\nVS={vs:.3f}")
    
    # Remove all but the last yticks
    ax.set_yticks([])  # Keep only the last tick
    #ax.yaxis.set_tick_params(labelsize=10)  # Adjust size if needed

    plt.show()
    return

def draw_spikes_pop(
    res,
    angle,
    side,
    pop,
    y_ax = 'cf_custom',
    f_ticks = [125,1000,10000],
    title=None,
    plot_sound = False,
    xlim=None,
    ylim=None,
    color = None,
    figsize = (7,4)
):
    spikes = res["angle_to_rate"][angle][side][pop]  
    num_neurons = len(spikes["global_ids"])
    cf = erbspace(CFMIN, CFMAX, num_neurons)
    neuron_to_cf = {global_id: freq for global_id, freq in zip(spikes["global_ids"], cf)}
    duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

    if color == None:
        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'

    if xlim == None: 
        xlim_array = [0,duration]
    else: xlim_array = xlim

    if y_ax == 'global_ids':
        y_values = spikes['senders']
        ylabel = f"{pop} global IDS"
    elif y_ax == 'cf':
        y_values = np.array([neuron_to_cf[sender] for sender in spikes["senders"]])
        ylabel = f"{pop} CF [Hz]"
    elif y_ax == 'ids':
        y_values = spikes['senders'] - spikes['global_ids'][0]
        ylabel = f"{pop} IDS"
    elif y_ax == 'cf_custom':
        y_values = spikes['senders'] - spikes['global_ids'][0]
        ylabel = f"{pop} CF [Hz]"
        label_indexes = np.zeros_like(f_ticks)
        for i, f in enumerate(f_ticks):
            _, label_indexes[i] = take_closest(cf, f*Hz)
    else:
        raise ValueError("Invalid value for 'ax'. Choose 'cf_custom', 'ids', 'cf', or 'global_ids'.")

    if(plot_sound):
        # Create figure with gridspec to control subplot sizes
        fig = plt.figure(figsize=(figsize[0], figsize[1]+1))
        # Create smaller subplot for sound (25% of height) and keep full width for spikes plot
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 7])
        
        # Sound plot - smaller and without axes/grid
        ax0 = fig.add_subplot(gs[0])
        x_sound = create_xax_time_sound(res)
        if len(x_sound) != len(res['basesound'].sound):
            x_sound = x_sound[0:len(res['basesound'].sound)]
        ax0.plot(x_sound, res['basesound'].sound, 'k', linewidth = '1')
        ax0.set_xlim(xlim_array)
        # Remove axes and grid
        ax0.axis('off')
        
        # Spikes plot - same size as when plot_sound is False
        ax1 = fig.add_subplot(gs[1])
        if ylim != None:
            if y_ax == 'cf':
                ax1.set_ylim(ylim)
            else:
                _, y0 = take_closest(cf, ylim[0]*Hz)
                _, y1 = take_closest(cf, ylim[1]*Hz)
                ax1.set_ylim([y0,y1])
        else:
            ax1.set_yticks(label_indexes)
            ax1.set_yticklabels(f_ticks*Hz) 
        
        ax1.plot(spikes['times'], y_values, '.', color=color, markersize=1)
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel("Time [ms]")
        ax1.set_xlim(xlim_array)
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(1, figsize=figsize)
        if ylim != None:
            if y_ax == 'cf':
                ax.set_ylim(ylim)
            else:
                _, y0 = take_closest(cf, ylim[0]*Hz)
                _, y1 = take_closest(cf, ylim[1]*Hz)
                ax.set_ylim([y0,y1])
        else:
            ax.set_yticks(label_indexes)
            ax.set_yticklabels(f_ticks*Hz) 
        
        ax.plot(spikes['times'], y_values, '.', color=color, markersize=1)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [ms]")
        ax.set_xlim(xlim_array)
    
    plt.show()
    return

def draw_spikes_pop_bothside(
    res,
    angle,
    pop,
    y_ax = 'cf_custom',
    f_ticks = [100,1000,10000],
    title=None,
    plot_sound = False,
    xlim=None,
    ylim=None,
    color = None,
    figsize = (7,8)
):
    fig, ax = plt.subplots(2, figsize = figsize, sharex = True)
    for i, side in enumerate(['L', 'R']):
        spikes = res["angle_to_rate"][angle][side][pop]  
        num_neurons = len(spikes["global_ids"])
        cf = erbspace(CFMIN, CFMAX, num_neurons)
        neuron_to_cf = {global_id: freq for global_id, freq in zip(spikes["global_ids"], cf)}
        duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

        if xlim == None: 
            xlim_array = [0,duration]
        else: xlim_array = xlim
            
        if ylim is None:
            ylim = [CFMIN/Hz, CFMAX/Hz]

        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'

        if y_ax == 'global_ids':
            y_values = spikes['senders']
            ylabel = f"{pop} global IDS"
        elif y_ax == 'cf':
            y_values = np.array([neuron_to_cf[sender] for sender in spikes["senders"]])
            ylabel = f"{pop} CF [Hz]"
        elif y_ax == 'ids':
            y_values = spikes['senders'] - spikes['global_ids'][0]
            ylabel = f"{pop} IDS"
        elif y_ax == 'cf_custom':
            y_values = spikes['senders'] - spikes['global_ids'][0]
            ylabel = f"{pop} CF [Hz]"
            label_indexes = np.zeros_like(f_ticks)
            for i, f in enumerate(f_ticks):
                _, label_indexes[i] = take_closest(cf, f*Hz)
        else:
            raise ValueError("Invalid value for 'ax'. Choose 'cf_custom', 'ids', 'cf', or 'global_ids'.")
  
        if ylim != None:
            if y_ax == 'cf':
                ax[i].set_ylim(ylim)
            else:
                _, y0 = take_closest(cf, ylim[0]*Hz)
                _, y1 = take_closest(cf, ylim[1]*Hz)
                ax[i].set_ylim([y0,y1])
        else:
            ax[i].set_yticks(label_indexes)
            ax[i].set_yticklabels(f_ticks*Hz) 
        
        ax[i].plot(spikes['times'], y_values, '.', color=color, markersize=1)
        ax[i].set_ylabel(ylabel)
        ax[i].set_xlim(xlim_array)
    ax[1].set_xlabel("Time [ms]")
    plt.show()
    return

def draw_psth_pop_bothside(
    res,
    angle,
    pop,
    title=None,
    xlim=None,
    ylim=None,
    bin_size = 1, #ms
    color = None,
    figsize = (7,4)
):
    fig, ax = plt.subplots(1, figsize = figsize)
    for color, side in zip(['m', 'g'],['L', 'R']):
        spikes = res["angle_to_rate"][angle][side][pop] 
        spike_times = spikes['times']
        spike_senders = spikes['senders']
        num_neurons = len(spikes["global_ids"])
        cf = erbspace(CFMIN, CFMAX, num_neurons)
        duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

        if xlim == None: 
            xlim_array = [0,duration]
        else: xlim_array = xlim
        
        if ylim is None:
            ylim = [CFMIN/Hz, CFMAX/Hz]

        time_mask = (spike_times >= xlim_array[0]) & (spike_times <= xlim_array[1])
        filtered_times = spike_times[time_mask]
        filtered_senders = spike_senders[time_mask]
        # Fix for the error - get indices properly
        _, ymin_idx = take_closest(cf, ylim[0]*Hz)
        _, ymax_idx = take_closest(cf, ylim[1]*Hz)
    
        # Convert indices to actual IDs
        ymin = spikes["global_ids"][0] + ymin_idx
        ymax = spikes["global_ids"][0] + ymax_idx
        cluster_mask = (filtered_senders >= ymin) & (filtered_senders <= ymax)
        cluster_times = filtered_times[cluster_mask]
        bins = np.arange(xlim_array[0], xlim_array[1] + bin_size, bin_size)
        ax.hist(cluster_times, bins=bins, alpha=0.7, color = color)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Spike Count')
        ax.grid(True, alpha=0.3)

    # You can add more customization to the axes if needed
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    return

def draw_psth_pop(
    res,
    angle,
    side,
    pop,
    title=None,
    xlim=None,
    ylim=None,
    bin_size = 1, #ms
    color = None,
    figsize = (7,4)
):
    spikes = res["angle_to_rate"][angle][side][pop]  
    spike_times = spikes['times']
    spike_senders = spikes['senders']
    num_neurons = len(spikes["global_ids"])
    cf = erbspace(CFMIN, CFMAX, num_neurons)
    duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

    if color == None:
        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'

    if xlim == None: 
        xlim_array = [0,duration]
    else: xlim_array = xlim

    time_mask = (spike_times >= xlim[0]) & (spike_times <= xlim[1])
    filtered_times = spike_times[time_mask]
    filtered_senders = spike_senders[time_mask]
    # Fix for the error - get indices properly
    fmin, ymin_idx = take_closest(cf, ylim[0]*Hz)
    fmax, ymax_idx = take_closest(cf, ylim[1]*Hz)
    
    # Convert indices to actual IDs
    ymin = spikes["global_ids"][0] + ymin_idx
    ymax = spikes["global_ids"][0] + ymax_idx
    cluster_mask = (filtered_senders >= ymin) & (filtered_senders <= ymax)
    cluster_times = filtered_times[cluster_mask]
    bins = np.arange(xlim[0], xlim[1] + bin_size, bin_size)
    fig, ax = plt.subplots(1, figsize = figsize)
    ax.hist(cluster_times, bins=bins, alpha=0.7, color = color)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Spike Count')
    ax.grid(True, alpha=0.3)

    # You can add more customization to the axes if needed
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    return

def calculate_firing_rates(angle_to_rate, pop, sides, angles, duration, cf_interval=None):
    """
    Calculate firing rates for different sides and angles.
    
    Parameters:
    - angle_to_rate: Dictionary mapping angles to rate data
    - pop: Population name (e.g., 'LSO')
    - sides: List of sides ('L', 'R', or both)
    - angles: List of angles to process
    - duration: Duration of the simulation
    - cf_interval: Optional frequency interval for filtering
    
    Returns:
    - tot_spikes: Dictionary of total spike rates by side
    - active_neuron_rate: Dictionary of active neuron rates by side
    """
    
    # Get frequency space
    num_neurons = len(angle_to_rate[0]['L'][pop]["global_ids"])
    cf = erbspace(CFMIN, CFMAX, num_neurons)
    
    if cf_interval is None:
        # Simple case - use all neurons
        tot_spikes = {
            side: [
                len(angle_to_rate[angle][side][pop]["times"]) / duration
                for angle in angles
            ]
            for side in sides
        }
        avg_neuron_rate = {
            side: [
                len(angle_to_rate[angle][side][pop]["times"]) / (duration*num_neurons)
                for angle in angles
            ]
            for side in sides
        }

        active_neuron_rate = {
            side: [
                avg_fire_rate_actv_neurons(angle_to_rate[angle][side][pop])
                * (1 * b2.second)
                / duration
                for angle in angles
            ]
            for side in sides
        }
        return tot_spikes, avg_neuron_rate, active_neuron_rate
    
    # Case with CF interval filtering
    tot_spikes = {}
    avg_neuron_rate = {}
    cluster_numerosity = {}
    
    
    for side in sides:
        tot_spikes[side] = []
        avg_neuron_rate[side] = []
        
        for angle in angles:
            # Find indices in CF array corresponding to interval bounds
            _, ymin_idx = take_closest(cf, cf_interval[0]*Hz)
            _, ymax_idx = take_closest(cf, cf_interval[1]*Hz)

            # Calculate actual neuron IDs from global_ids and indices
            base_id = angle_to_rate[angle][side][pop]["global_ids"][0]
            ymin = base_id + ymin_idx
            ymax = base_id + ymax_idx

            # Filter spikes within the specified range
            cluster_mask = (angle_to_rate[angle][side][pop]['senders'] >= ymin) & (angle_to_rate[angle][side][pop]['senders'] <= ymax)
            cluster_times = angle_to_rate[angle][side][pop]['times'][cluster_mask]

            # Calculate rate for this angle and side
            tot_spikes[side].append(len(cluster_times) / duration)
            avg_neuron_rate[side].append(len(cluster_times)/((ymax - ymin)*duration))

            # Compute cluster numerosity (unique senders)
            unique_senders = np.unique(angle_to_rate[angle][side][pop]['senders'][cluster_mask])
        
        cluster_numerosity[side] = len(unique_senders)
    
    return tot_spikes, avg_neuron_rate, avg_neuron_rate #unique set not computed for a cf_interval

def normalize_rates(plotted_rate, sides):
    """
    Normalize firing rates using min-max normalization.
    
    Parameters:
    - plotted_rate: Dictionary of firing rates by side
    - sides: List of sides to process
    
    Returns:
    - normalized_rate: Dictionary of normalized firing rates by side
    - original_values: Dictionary containing original min/max values and their indices
    """
    normalized_rate = {side: [] for side in sides}
    original_values = {}
    
    for side in sides:
        # Find the minimum and maximum values across all angles for this side
        min_value = min(plotted_rate[side])
        max_value = max(plotted_rate[side])
        
        # Store original min/max values before normalization
        original_values[side] = {
            'min_value': min_value,
            'max_value': max_value,
            'min_angle_idx': plotted_rate[side].index(min_value),
            'max_angle_idx': plotted_rate[side].index(max_value)
        }
        
        # Avoid division by zero - check if max and min are different
        if max_value > min_value:  
            # Apply min-max normalization: (x - min) / (max - min)
            normalized_rate[side] = [(val - min_value) / (max_value - min_value) for val in plotted_rate[side]]
        else:
            # If all values are the same, set normalized values to 0.5
            normalized_rate[side] = [0.5 for _ in plotted_rate[side]]
    
    return normalized_rate, original_values

def add_rate_annotations(ax, original_values, normalized_rates, angles, sides, side_colors):
    """
    Add annotations showing original rate values at min/max points.
    
    Parameters:
    - ax: Matplotlib axis to plot on
    - original_values: Dictionary with original min/max values and indices
    - normalized_rates: Dictionary of normalized rates by side
    - angles: List of angles
    - sides: List of sides
    - side_colors: Dictionary mapping sides to colors
    """
    # Get current axis limits to avoid placing annotations outside visible area
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    y_margin = (ymax - ymin) * 0.05  # 5% margin from top/bottom
    
    # Track used positions to avoid overlaps
    used_positions = {}
    
    for side_idx, side in enumerate(sides):
        min_value = original_values[side]['min_value']/Hz
        max_value = original_values[side]['max_value']/Hz
        min_angle_idx = original_values[side]['min_angle_idx']
        max_angle_idx = original_values[side]['max_angle_idx']
        
        # Format values as Hz or kHz
        if max_value < 1000:
            max_text = f"{max_value:.2f} Hz"
        else:
            max_text = f"{max_value/1000:.2f} kHz"
            
        if min_value < 1000:
            min_text = f"{min_value:.2f} Hz"
        else:
            min_text = f"{min_value/1000:.2f} kHz"
        
        # Calculate positions to avoid overlapping
        min_x = angles[min_angle_idx]
        min_y = normalized_rates[side][min_angle_idx]
        max_x = angles[max_angle_idx]
        max_y = normalized_rates[side][max_angle_idx]
        
        # Offset for max annotation - try to position near the top of the plot
        # but avoid overlapping with other annotations
        max_y_pos = max_y
        max_offset_y = 15 + (side_idx * 15)  # Vertical offset increases with each side
        
        # Check if we're close to another annotation
        position_key = f"{max_x}_{max_y}"
        if position_key in used_positions:
            # Adjust the offset to avoid overlap
            max_offset_y += 15
        used_positions[position_key] = True
        
        # For min annotation, position below the point
        min_y_pos = min_y
        min_offset_y = -15 - (side_idx * 15)  # Vertical offset decreases with each side
        
        # Check if we're close to another annotation
        position_key = f"{min_x}_{min_y}"
        if position_key in used_positions:
            # Adjust the offset to avoid overlap
            min_offset_y -= 15
        used_positions[position_key] = True
        
        # Ensure annotations are within plot boundaries
        if max_y + max_offset_y/72 > ymax - y_margin:
            # If annotation would be outside top, place it below the point instead
            max_offset_y = -15
        
        if min_y + min_offset_y/72 < ymin + y_margin:
            # If annotation would be outside bottom, place it above the point instead
            min_offset_y = 15
            
        # Add annotations (text only, no boxes)
        ax.annotate(
            max_text,
            (max_x, max_y),
            xytext=(10, max_offset_y),
            textcoords='offset points',
            color=side_colors[side],
            fontsize=9,
            fontweight='bold',
            horizontalalignment='left',
            verticalalignment='center'
        )
        
        ax.annotate(
            min_text,
            (min_x, min_y),
            xytext=(10, min_offset_y),
            textcoords='offset points',
            color=side_colors[side],
            fontsize=9,
            fontweight='bold',
            horizontalalignment='left',
            verticalalignment='center'
        )

def plot_cf_intervals_grid(data, intervals, pop='MSO', rate=False):
    # Calculate number of rows and columns for the subplots
    n_intervals = len(intervals)
    n_cols = 2
    n_rows = (n_intervals + n_cols - 1) // n_cols  # Ceiling division
    
    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    axes = axes.flatten()  # Flatten to easily iterate through all axes
    
    # Loop through intervals and create plots
    for i, interval in enumerate(intervals):
        if i < n_intervals:  # Make sure we don't exceed the number of intervals
            # Create a title for this subplot
            title = f"{interval[0]}-{interval[1]} Hz"
            
            # Plot in the corresponding subplot
            draw_rate_vs_angle_pop(
                data=data,
                pop=pop,
                rate=rate,
                cf_interval=interval,
                ax=axes[i],
                title=title
            )
    
    # Hide any unused subplots
    for i in range(n_intervals, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_tonotopic_heatmaps(
    data,
    pop='LSO',
    num_cells_per_interval=50,
    row_norm = True,
    title=None,
    figsize=(18, 12),
    cmap='viridis',
    diff_cmap='coolwarm',
    vmax_given=None,
    y_axis='cf',
    f_ticks=None,
    show_sides=True
):
    """
    Generates heatmaps for auditory neural responses across angles and frequency bands.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing simulation data with angle_to_rate information
    pop : str, default='LSO'
        Population name to analyze (e.g., 'LSO', 'MSO')
    num_cells_per_interval : int, default=50
        Number of cells to include in each frequency interval
    title : str, optional
        Overall title for the plot
    figsize : tuple, default=(18, 12)
        Figure size (width, height) in inches
    cmap : str, default='viridis'
        Colormap for the left and right heatmaps
    diff_cmap : str, default='coolwarm'
        Colormap for the difference heatmap (should be diverging)
    vmax_given : float, optional
        Maximum value for difference colormap normalization
    y_axis : str, default='cf'
        Type of y-axis to display: 'cf' for characteristic frequency or 'cells' for cell numbers
    f_ticks : list, optional
        List of frequency values to display as ticks (if provided, only these ticks will be shown)
    show_sides : bool, default=True
        Whether to show individual left/right heatmaps along with difference map
        
    Returns:
    --------
    fig : matplotlib Figure
        The generated figure with heatmaps
    """
    # Extract required data
    angle_to_rate = data["angle_to_rate"]
    duration = (data.get("simulation_time", data["basesound"].sound.duration / b2.ms) * b2.ms)
    angles = list(angle_to_rate.keys())
    sides = ["L", "R"]
    
    # Get total number of neurons
    num_neurons = len(angle_to_rate[angles[0]]['L'][pop]["global_ids"])
    
    # Generate full CF array using erbspace
    cf = erbspace(CFMIN, CFMAX, num_neurons)
    
    # Calculate number of intervals based on total neurons and interval size
    num_intervals = num_neurons // num_cells_per_interval
    if num_neurons % num_cells_per_interval > 0:
        num_intervals += 1  # Add one more interval for remaining cells
    
    # Initialize matrices to store firing rates
    rate_matrices = {side: np.zeros((num_intervals, len(angles))) for side in sides}
    
    # Store characteristic frequencies for each interval
    cf_ids = np.zeros(num_intervals)
    
    # Calculate the base neuron ID for each side
    base_ids = {
        side: angle_to_rate[angles[0]][side][pop]["global_ids"][0]
        for side in sides
    }
    
    # Calculate firing rates for each interval and angle
    for side in sides:
        neuron_to_cf = {global_id: freq for global_id, freq in zip(angle_to_rate[0][side][pop]["global_ids"], cf)}
        for i in range(num_intervals):
            # Define the range of neurons for this interval
            start_idx = i * num_cells_per_interval
            end_idx = min((i + 1) * num_cells_per_interval, num_neurons)
            
            # Calculate actual neuron IDs
            ymin = base_ids[side] + start_idx
            ymax = base_ids[side] + end_idx - 1  # -1 because end_idx is exclusive
            
            # Calculate central neuron ID and its characteristic frequency
            ycentral = int((ymin + ymax)/2)
            cf_ids[i] = neuron_to_cf[ycentral]
            
            for j, angle in enumerate(angles):
                # Filter spikes within the specified range
                cluster_mask = (
                    (angle_to_rate[angle][side][pop]['senders'] >= ymin) & 
                    (angle_to_rate[angle][side][pop]['senders'] <= ymax)
                )
                cluster_times = angle_to_rate[angle][side][pop]['times'][cluster_mask]
                
                # Calculate average firing rate for this interval and angle
                num_cells = end_idx - start_idx
                firing_rate = len(cluster_times) / (duration * num_cells)
                rate_matrices[side][i, j] = firing_rate
    
    # Normalize each row by its maximum value
    if(row_norm):
        for side in sides:
            for i in range(num_intervals):
                max_val = np.max(rate_matrices[side][i, :])
                if max_val > 0:  # Avoid division by zero
                    rate_matrices[side][i, :] = rate_matrices[side][i, :] / max_val
    
    # Calculate difference matrix (L - R)
    diff_matrix = rate_matrices['L'] - rate_matrices['R']
    
    # Create figure based on display mode
    if show_sides:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Generate side heatmaps
        im_left = axes[0].imshow(rate_matrices['L'], cmap=cmap, aspect='auto', interpolation='none')
        im_right = axes[1].imshow(rate_matrices['R'], cmap=cmap, aspect='auto', interpolation='none')
        
        # Add colorbars for side plots
        cbar_left = plt.colorbar(im_left, ax=axes[0])
        cbar_right = plt.colorbar(im_right, ax=axes[1])
        if(row_norm):
            cbar_left.set_label('Normalized Firing Rate')
            cbar_right.set_label('Normalized Firing Rate')
        else:
            cbar_left.set_label('Firing Rate')
            cbar_right.set_label('Firing Rate')
        
        # Set titles for side plots
        axes[0].set_title('Left Side')
        axes[1].set_title('Right Side')
        
        # Set up diff plot (ax=axes[2])
        diff_ax = axes[2]
        diff_ax.set_title('Difference (Left - Right)')
    else:
        fig, diff_ax = plt.subplots(1, figsize=figsize)
    
    # For difference heatmap, use a diverging colormap centered at zero
    vmax = max(abs(np.min(diff_matrix)), abs(np.max(diff_matrix)))
    print('previously vmax set at ', vmax)
    if vmax_given is not None and vmax_given >= vmax:
        vmax = vmax_given
        print('now vmax set at ', vmax)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    
    # Create difference heatmap
    im_diff = diff_ax.imshow(diff_matrix, cmap=diff_cmap, aspect='auto', 
                       interpolation='none', norm=norm)
    cbar_diff = plt.colorbar(im_diff, ax=diff_ax)
    cbar_diff.set_label('Difference (L - R)')
    
    # Define function to set up axes formatting
    def setup_axis(ax):
        # Set x-axis labels (angles)
        ax.set_xticks(np.arange(len(angles)))
        ax.set_xticklabels([f"{angle}°" for angle in angles])
        ax.set_xlabel('Azimuth Angle')
        
        # Set y-axis formatting
        if f_ticks is not None and y_axis == 'cf':
            # Use only the specified frequency ticks
            y_positions = []
            y_labels = []
            
            # Add fixed 20 Hz tick at the bottom edge (not centered)
            y_positions.append(num_intervals - 0.5)  # Bottom edge
            y_labels.append("20 kHz")

            # Add fixed 20 kHz tick at the top edge (not centered)
            y_positions.append(-0.5)  # Top edge
            y_labels.append("125 Hz")

            for freq in f_ticks:
                # Format frequency label
                if freq < 1000:
                    label = f"{int(round(freq))} Hz"
                else:
                    label = f"{int(freq/1000)} kHz"
                
                # Find approximate position (closest interval) for this frequency
                distances = np.abs(cf_ids - freq)
                closest_idx = np.argmin(distances)
                
                y_positions.append(closest_idx)
                y_labels.append(label)
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels)
        else:
            # Show all intervals
            ax.set_yticks(np.arange(num_intervals))
            
            if y_axis == 'cf':
                # Format frequency labels
                freq_labels = []
                for freq in cf_ids:
                    if freq < 1000:
                        freq_labels.append(f"{int(round(freq))} Hz")
                    else:
                        freq_labels.append(f"{freq/1000:.1f}k Hz")
                
                ax.set_yticklabels(freq_labels)
            else:  # y_axis == 'cells'
                # Show cell number ranges
                cell_labels = []
                for i in range(num_intervals):
                    start_idx = i * num_cells_per_interval
                    end_idx = min((i + 1) * num_cells_per_interval, num_neurons) - 1
                    cell_labels.append(f"{start_idx}-{end_idx}")
                
                ax.set_yticklabels(cell_labels)
        
        ax.set_ylabel('Characteristic Frequency' if y_axis == 'cf' else 'Cell Indices')
        ax.invert_yaxis()  # Ensure low frequencies are at the bottom
    
    # Apply axis formatting to all axes
    if show_sides:
        for ax in axes:
            setup_axis(ax)
    else:
        setup_axis(diff_ax)
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)  # Make room for the overall title
        fig.suptitle(title, fontsize=16)
    
    return

def draw_rate_vs_angle_pop_multi(
    
    data_list,                      # list of datasets
    title=None,
    pop='LSO',
    rate=True,                      # True = avg, False = population
    cf_interval=None,
    sides=None,
    color=None,
    norm=False,                     # False, True, "zscore", "minmax"
    figsize=[7,4],
    ax=None,
    ylim=None,
    label=None,
    error_type="sem"                # "sem" or "std"
):
    # -----------------------------------------------------
    # Setup
    # -----------------------------------------------------
    if sides is None:
        sides = ["L","R"]

    angles = list(data_list[0]["angle_to_rate"].keys())

    # Colors
    if isinstance(color, dict):
        side_colors = color
    elif isinstance(color, str):
        side_colors = {sides[0]: color}
    else:
        side_colors = {"L": "m", "R": "g"}

    # Accumulators
    avg_accum = {s: [] for s in sides}
    pop_accum = {s: [] for s in sides}

    # -----------------------------------------------------
    # Extract rate curves from each dataset
    # -----------------------------------------------------
    for data in data_list:
        angle_to_rate = data["angle_to_rate"]
        duration = data.get(
            "simulation_time",
            data["basesound"].sound.duration / b2.ms
        ) * b2.ms

        tot_spikes, avg_neuron_rate, _ = calculate_firing_rates(
            angle_to_rate, pop, sides, angles, duration, cf_interval
        )

        for s in sides:
            avg_accum[s].append(np.array(avg_neuron_rate[s]))
            pop_accum[s].append(np.array(tot_spikes[s]))

    # -----------------------------------------------------
    # Compute mean + error
    # -----------------------------------------------------
    def compute_mean_and_error(accum):
        arr = np.vstack(accum)
        mean = arr.mean(axis=0)

        if error_type == "std":
            err = arr.std(axis=0)
        else:
            err = arr.std(axis=0) / np.sqrt(arr.shape[0])  # SEM

        return mean, err

    avg_mean, avg_err = {}, {}
    pop_mean, pop_err = {}, {}

    for s in sides:
        avg_mean[s], avg_err[s] = compute_mean_and_error(avg_accum[s])
        pop_mean[s], pop_err[s] = compute_mean_and_error(pop_accum[s])

    # -----------------------------------------------------
    # Choose rate type
    # -----------------------------------------------------
    if rate is True:
        mean_used = avg_mean
        err_used = avg_err
        ylabel_text = "Avg Firing Rate [Hz]"
    else:
        mean_used = pop_mean
        err_used = pop_err
        ylabel_text = "Population Firing Rate [Hz]"

    # -----------------------------------------------------
    # NORMALIZATION OPTIONS
    # norm = False | True | "zscore" | "minmax"
    # -----------------------------------------------------
    original_values = None

    if norm:
        original_values = {s: mean_used[s].copy() for s in sides}

        for s in sides:
            x = mean_used[s]
            err = err_used[s]

            if norm is True:
                # Normalize by maximum
                max_val = x.max() if x.max() != 0 else 1e-9
                mean_used[s] = x / max_val
                err_used[s] = err / max_val

                

            elif norm == "zscore":
                mu = x.mean()
                sigma = x.std() if x.std() > 1e-12 else 1e-12
                mean_used[s] = (x - mu) / sigma
                err_used[s] = err / sigma


            elif norm == "minmax":
                mn = x.min()
                mx = x.max()
                rng = mx - mn if mx != mn else 1e-9
                mean_used[s] = (x - mn) / rng
                err_used[s] = err / rng



            else:
                raise ValueError("norm must be False, True, 'zscore', or 'minmax'")
            
        ylabel_text = "Normalized " + ylabel_text

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for s in sides:
        pl = label if label else f"{s} (mean)"

        ax.errorbar(
            angles,
            mean_used[s],
            yerr=err_used[s],
            fmt='o-',
            color=side_colors[s],
            ecolor=side_colors[s],
            elinewidth=1.4,
            capsize=4,
            label=pl
        )

    if len(sides) > 1 and label is None:
        ax.legend()

    # Aesthetics
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{a}°" for a in angles])
    ax.set_xlabel("Azimuth Angle")
    ax.set_ylabel(ylabel_text)

    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax, original_values if norm else ax

def plot_anf_rasterplot(
    spikes_series,
    figsize=(10, 5),
    color="black",
    linewidth=0.5,
    remove_yticks=False,
    title="ANF Raster Plot",
):
    """
    Plot a raster plot from a Pandas Series of spike times.
    
    spikes_series: pd.Series
        Each entry is a list of spike times for one neuron.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.eventplot(
        spikes_series.values,
        colors=color,
        linewidths=linewidth
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index")

    if remove_yticks:
        ax.set_yticks([])

    ax.set_title(title)
    plt.tight_layout()
    return fig, ax

def plot_sound(
    sound,
    figsize=(10, 4),
    title="Sound waveform",
    time_in_ms=False,
    xlim = None):
    """
    Plot a Brian2Hears Sound object over time.
    Works for mono or stereo.
    
    sound: brian2hears.Sound
    """
    # Sound → numpy array: shape (samples, channels)
    wave = np.array(sound)
    fs = float(sound.samplerate)

    # time axis
    t = np.arange(wave.shape[0]) / fs
    if time_in_ms:
        t = t * 1000
        xlabel = "Time (ms)"
    else:
        xlabel = "Time (s)"

    fig, ax = plt.subplots(figsize=figsize)

    # mono
    if wave.ndim == 2 and wave.shape[1] == 1:
        ax.plot(t, wave[:, 0], linewidth=0.8)
    # stereo
    else:
        for ch in range(wave.shape[1]):
            ax.plot(t, wave[:, ch], linewidth=0.8, label=f"Ch {ch}")
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    return fig, ax

def greenwood_human(x, A=165.4, a=2.1, k=1.0):
    return A * (10**(a * x) - k)

def greenwood_inverse(f, A=165.4, a=2.1, k=1.0):
    # compute x from frequency (inverse Greenwood)
    return (1.0 / a) * np.log10(f / A + k)

def greenwood_cf_array(CFMIN, CFMAX, n_neurons):
    # convert CF bounds (in Hz) -> positions
    print('CFMIN=', CFMIN)
    x_min = greenwood_inverse(CFMIN)
    print(x_min)
    x_max = greenwood_inverse(CFMAX)

    # linearly spaced positions along the cochlea
    x = np.linspace(x_min, x_max, n_neurons)

    # forward Greenwood: positions -> frequencies
    cf = greenwood_human(x)
    return cf * Hz  # keep Brian2 unit

def draw_spikes_and_psth_bothside(
    res,
    angle,
    pop,
    y_ax='cf_custom',
    f_ticks=[125, 1000, 10000],
    title=None,
    xlim=None,
    ylim=None,
    bin_size=1,       # histogram time bin in ms
    cf_bin_size=3,    # CF histogram binning (neurons per bar)
    raster_dot_size=1,
    figsize=(10, 12)
):

    side_colors = {'L': 'm', 'R': 'g'}

    duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

    if xlim is None:
        xlim = [0, duration]

    if ylim is None:
        ylim = [CFMIN/Hz, CFMAX/Hz]

    # -----------------------------------------------------------
    # LAYOUT
    # -----------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        4, 2, figure=fig,
        width_ratios=[4, 1],    # raster 80%, histogram 20%
        height_ratios=[0.4, 1, 1, 0.8],
        hspace=0.35,
        wspace=0.05,
    )

    ax_sound = fig.add_subplot(gs[0, 0])
    sound = res["basesound"].sound
    t = np.arange(len(sound)) / sound.samplerate * 1000
    ax_sound.plot(t, sound, 'b', lw=2)
    ax_sound.set_ylabel("Amplitude")
    ax_sound.set_xlim(xlim)
    ax_sound.set_title("Sound waveform")
    ax_sound.grid(True, alpha=0.3)


    ax_raster_L = fig.add_subplot(gs[1, 0])
    ax_hist_L   = fig.add_subplot(gs[1, 1], sharey=ax_raster_L)

    ax_raster_R = fig.add_subplot(gs[2, 0])
    ax_hist_R   = fig.add_subplot(gs[2, 1], sharey=ax_raster_R)

    ax_psth     = fig.add_subplot(gs[3, 0], sharex=ax_raster_L)


    # ===========================================================
    # Helper function to compute FINAL spike set (time+CF filtered)
    # ===========================================================
    def filter_spikes(spikes, xlim, ylim):
        times = spikes["times"]
        senders = spikes["senders"]
        gids = spikes["global_ids"]
        n = len(gids)
        #print('Total spikes before filtering:', len(times))
        # Time filtering
        mask_t = (times >= xlim[0]) & (times <= xlim[1])
        times_t = times[mask_t]
        senders_t = senders[mask_t]
        #print('Total spikes after time filtering:', len(times_t))

        # CF mapping
        cf_hz = erbspace(CFMIN, CFMAX, n) / b2.Hz
        _, ymin_idx = take_closest(cf_hz, ylim[0])
        #print('cf_min=', cf_hz[ymin_idx])
        _, ymax_idx = take_closest(cf_hz, ylim[1])
        #print('cf_max=', cf_hz[ymax_idx])

        cf_min_id = gids[0] + ymin_idx
        cf_max_id = gids[0] + ymax_idx
    
        # CF filtering
        #print('Total cell considered before CF filtering:', len(senders_t))
        mask_cf = (senders_t >= cf_min_id) & (senders_t <= cf_max_id)
        #print('Total spikes after CF filtering:', np.sum(mask_cf))
        return (
            times_t[mask_cf],     # times inside window
            senders_t[mask_cf],   # senders inside window
            cf_hz,
            ymin_idx,
            ymax_idx,
            gids
        )


    # ===========================================================
    # RASTER + CF HISTOGRAMS
    # ===========================================================
    for side, ax_raster, ax_hist in [
        ('L', ax_raster_L, ax_hist_L),
        ('R', ax_raster_R, ax_hist_R)
    ]:

        spikes = res["angle_to_rate"][angle][side][pop]

        # Filter spikes according to visible time & CF window
        times_f, senders_f, cf_full, ymin_idx, ymax_idx, gids = filter_spikes(
            spikes, xlim, ylim
        )

        n_neurons = len(gids)
        local_ids_f = senders_f - gids[0]

        # RASTER MAPPING
        if y_ax == "cf":
            cf_hz = cf_full
            y_values = cf_hz[local_ids_f]
            ax_raster.set_ylabel(f"{pop} CF [Hz]")
            ax_raster.set_ylim(ylim)

        else:
            y_values = local_ids_f
            ax_raster.set_ylim([ymin_idx, ymax_idx])

            if y_ax == "ids":
                ax_raster.set_ylabel(f"{pop} IDs")
            elif y_ax == "global_ids":
                ax_raster.set_ylabel(f"{pop} global IDs")
            elif y_ax == "cf_custom":
                ax_raster.set_ylabel(f"{pop} CF [Hz]")
                tick_pos = []
                for f in f_ticks:
                    _, idx = take_closest(cf_full, f)
                    tick_pos.append(idx)
                ax_raster.set_yticks(tick_pos)
                ax_raster.set_yticklabels(f_ticks)

        # Plot RASTER
        ax_raster.plot(times_f, y_values, '.', color=side_colors[side], markersize=raster_dot_size)
        ax_raster.set_xlim(xlim)
        ax_raster.text(
            0.0, 1.05, f"{side} side",
            transform=ax_raster.transAxes,
            fontsize=12, fontweight='bold',
            color=side_colors[side]
        )

        # ===================================================
        # CF HISTOGRAM: computed ONLY from filtered spikes
        # ===================================================
        spike_count = np.bincount(local_ids_f, minlength=n_neurons)

        # group CF bins visually
        bins_cf = np.arange(0, n_neurons, cf_bin_size)
        grouped_counts = [spike_count[i:i+cf_bin_size].sum() for i in bins_cf]
        grouped_y = [np.arange(n_neurons)[i:i+cf_bin_size].mean() for i in bins_cf]

        # restrict to ylim region
        grouped_y = np.array(grouped_y)
        grouped_counts = np.array(grouped_counts)
        mask_vis = (grouped_y >= ymin_idx) & (grouped_y <= ymax_idx)

        ax_hist.barh(
            grouped_y[mask_vis],
            grouped_counts[mask_vis],
            height=0.8 * cf_bin_size,
            color=side_colors[side],
            alpha=0.4
        )
        ax_hist.set_ylim(ax_raster.get_ylim())
        ax_hist.set_xlabel("Spike count")
        ax_hist.grid(False)
        ax_hist.tick_params(axis='y', labelleft=False)


    # ===========================================================
    # PSTH — ALSO FILTERED BY TIME & CF RANGE
    # ===========================================================
    for side in ['L', 'R']:
        color = side_colors[side]
        spikes = res["angle_to_rate"][angle][side][pop]

        times_f, senders_f, cf_full, ymin_idx, ymax_idx, gids = filter_spikes(
            spikes, xlim, ylim
        )

        bins = np.arange(xlim[0], xlim[1] + bin_size, bin_size)
        ax_psth.hist(times_f, bins=bins, alpha=0.4, color=color, label=side)

    ax_psth.set_xlabel("Time [ms]")
    ax_psth.set_ylabel("Spike count")
    ax_psth.legend()
    ax_psth.grid(True, alpha=0.3)
    ax_psth.spines['top'].set_visible(False)
    ax_psth.spines['right'].set_visible(False)


    # -----------------------------------------------------------
    # TITLE & LAYOUT
    # -----------------------------------------------------------
    if title:
        fig.suptitle(title)
        plt.subplots_adjust(top=0.92)

    plt.show()

    return

def draw_rate_vs_angle(
    data,
    title,
    rate=True,  # True=avg, False=population, 'diff'=normalized difference
    show_pops=["ANF", "SBC", "GBC", "MNTBC", "MSO", "LSO"],
    ylim=None,
    hist_logscale=True,
    show_hist=True,
    cf_interval=None
):
    angle_to_rate = data["angle_to_rate"]
    duration = data.get("simulation_time", data["basesound"].sound.duration / b2.ms) * b2.ms
    angles = list(angle_to_rate.keys())
    sides = ["L", "R"]

    n_rows = math.ceil(len(show_pops)/2)
    n_cols = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 2*len(show_pops)))
    ax = np.array(ax).flatten()

    for i, pop in enumerate(show_pops):
        tot_spikes, avg_neuron_rate, active_neuron_rate = calculate_firing_rates(
            angle_to_rate, pop, sides, angles, duration, cf_interval
        )
        ax_i = ax[i]

        # Determine plotted rates
        if rate == True:
            plotted_rate = avg_neuron_rate
            ylabel_text = "Avg Firing Rate [Hz]"
        elif rate == False:
            plotted_rate = tot_spikes
            ylabel_text = "Pop Firing Rate [Hz]"
        elif rate == 'norm':
            plotted_rate, original_values = normalize_rates(avg_neuron_rate, sides)
            ylabel_text = "Normalized Avg Firing Rate"
            add_rate_annotations(ax_i, original_values, plotted_rate, angles, sides, {"L":"m","R":"g"})
        elif rate == 'diff':
            plotted_rate = {
                "L": np.array(avg_neuron_rate["L"]) / np.max(avg_neuron_rate["L"]),
                "R": np.array(avg_neuron_rate["R"]) / np.max(avg_neuron_rate["R"]),
                "L_pop": np.array(tot_spikes["L"]) / np.max(tot_spikes["L"]),
                "R_pop": np.array(tot_spikes["R"]) / np.max(tot_spikes["R"])
            }
            ylabel_text = "Diff in Avg-TotSpikes Rate"
        else:
            raise ValueError("rate must be True, False, or 'diff'")

        # Plotting
        if rate != 'diff':
            for side, color in zip(sides, ['m','g']):
                ax_i.plot(angles, plotted_rate[side], 'o-', color=color, label=side)
            if len(sides)>1:
                ax_i.legend()
        else:
            ax_i.plot(angles, plotted_rate["L"], 'o-', color='m', label='Avg_L')
            ax_i.plot(angles, plotted_rate["L_pop"], 'o-', color='darkmagenta', label='Pop_L')
            ax_i.plot(angles, plotted_rate["R"], 'o-', color='g', label='Avg_R')
            ax_i.plot(angles, plotted_rate["R_pop"], 'o-', color='darkgreen', label='Pop_R')
            ax_i.legend()

        # Formatting
        ax_i.set_xticks(angles)
        ax_i.set_xticklabels([f"{j}°" for j in angles])
        ax_i.set_xlabel("Azimuth Angle")
        ax_i.set_ylabel(ylabel_text)
        if ylim:
            ax_i.set_ylim(ylim)
        ax_i.set_title(pop)
        ax_i.spines['top'].set_visible(False)

        # Optional histogram
        if show_hist:
            v = ax_i.twinx()
            v.grid(False)
            distr = {side: [firing_neurons_distribution(angle_to_rate[angle][side][pop]) for angle in angles] for side in sides}
            senders_renamed = {side: [shift_senders(angle_to_rate[angle][side][pop], hist_logscale) for angle in angles] for side in sides}
            max_spikes_single_neuron = max(flatten(distr.values()))
            draw_hist(v, senders_renamed, angles, num_neurons=len(angle_to_rate[0]["L"][pop]["global_ids"]), max_spikes_single_neuron=max_spikes_single_neuron, logscale=hist_logscale)

    plt.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()
    return ax

def draw_rate_vs_angle_pop(
    data,
    pop='LSO',
    rate=True,  # True=avg, False=population, 'mm_norm', 'max_norm', diff'=normalized difference
    cf_interval=None,
    sides=None,
    color=None,
    show_hist=True,
    hist_logscale=True,
    figsize=[7,4],
    ax=None,
    title=None,
    ylim=None,
    label=None
):
    angle_to_rate = data["angle_to_rate"]
    duration = data.get("simulation_time", data["basesound"].sound.duration / b2.ms) * b2.ms
    angles = list(angle_to_rate.keys())
    if sides is None:
        sides = ["L","R"]

    # Side colors
    if isinstance(color, dict):
        side_colors = color
    elif isinstance(color, str):
        side_colors = {sides[0]: color}
    else:
        side_colors = {"L":"m","R":"g"}

    tot_spikes, avg_neuron_rate, active_neuron_rate = calculate_firing_rates(
        angle_to_rate, pop, sides, angles, duration, cf_interval
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig=True
    else:
        created_fig=False

    original_values = None

    # Determine plotted rates
    if rate == True:
        plotted_rate = avg_neuron_rate
        ylabel_text = "Avg Firing Rate [Hz]"
    elif rate == False:
        plotted_rate = tot_spikes
        ylabel_text = "Pop Firing Rate [Hz]"
    elif rate == 'mm_norm':
        plotted_rate, original_values = normalize_rates(avg_neuron_rate, sides)
        ylabel_text = "Min-Max Norm Firing Rate"
    elif rate == 'max_norm':
        plotted_rate = {
            "L": np.array(avg_neuron_rate["L"]) / np.max(avg_neuron_rate["L"]),
            "R": np.array(avg_neuron_rate["R"]) / np.max(avg_neuron_rate["R"]),
        }
        ylabel_text = "Max Norm Firing Rate"
    elif rate == 'diff':
        plotted_rate = {
            "L": np.array(avg_neuron_rate["L"]) / np.max(avg_neuron_rate["L"]),
            "R": np.array(avg_neuron_rate["R"]) / np.max(avg_neuron_rate["R"]),
            "L_pop": np.array(tot_spikes["L"]) / np.max(tot_spikes["L"]),
            "R_pop": np.array(tot_spikes["R"]) / np.max(tot_spikes["R"])
        }
        ylabel_text = "Diff Avg-Pop Computation"
    else:
        raise ValueError("rate must be True, False, or 'diff'")


    # Plotting
    if show_hist:
        v = ax.twinx()
        #v.set_zorder(ax.get_zorder() - 1)
        v.grid(False)
        distr = {side: [firing_neurons_distribution(angle_to_rate[angle][side][pop]) for angle in angles] for side in sides}
        senders_renamed = {side: [shift_senders(angle_to_rate[angle][side][pop], hist_logscale) for angle in angles] for side in sides}
        max_spikes_single_neuron = max(flatten(distr.values()))
        draw_hist(v, senders_renamed, angles, num_neurons=len(angle_to_rate[0]["L"][pop]["global_ids"]), max_spikes_single_neuron=max_spikes_single_neuron, logscale=hist_logscale)
        v.spines["right"].set_visible(True)

    if rate != 'diff' and rate != 'max_norm':
        for side in sides:
            line_label = label if label is not None else f"Side {side}"
            ax.plot(angles, plotted_rate[side], 'o-', color=side_colors.get(side,'k'), label=line_label)
        if len(sides)>1 and label is None:
            ax.legend()
    elif rate == 'max_norm':
        ax.plot(angles, plotted_rate["L"], 'o-', color='m', label='Avg_L')
        ax.plot(angles, plotted_rate["R"], 'o-', color='g', label='Avg_R')
        ax.legend()
    elif rate == 'diff':
        ax.plot(angles, plotted_rate["L"], 'o-', color='m', label='Avg_L')
        ax.plot(angles, plotted_rate["L_pop"], 'o-', color='darkmagenta', label='Pop_L')
        ax.plot(angles, plotted_rate["R"], 'o-', color='g', label='Avg_R')
        ax.plot(angles, plotted_rate["R_pop"], 'o-', color='darkgreen', label='Pop_R')
        ax.legend()

    ax.set_xticks(angles)
    ax.set_xticklabels([f"{j}°" for j in angles])
    ax.set_xlabel("Azimuth Angle")
    ax.set_ylabel(ylabel_text)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)


    if created_fig:
        plt.tight_layout()
        plt.show()

    return ax

def draw_rate_vs_angle_comparison(
    data1,
    data2,
    side = 'L',
    title=None,
    pop='LSO',
    labels = None,
    rate=True,
    cf_interval=None,
    figsize=[7, 4],
    ax=None,  # Add parameter to accept an existing axis
):
    
    # Create a new figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        created_fig = False
    
    for data, col, lab in zip([data1, data2], ['m', 'k'], labels):
        angle_to_rate = data["angle_to_rate"]
        duration = (data.get("simulation_time", data["basesound"].sound.duration / b2.ms) * b2.ms)
        angles = list(angle_to_rate.keys())
        num_neurons = len(angle_to_rate[0][side][pop]["global_ids"])
        cf = erbspace(CFMIN, CFMAX, num_neurons)

        if cf_interval == None:
            tot_spikes = {
                len(angle_to_rate[angle][side][pop]["times"]) / duration
                for angle in angles
            }
            active_neuron_rate = {
                avg_fire_rate_actv_neurons(angle_to_rate[angle][side][pop])
                * (1 * b2.second)
                / duration
                for angle in angles
            }
            distr = {
                firing_neurons_distribution(angle_to_rate[angle][side][pop])
                for angle in angles
            }
        else:
            tot_spikes = {}
            active_neuron_rate = {}
            cluster_numerosity = {}
            tot_spikes[side] = []
            active_neuron_rate[side] = []
            for angle in angles:
                # Find indices in CF array corresponding to interval bounds
                _, ymin_idx = take_closest(cf, cf_interval[0]*Hz)
                _, ymax_idx = take_closest(cf, cf_interval[1]*Hz)

                # Calculate actual neuron IDs from global_ids and indices
                base_id = angle_to_rate[angle][side][pop]["global_ids"][0]
                ymin = base_id + ymin_idx
                ymax = base_id + ymax_idx

                # Filter spikes within the specified range
                cluster_mask = (angle_to_rate[angle][side][pop]['senders'] >= ymin) & (angle_to_rate[angle][side][pop]['senders'] <= ymax)
                cluster_times = angle_to_rate[angle][side][pop]['times'][cluster_mask]

                # Calculate rate for this angle and side
                tot_spikes[side].append(len(cluster_times) / duration)
                active_neuron_rate[side].append(len(cluster_times)/((ymax - ymin)*duration))

                # NEW: Compute cluster numerosity (unique senders)
                unique_senders = np.unique(angle_to_rate[angle][side][pop]['senders'][cluster_mask])
            cluster_numerosity[side] = len(unique_senders)
            print(f"side {side}, considered {cluster_numerosity[side]} cells of {num_neurons} total")

        plotted_rate = active_neuron_rate if rate else tot_spikes
        ax.plot(angles, plotted_rate["L"]/np.max(plotted_rate["L"]), 'o-', color = col, label = lab)

    ylabel_text = "Avg Firing Rate [Hz]" if rate else "Pop Firing Rate [Hz]"
    ax.set_ylabel(ylabel_text)
    ax.set_xticks(angles)
    ax.set_xlabel("Azimuth Angle")
    ax.legend()
    ax.set_xticklabels([f"{j}°" for j in angle_to_rate.keys()])
    
    # Set title if provided
    if title:
        ax.set_title(title)

    if created_fig:
        plt.tight_layout()
        plt.show()
        
    return ax
