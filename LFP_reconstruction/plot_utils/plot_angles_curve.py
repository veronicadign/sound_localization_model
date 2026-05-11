import os
import h5py
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def plot_lfp_tuning_curve():
    angles = [-90, -45, 0, 45, 90]
    side = 'L'  # Assuming you want to plot for the left ear (ipsilateral)
    base_output_dir = "RESULTS/lfp_tmp/output_baseline_simulation_"
    
    # Settings for phase-extraction (matches your previous logic)
    skip_ms = 10.0
    stimulus_freq = 500.0 # Make sure this matches what you simulated!
    
    amplitudes = []
    valid_angles = []

    for angle in angles:
        h5_path = os.path.join(base_output_dir + f"angle{angle}_{side}", "PointSourcePotential_sum.h5")
        print(h5_path)
        
        if not os.path.exists(h5_path):
            print(f"Warning: File not found for angle {angle}. Skipping.")
            continue
            
        with h5py.File(h5_path, 'r') as f:
            lfp = f['data'][()] * 1e3  # mV → µV
            srate = float(f['srate'][()])
            
        # 1. Isolate the central electrode (closest to the soma, usually the largest sink)
        n_ch = lfp.shape[0]
        center_ch = n_ch // 2 
        lfp_center = lfp[center_ch, :] # Shape: (timepoints,)
        
        # 2. Skip ramp and detrend
        dt_ms = 1e3 / srate
        skip_idx = int(skip_ms / dt_ms)
        lfp_ss = lfp_center[skip_idx:]
        lfp_ss = scipy.signal.detrend(lfp_ss)
        
        # 3. Phase-fold the signal
        T_int = max(1, int(round(1e3 / stimulus_freq / dt_ms)))
        n_cycles = len(lfp_ss) // T_int
        
        lfp_phase = (lfp_ss[:n_cycles * T_int]
                     .reshape(n_cycles, T_int)
                     .mean(axis=0))
                     
        # 4. Calculate Peak-to-Peak Amplitude (Max - Min of the average cycle)
        peak_to_peak = np.max(lfp_phase) - np.min(lfp_phase)
        
        amplitudes.append(peak_to_peak)
        valid_angles.append(angle)

    # --- PLOTTING ---
    plt.figure(figsize=(8, 5))
    
    # Plot the tuning curve
    plt.plot(valid_angles, amplitudes, marker='o', color='firebrick', linewidth=2, markersize=8)
    
    plt.title('MSO LFP Tuning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Sound Azimuth Angle (°)', fontsize=12)
    plt.ylabel('LFP Max Peak Amplitude (µV)', fontsize=12)
    plt.xticks(valid_angles)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Optional: Highlight the "Best Angle"
    best_idx = np.argmax(amplitudes)
    plt.axvline(valid_angles[best_idx], color='gray', linestyle='--', alpha=0.5)
    plt.text(valid_angles[best_idx] + 2, np.min(amplitudes), f"Max @ {valid_angles[best_idx]}°", color='gray')

    out_file = "MSO_Tuning_Curve.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.show()
    print(f"Tuning curve saved to {out_file}")

if __name__ == "__main__":
    plot_lfp_tuning_curve()