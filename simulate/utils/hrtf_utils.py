import math
import brian2 as b2
import numpy as np
from brian2 import ms
from brian2hears import IRCAM_LISTEN, Sound, dB
from typing import Union  # <-- ADD THIS

from utils.path_utils import Paths
from utils.custom_sounds import Tone, ToneBurst
from utils.log_utils import logger
from utils.manual_fixes_to_b2h.HeadlessDatabase import HeadlessDatabase
from sorcery import dict_of
from .cochlea_utils import ANGLE_TO_SOFA
import os
from scipy.signal import fftconvolve, resample
import pysofaconventions as sofa
from scipy.fft import rfft, irfft
import matplotlib.pyplot as plt


def apply_gating(sound: Sound, ramp_ms: float = 10.0) -> Sound:
    """
    Apply raised-cosine (Hann) onset/offset gating to a Brian2Hears Sound.

    Parameters
    ----------
    sound : Sound
        Brian2Hears Sound object, mono or binaural.
    ramp_ms : float
        Duration of onset/offset ramps in milliseconds.

    Returns
    -------
    Sound
        Gated sound (same samplerate and duration).
    """
    fs = int(sound.samplerate)
    n_samples = sound.nsamples
    ramp_samples = int((ramp_ms / 1000.0) * fs)

    if ramp_samples <= 1:
        return sound   # no gating needed

    # --- Create raised-cosine (Hann) ramps ---
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))

    # full envelope = [ramp_up | steady | ramp_down]
    envelope = np.ones(n_samples)
    envelope[:ramp_samples] = ramp
    envelope[-ramp_samples:] = ramp[::-1]

    # Apply to sound (supports mono or 2 channels)
    gated_data = sound.data * envelope.reshape(-1, 1)

    # Create new Sound object with same samplerate
    gated_sound = Sound(gated_data, samplerate=fs * b2.Hz)

    return gated_sound

def sel_range(s, start=0 * ms, end=10 * ms):
    return s[start:end]

def angle_to_itd(angle, w_head: int = 22, v_sound: int = 33000):
    delta_x = w_head * np.sin(np.deg2rad(angle))
    return round(1000 * delta_x / v_sound, 2) * b2.ms

def calculate_lefttoright_level_diff(freq, angle):
    azimuth_rad = np.radians(angle)
    head_radius = 0.0875  # meters
    speed_of_sound = 343.0  # m/s
    # Calculate wavelength and ka (wavelength * radius)
    wavelength = speed_of_sound / (freq / b2.Hz)
    ka = 2 * np.pi * head_radius / wavelength
    # Frequency-dependent shadowing effect
    # Higher ka values (higher frequencies or larger heads) create more shadowing
    shadowing = np.minimum(20, ka * 2)  # Limited to 20 dB maximum
    # Calculate ILD based on shadowing and azimuth
    # The sin term gives directionality, shadowing term scales with frequency
    ild = shadowing * np.sin(azimuth_rad)
    # returns left - right difference
    return np.abs(ild) * dB

def synthetic_ild(sound: Tone, angle: int):
    if type(sound) is not Tone:
        # Linear interpolation between +15dB and -15dB for angles between -90 and +90
        # For angle 0, diff will be 0dB
        diff = np.abs((angle / 90) * 15) * dB  # This gives +15 for -90, 0 for 0, and -15 for +90
    else:
        diff = calculate_lefttoright_level_diff(sound.frequency, angle)
    
    logger.debug(f"ILD calculated as {diff}")
    left = Sound(sound.sound)
    right = Sound(sound.sound)
    if angle > 0:
        left.set_level(left.get_level() - diff)
    else:
        right.set_level(right.get_level() - diff) 

    # azimuth_rad = np.radians(angle)
    # max_mask = np.abs(np.sin(azimuth_rad))
    # if angle < 0:  # Sound comes from the left
    #     # Left ear gets less masking, right ear gets more
    #     left_masking_factor = -(1 - max_mask)
    #     right_masking_factor = -max_mask
    # else:  # Sound comes from the right
    #     # Right ear gets less masking, left ear gets more
    #     left_masking_factor = max_mask
    #     right_masking_factor = 1 - max_mask

    # logger.debug(
    #     f"{angle} -> left {diff * left_masking_factor}; right {diff * right_masking_factor}"
    # )
    # logger.debug(
    #     f"original ILD: {diff}; new ILD: {diff * left_masking_factor + diff * right_masking_factor}"
    # )
    # # logger.debug(
    # #     f"left {diff} * {left_masking_factor}; right {diff} * {right_masking_factor}"
    # # )
    # left.level += diff * left_masking_factor
    # right.level += diff * right_masking_factor

    # previous synthetic_ild(sound: Tone, angle: int):
#     if type(sound) is not Tone:
#         logger.error(f"selected HRTF synthetic_ild, but it only supports Tones")
#         raise TypeError(f"sound is {type(sound)}, while it should be {Tone}")
#     if angle is 0:
#         return Sound((sound.sound, sound.sound), samplerate=sound.sound.samplerate)
#     diff = calculate_lefttoright_level_diff(sound.frequency, angle)
#     left = Sound(sound.sound)
#     right = Sound(sound.sound)
#     if diff < 0 * dB:
#         left.level += diff
#     else:
#         right.level -= diff
#     return Sound((left, right), samplerate=sound.sound.samplerate)

    return Sound((left, right), samplerate=sound.sound.samplerate)

def run_hrtf_ircam(sound: Union[Sound, Tone, ToneBurst], angle, hrtf_params) -> tuple[Sound, Sound]:
    logger.debug(f"[run_hrtf] Starting HRTF for angle={angle} subj={hrtf_params['subj_number']}")

    subj = hrtf_params["subj_number"]
    orig_sound = sound

    if type(sound) is not Sound:
        sound = sound.sound

    # --- Apply gating ---
    if hrtf_params.get("apply_gating"):
        logger.debug("[run_hrtf] Applying gating before HRTF...")
        sound = apply_gating(sound, ramp_ms=hrtf_params.get("ramp_ms"))

    samplerate = sound.samplerate
    original_duration = sound.duration

    if subj == "itd_only":
        logger.debug("[run_hrtf] Using ITD-only synthetic transformation.")
        sound = Sound.sequence(
            Sound.silence(5 * ms, sound.samplerate),
            sound,
        )
        hrtfset = HeadlessDatabase(13, azim_max=90).load_subject()
        binaural_sound: Sound = hrtfset(azim=angle)(sound)

    elif subj == "ild_only":
        logger.debug("[run_hrtf] Using ILD-only synthetic transformation.")
        binaural_sound: Sound = synthetic_ild(orig_sound, angle)

    else:
        logger.debug("[run_hrtf] Using IRCAM HRTF DB...")
        hrtfdb = IRCAM_LISTEN(Paths.IRCAM_DIR)
        hrtfset = hrtfdb.load_subject(hrtfdb.subjects[subj])
        hrtf = hrtfset(azim=ANGLE_TO_SOFA[angle], elev=0)
        binaural_sound: Sound = hrtf(sound)

    binaural_sound = binaural_sound.resized(math.floor(original_duration * samplerate))
    logger.debug("[run_hrtf] HRTF computation complete.")

    return binaural_sound, sound

def load_hrtf_subject(subj: int):


    # --- List and sort all .sofa files directly from the directory ---
    files = sorted(
        f for f in os.listdir(Paths.SOFA_DIR)
        if f.lower().endswith(".sofa")
    )

    if not files:
        raise FileNotFoundError(f"No .sofa files found in directory: {Paths.SOFA_DIR}")

    # --- Index validation ---
    if not (0 <= subj < len(files)):
        raise IndexError(
            f"subj={subj} is out of range: directory contains {len(files)} SOFA files."
        )

    sofa_path = files[subj]

    # --- Load the selected SOFA file ---
    hrtf = sofa.SOFAFile(str(Paths.SOFA_DIR + sofa_path), "r")  # SOFAFile expects a string path

    return hrtf

def apply_sofa_hrtf_to_sound(
    sofa_file,
    sound: Sound,
    azimuth_deg: float,
    elevation_deg: float = 0.0,
) -> Sound:
    """
    Apply the SOFA HRTF for the closest (azimuth, elevation) to a mono
    Brian2Hears Sound. Automatically resamples SOFA IRs to match the
    Sound's samplerate if needed.
    """

    # ---------- Extract IRs and positions ----------
    ir = sofa_file.getDataIR()                 # shape (M, R, N)
    positions = sofa_file.getSourcePositionValues()  # shape (M, C)

    # Spherical coords expected: az = positions[:,0], el = positions[:,1]
    az = positions[:, 0]
    el = positions[:, 1]

    # ---------- Find nearest measurement ----------
    diff_az = az - azimuth_deg
    diff_el = el - elevation_deg
    dist = np.sqrt(diff_az**2 + diff_el**2)
    m_idx = int(np.argmin(dist))

    left_ir = ir[m_idx, 0, :]
    right_ir = ir[m_idx, 1, :]

    # ---------- Sampling rate handling ----------
    # SOFA may store fs either as [I] or [M]; flatten and get the first value
    sofa_fs = float(np.array(sofa_file.getSamplingRate()).ravel()[0])
    sound_fs = float(sound.samplerate)

    # If different → resample IRs to match the sound samplerate
    if sound_fs != sofa_fs:
        # Compute new length after resampling
        N_old = left_ir.shape[0]
        N_new = int(round(N_old * (sound_fs / sofa_fs)))

        left_ir  = resample(left_ir,  N_new)
        right_ir = resample(right_ir, N_new)

    # ---------- Extract mono signal ----------
    x = np.asarray(sound).flatten()

    N = len(x)

    ir_len = max(len(left_ir), len(right_ir))

    # nmax = next power of 2 >= (signal + ir_length)
    nmax = max(N + ir_len, ir_len)
    nmax = 2**int(np.ceil(np.log2(nmax)))

    # Zero-pad HRIRs to nmax
    Lpad = np.hstack([left_ir,  np.zeros(nmax - len(left_ir))])
    Rpad = np.hstack([right_ir, np.zeros(nmax - len(right_ir))])
    xpad = np.hstack([np.zeros(ir_len), x, np.zeros(nmax - ir_len - N)])

    # ---------- FFT convolution ----------
    L_fft = rfft(Lpad, n=nmax)
    R_fft = rfft(Rpad, n=nmax)
    X_fft = rfft(xpad, n=nmax)

    L_out = irfft(L_fft * X_fft).real
    R_out = irfft(R_fft * X_fft).real

    L_final = L_out[ir_len:ir_len + N]
    R_final = R_out[ir_len:ir_len + N]

    # ---------- Return binaural Brian2Hears sound ----------
    stereo = np.column_stack([L_final, R_final])
    return Sound(stereo, sound.samplerate)

def run_hrtf(
    sound,               # could be Sound or Tone/ToneBurst wrapper
    angle: float,        # internal angle convention: +left, -right
    hrtf_params          # must contain {"sofa_file": ..., "apply_gating": ...}
):
    """
    Unified HRTF interface for SOFA-based HRTFs (e.g., ARI).
    Applies angle conversion, optional gating, and the SOFA HRIR convolution.
    """

    logger.debug(f"[run_hrtf] Starting HRTF for angle={angle} subj={hrtf_params['subj_number']}")

    subj = hrtf_params["subj_number"]
    hrtf = load_hrtf_subject(subj)

    if type(sound) is not Sound:
        sound = sound.sound

    # --- Apply gating ---
    if hrtf_params.get("apply_gating"):
        logger.debug("[run_hrtf] Applying gating before HRTF...")
        sound = apply_gating(sound, ramp_ms=hrtf_params.get("ramp_ms"))


    # -------- Apply SOFA HRTF --------
    binaural_sound = apply_sofa_hrtf_to_sound(
        sofa_file=hrtf,
        sound=sound,
        azimuth_deg=ANGLE_TO_SOFA[angle],
        elevation_deg=0.0,     # horizontal plane
    )

    return binaural_sound, sound

def load_hrir(subject_id: int, azimuth_deg: float):
    """
    Load left/right HRIR for a given subject and azimuth (elevation = 0°).

    Returns
    -------
    left_ir, right_ir : np.ndarray
    sofa_fs           : float
    """

    files = sorted(
        f for f in os.listdir(Paths.SOFA_DIR)
        if f.lower().endswith(".sofa")
    )

    sofa_path = os.path.join(Paths.SOFA_DIR, files[subject_id])
    sofa_file = sofa.SOFAFile(sofa_path, "r")

    ir = sofa_file.getDataIR()                     # (M, R, N)
    positions = sofa_file.getSourcePositionValues()  # (M, C)

    az = positions[:, 0]
    el = positions[:, 1]

    az_target = ANGLE_TO_SOFA[azimuth_deg]

    # Find closest measurement (elevation fixed to 0)
    dist = np.sqrt((az - az_target)**2 + (el - 0)**2)
    m_idx = int(np.argmin(dist))

    left_ir  = ir[m_idx, 0, :]
    right_ir = ir[m_idx, 1, :]

    sofa_fs = float(np.array(sofa_file.getSamplingRate()).ravel()[0])

    return left_ir, right_ir, sofa_fs

def plot_hrir(subject_id: int, azimuth_deg: float, target_fs=None):
    """
    Plot HRIR (left & right) in time domain.
    """

    left_ir, right_ir, sofa_fs = load_hrir(subject_id, azimuth_deg)

    # Optional resampling
    fs = sofa_fs
    if target_fs is not None and target_fs != sofa_fs:
        N_new = int(round(len(left_ir) * target_fs / sofa_fs))
        left_ir  = resample(left_ir,  N_new)
        right_ir = resample(right_ir, N_new)
        fs = target_fs

    times = np.arange(len(left_ir)) / fs * 1000  # ms

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, left_ir,  color='m', label='Left')
    ax.plot(times, right_ir, color='g', label='Right')

    ax.set_title(f"HRIR – Subject {subject_id}, Azimuth {azimuth_deg}°, Elevation 0°")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_hrtf_magnitude(subject_id: int, azimuth_deg: float, fs: float, figsize=(10, 4)):
    """
    Plot HRTF magnitude response (dB) for left and right ears.
    """

    left_ir, right_ir, sofa_fs = load_hrir(subject_id, azimuth_deg)

    # Resample HRIR if needed
    if sofa_fs != fs:
        N_new = int(round(len(left_ir) * fs / sofa_fs))
        left_ir  = resample(left_ir,  N_new)
        right_ir = resample(right_ir, N_new)

    # FFT padding (same logic as your notebook)
    ir_len = max(len(left_ir), len(right_ir))
    nmax = 2**int(np.ceil(np.log2(ir_len)))

    Lpad = np.hstack([left_ir,  np.zeros(nmax - len(left_ir))])
    Rpad = np.hstack([right_ir, np.zeros(nmax - len(right_ir))])

    L_fft = rfft(Lpad)
    R_fft = rfft(Rpad)

    freqs = np.fft.rfftfreq(nmax, d=1/fs)

    eps = 1e-12
    L_mag = 20 * np.log10(np.abs(L_fft) + eps)
    R_mag = 20 * np.log10(np.abs(R_fft) + eps)

    fig, ax = plt.subplots(figsize=figsize)

    ax.semilogx(freqs, L_mag, color='m', label="Left")
    ax.semilogx(freqs, R_mag, color='g', label="Right")

    for v in [200, 500, 1200, 4000, 16000]:
        ax.axvline(v,  linestyle='--', color='gray')

    ax.set_xlim(200, 20000)

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"HRTF Magnitude – Subject {subject_id}, Azimuth {azimuth_deg}°, Elevation 0°")

    ax.set_xscale('log')
    ax.set_xticks(np.array([0.2, 0.5, 1, 2, 5, 10, 20]) * 1000)
    ax.set_xticklabels(['0.2', '0.5', '1', '2', '5', '10', '20'])

    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

