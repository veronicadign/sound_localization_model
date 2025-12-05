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
from .cochlea_utils import ANGLE_TO_IRCAM, ANGLE_TO_ARI, ITD_REMOVAL_STRAT
import os
from scipy.signal import fftconvolve, resample
import pysofaconventions as sofa


def apply_gating(sound: Sound, ramp_ms: float = 5.0) -> Sound:
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

def run_hrtf(sound: Union[Sound, Tone, ToneBurst], angle, hrtf_params) -> tuple[Sound, Sound]:
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
        hrtf = hrtfset(azim=ANGLE_TO_IRCAM[angle], elev=0)
        binaural_sound: Sound = hrtf(sound)

    binaural_sound = binaural_sound.resized(math.floor(original_duration * samplerate))
    logger.debug("[run_hrtf] HRTF computation complete.")

    return binaural_sound, sound

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
    x = np.asarray(sound)
    if x.ndim == 2 and x.shape[1] == 1:
        mono = x[:, 0]
    elif x.ndim == 1:
        mono = x
    else:
        raise ValueError("apply_sofa_hrtf_to_sound expects a mono Sound input.")

    # ---------- Convolution ----------
    left_conv  = fftconvolve(mono, left_ir,  mode="full")
    right_conv = fftconvolve(mono, right_ir, mode="full")

    # ---------- Return binaural Brian2Hears sound ----------
    stereo = np.column_stack([left_conv, right_conv])
    binaural_sound = Sound(stereo, sound.samplerate)

    return binaural_sound

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

def run_hrtf_sofa(
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
    orig_sound = sound
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
        azimuth_deg=ANGLE_TO_ARI[angle],
        elevation_deg=0.0,     # horizontal plane
    )

    # -------- Resize to match original duration if needed --------
    target_len = math.floor(sound.duration * float(sound.samplerate))
    if binaural_sound.nsamples > target_len:
        binaural_sound = binaural_sound[:target_len]

    return binaural_sound, sound


