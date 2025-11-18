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
from .cochlea_utils import ANGLE_TO_IRCAM, ITD_REMOVAL_STRAT


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

    return Sound((left, right), samplerate=sound.sound.samplerate)


# def synthetic_ild(sound: Tone, angle: int):
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




def run_hrtf(sound: Union[Sound, Tone, ToneBurst], angle, hrtf_params) -> Sound:
    subj = hrtf_params["subj_number"]
    orig_sound = sound
    if type(sound) is not Sound:  # assume good faith, ok to fail otherwise
        sound = sound.sound
    #sound = apply_gating(sound, ramp_ms=hrtf_params.get("ramp_ms", 5.0))
    samplerate = sound.samplerate
    original_duration = sound.duration
    if subj == "itd_only":
        # delay sound to mimic time to reach ear (conservative approximation,
        # it changes among HRTFs)
        sound = Sound.sequence(
            Sound.silence(5 * ms, sound.samplerate),
            sound,
        )
        hrtfset = HeadlessDatabase(13, azim_max=90).load_subject()
        hrtf = hrtfset(azim=angle)
        binaural_sound: Sound = hrtf(sound)
    elif subj == "ild_only":
        binaural_sound: Sound = synthetic_ild(orig_sound, angle)
    else:
        hrtfdb = IRCAM_LISTEN(Paths.IRCAM_DIR)
        hrtfset = hrtfdb.load_subject(hrtfdb.subjects[subj])
        hrtf = hrtfset(azim=ANGLE_TO_IRCAM[angle], elev=0)
        binaural_sound: Sound = hrtf(sound)

    binaural_sound = binaural_sound.resized(math.floor(original_duration * samplerate))
    return binaural_sound
