from __future__ import division, print_function, absolute_import
import sys
from pathlib import Path
from typing import Union 


repo_root = Path(__file__).resolve().parents[3]            # top-level directory (where cochlea-1 lives)
cochlea_local_path = repo_root / "cochlea-1"

# ---------------------------------------------------------------------
# 2. Add local cochlea path (so it overrides installed version)
# ---------------------------------------------------------------------
sys.path.insert(0, str(cochlea_local_path))
print(f"✅ Using local cochlea package from: {cochlea_local_path}")

import cochlea  # import it now while path is active

simulate_dir = Path(__file__).resolve().parents[1]
print(simulate_dir)
sys.path.insert(0, str(simulate_dir))
print(f"✅ Using utils from: {simulate_dir}") 

from os import makedirs
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from joblib import Memory

# --- External model libraries ---
import thorns as th

# --- Brian2Hears sound objects still used for convenience ---
from brian2 import Hz, kHz, seed
from brian2hears import Sound, dB, erbspace

# --- Project utilities ---
from utils.path_utils import Paths
from utils.custom_sounds import Tone, ToneBurst
from utils.log_utils import logger, tqdm
from utils.cochlea_utils import CFMAX, CFMIN, NUM_CF, AnfResponse
from utils.hrtf_utils import run_hrtf

# --- Setup ---
COCHLEA_KEY = f"Zilany"
CACHE_DIR = Paths.ANF_SPIKES_DIR + COCHLEA_KEY + "/"
makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

# =============================================================================
# Utility Functions
# =============================================================================

def resample_sound(sound: Sound, original_fs, target_fs=50000):
    """Resample a single-channel Sound object to target_fs."""
    ratio = target_fs / original_fs
    sound_data = np.array(sound)
    new_length = int(len(sound_data) * ratio)
    resampled_data = signal.resample(sound_data, new_length)
    return Sound(resampled_data, samplerate=target_fs * Hz)


def resample_binaural_sound(binaural_sound: Sound):
    """Resample both channels of a binaural Sound to 100 kHz for Zilany model."""
    original_fs = float(binaural_sound.samplerate / Hz)
    left_resampled = resample_sound(binaural_sound.left, original_fs, target_fs=100000)
    right_resampled = resample_sound(binaural_sound.right, original_fs, target_fs=100000)
    return Sound((left_resampled, right_resampled), samplerate=100 * kHz)


# =============================================================================
# Main Model Function
# =============================================================================

@memory.cache
def sound_to_spikes(
    sound: Union[Sound, Tone, ToneBurst],  # <-- ✅ FIXED HERE
    angle,
    params: dict,
    plot_spikes=False
) -> AnfResponse:
    """
    Generate auditory nerve spike trains using the Zilany et al. (2014) cochlea model.
    Replaces the TanCarney + ZhangSynapse Brian2Hears model.
    """

    hrtf_params = params["hrtf_params"]
    rng_seed = params["rng_seed"]
    noise_level = params["omni_noise_level"] * dB
    coch_par = params.get("cochlea_params", {})
    seed(rng_seed)

    logger.debug(f"Generating spikes for {sound=} {angle=} {plot_spikes=} {hrtf_params=}")

    # --- 1. HRTF & Noise ---
    binaural = run_hrtf(sound, angle, hrtf_params)
    logger.debug(f"Binaural sound post-HRTF level={binaural.level}")
    noise = Sound.whitenoise(binaural.duration).atlevel(noise_level)
    binaural_sound = resample_binaural_sound(binaural + noise)
    logger.debug(f"Binaural sound post noise level={binaural.level}")

    # --- 2. Prepare input signals ---
    left_data = np.asarray(binaural_sound.left)[:,0]
    right_data = np.asarray(binaural_sound.right)[:,0]
    fs = float(binaural_sound.samplerate / Hz)


    cf = (CFMIN/Hz, CFMAX/Hz, NUM_CF)
    binaural_IHC_response = {}
    logger.info("Generating simulated ANF spikes using Zilany model...")

    # --- 3. Run Zilany model for each ear ---
    for channel, data in zip(["L", "R"], [left_data, right_data]):
        logger.debug(f"Running Zilany model for {channel} ear...")

        anf = cochlea.run_zilany2014(
        data,
        fs,
        cf=cf,
        seed=rng_seed,
        **coch_par
        )

        # Collect spikes
        binaural_IHC_response[channel] = anf.spikes

        # Optional neurogram plot
        if plot_spikes:
            anf_acc = th.accumulate(anf, keep=['cf', 'duration'])
            anf_acc.sort_values('cf', ascending=False, inplace=True)
            plt.figure(figsize=(7, 3))
            th.plot_neurogram(anf_acc, fs)
            plt.title(f"Zilany2014 ANF Neurogram - {channel} ear")
            plt.xlabel("Time (s)")
            plt.ylabel("CF (Hz)")
            plt.tight_layout()
            plt.show()

    logger.info("Zilany spike generation complete.")

    # --- 4. Return wrapped result ---
    return AnfResponse(binaural_IHC_response, binaural_sound.left, binaural_sound.right)
