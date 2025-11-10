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

# ... (all your previous code unchanged) ...

def run_hrtf(sound: Union[Sound, Tone, ToneBurst], angle, hrtf_params) -> Sound:
    subj = hrtf_params["subj_number"]
    orig_sound = sound
    if type(sound) is not Sound:  # assume good faith, ok to fail otherwise
        sound = sound.sound
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
