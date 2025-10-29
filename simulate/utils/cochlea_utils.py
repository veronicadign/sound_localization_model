from brian2 import Hz, kHz
from dataclasses import dataclass
from brian2hears import Sound

NUM_CF = 3500  # 3500 (3.5k IHCs -> 10 ANF for each -> 35000 ANF)
# of course, if you change this, you need to regenerate all created files
NUM_ANF_PER_HC = 10
CFMIN = 20 * Hz
CFMAX = 20 * kHz
IRCAM_HRTF_ANGLES = [90, 75, 60, 45, 30, 15, 0, 345, 330, 315, 300, 285, 270]
ANGLES = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
ANGLE_TO_IRCAM = {i: j for i, j in zip(ANGLES, IRCAM_HRTF_ANGLES)}


class ITD_REMOVAL_STRAT:
    COMPUTED = "computed"
    ESTIMATE_FROM_HRTF = "estimate_from_hrtf"


@dataclass
class AnfResponse:
    binaural_anf_spiketrain: dict
    left: Sound
    right: Sound


@dataclass
class SoundFromAngles:
    basesound: Sound
    angle_to_response: dict[str, AnfResponse]

