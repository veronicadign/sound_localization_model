from brian2 import Hz, kHz, ms
from brian2hears import Sound
from joblib.memory import MemorizedFunc
from sorcery import dict_of
from utils.custom_sounds import Click, Tone, ToneBurst, WhiteNoise, Clicks, HarmonicComplex
from utils.log_utils import logger

from cochleas.DCGC import COCHLEA_KEY as DCGC_COC_KEY
from cochleas.DCGC import sound_to_spikes as DCGC_cochlea
from cochleas.GammatoneCochlea import COCHLEA_KEY as GAMMATONE_COC_KEY
from cochleas.GammatoneCochlea import sound_to_spikes as gammatone_cochlea
from cochleas.PpgCochlea import COCHLEA_KEY as PPG_COC_KEY
from cochleas.PpgCochlea import tone_to_ppg_spikes as ppg_cochlea
from cochleas.TanCarneyCochlea import COCHLEA_KEY as TC_COC_KEY
from cochleas.TanCarneyCochlea import sound_to_spikes as tc_cochlea

from utils.cochlea_utils import ANGLES, IRCAM_HRTF_ANGLES, NUM_ANF_PER_HC, NUM_CF, AnfResponse
import nest


SOUND_FREQUENCIES = [100 * Hz, 1 * kHz, 10 * kHz]
INFO_FILE_NAME = "info.txt"
INFO_HEADER = "this directory holds all computed angles, for a specific sound, with a specific cochlear backend. the pickled sound is also available. for cochleas that do not use HRTF, left and right sounds are the same. \n"
COCHLEAS = {
    GAMMATONE_COC_KEY: gammatone_cochlea,
    PPG_COC_KEY: ppg_cochlea,
    TC_COC_KEY: tc_cochlea,
    DCGC_COC_KEY: DCGC_cochlea,
}


def create_sound_key(sound):
    add_info = None
    if type(sound) is Tone:
        add_info = str(sound.frequency).replace(" ", "")
        sound_type = "tone"
        level = int(sound.sound.level)
    elif type(sound) is ToneBurst:
        add_info = str(sound.frequency).replace(" ", "")
        sound_type = "toneburst"
        level = int(sound.sound.level)
    elif type(sound) is WhiteNoise:
        sound_type = "whitenoise"
        level = int(sound.sound.level)
    elif type(sound) is Click:
        sound_type = "click"
        if sound.peak is not None:
            level = sound.peak
        else:
            level = "XX"
    elif type(sound) is Clicks:
        sound_type = "clicks"
        add_info = str(sound.number).replace(" ", "")
        if sound.peak is not None:
            level = sound.peak
        else:
            level = "XX"
    elif type(sound) is HarmonicComplex:
        sound_type = "harmonic"
        if sound.sound.level is not None:
            level = int(sound.sound.level)
        else:
            level = "XX"
    else:
        raise NotImplementedError(f"sound {sound} is not a Tone")
    return f"{sound_type}{f"_{add_info}" if add_info else ""}_{level}dB"



def load_anf_response(
    sound: Tone | Sound | ToneBurst | Click | Clicks | WhiteNoise,
    angle: int,
    cochlea_key: str,
    params: dict,
    ignore_cache=False,
):
    cochlea_func: MemorizedFunc = COCHLEAS[cochlea_key]
    params = params[cochlea_key]
    if not cochlea_func.check_call_in_cache(sound, angle, params):
        logger.info(f"saved ANF not found. generation will take some time...")
    if ignore_cache:
        logger.info(f"ignoring cache. generation will take some time...")
    logger.info(
        f"generating ANF for {
        dict_of(sound,angle,cochlea_key,params)}"
    )
    if ignore_cache:
        cochlea_func = cochlea_func.call  # forces execution
    try:
        anf = cochlea_func(sound, angle, params)
    except TypeError as e:
        if "unexpected" in e.args[0]:
            logger.error(f"{e}, please check the signature of cochlea")
        raise e

    return anf

def spikes_to_nestgen(anf_response: AnfResponse):
    nest.set_verbosity("M_ERROR")
    anfs_per_ear = {}
    for channel, response_ANF in anf_response.binaural_anf_spiketrain.items():
        s = []
        # make sure all ANFs have a spiking array and no zero-time spikes are present
        for i in range(NUM_CF * NUM_ANF_PER_HC):
            spiketimes = response_ANF.get(i, [] * ms)
            s.append(spiketimes[spiketimes != 0] / ms)

        anfs = nest.Create(
            "spike_generator",
            NUM_CF * NUM_ANF_PER_HC,
            params={
                "spike_times": s,
                # "spike_times": [i[i != 0] / ms for i in response_ANF.values()],
                "allow_offgrid_times": [True] * NUM_CF * NUM_ANF_PER_HC,
            },
        )
        anfs_per_ear[channel] = anfs

    return anfs_per_ear
