import brian2 as b2
import brian2hears as b2h


DEFAULT_SOUND_DURATION = 25 * b2.ms
DEFAULT_SILENCE_DURATION = DEFAULT_BURST_SINGLE_DURATION = 25 * b2.ms
DEFAULT_BURST_REP = 3
DEFAULT_CLICKS_NUMBER = 10
DEFAULT_CLICKS_DURATION = 10
DEFAULT_CLICKS_INTERVAL = 50


# i considered subclassing for a bit but i don't know enough
class Tone:
    frequency: b2.Quantity
    sound: b2h.Sound

    def __init__(
        self, frequency: b2.Quantity, duration=DEFAULT_SOUND_DURATION,level=None, **kwargs
    ):
        self.frequency = frequency
        self.sound = b2h.Sound.tone(frequency, duration, **kwargs)
        if level is not None:
            self.sound.level = level


class ToneBurst:
    frequency: b2.Quantity
    sound: b2h.Sound
    burst_num: int

    def __init__(
        self,
        frequency: b2.Quantity,
        single_duration=DEFAULT_BURST_SINGLE_DURATION,
        burst_num=DEFAULT_BURST_REP,
        silence_duration=DEFAULT_SILENCE_DURATION,
        level=None,
        **kwargs,
    ):
        self.frequency = frequency
        self.burst_num = burst_num
        self.sound = b2h.Sound.sequence(
            [
                b2h.Sound.tone(frequency, single_duration, **kwargs),
                b2h.Sound.silence(silence_duration),
            ]
        ).repeat(burst_num)
        if level is not None:
            self.sound.level = level


class WhiteNoise:
    sound: b2h.Sound

    def __init__(self, duration=DEFAULT_SOUND_DURATION, level=None, **kwargs):
        self.sound = b2h.Sound.whitenoise(duration, **kwargs)
        if level is not None:
            self.sound.level = level

class HarmonicComplex:
    frequency: b2.Quantity
    sound: b2h.Sound

    def __init__(
        self, frequency: b2.Quantity, duration=DEFAULT_SOUND_DURATION, level=None, **kwargs
    ):
        self.frequency = frequency
        self.sound = b2h.Sound.harmoniccomplex(frequency, duration, **kwargs)
        if level is not None:
            self.sound.level = level

class Click:
    sound: b2h.Sound

    def __init__(self, duration=DEFAULT_SOUND_DURATION, click_duration = DEFAULT_CLICKS_DURATION, level=None, **kwargs):
        if level is not None:
            self.peak = level
        if isinstance(click_duration, b2.Quantity):
            c = b2h.click(click_duration, self.peak)
            i = b2h.silence(duration - click_duration)
        else:
            c = b2h.click(click_duration, self.peak)
            click_duration_ms = click_duration*(1/(c.samplerate / b2.Hz))*1000*b2.ms
            print(click_duration_ms)
            i = b2h.silence(duration - click_duration_ms)
            print(i)
        p = c + i
        self.sound = p


class Click_Train:
    sound: b2h.Sound

    def __init__(self, duration=DEFAULT_SOUND_DURATION, click_duration = DEFAULT_CLICKS_DURATION, interval = DEFAULT_CLICKS_INTERVAL, level=None, **kwargs):
        if level is not None:
            self.peak = level

        if isinstance(click_duration, b2.Quantity) and isinstance(interval, b2.Quantity):
            c = b2h.click(click_duration, self.peak)
            i = b2h.silence(interval)
            p = b2h.Sound.sequence(c,i)
            n = round(duration/(click_duration + interval))
            train = p.repeat(n)
        else:
            c = b2h.click(click_duration, self.peak)
            click_duration_ms = click_duration*(1/(c.samplerate / b2.Hz))*1000*b2.ms
            interval_ms = interval*(1/(c.samplerate / b2.Hz))*1000*b2.ms
            i = b2h.silence(interval_ms)
            p = b2h.Sound.sequence(c,i)
            n = round(int(duration/(click_duration_ms + interval_ms)))
            train = p.repeat(n)
    
        self.sound = train
        self.number = n



class ToneFromAngle(Tone):
    # x = ToneFromAngle(20, 20 * b2.Hz)
    angle: int

    def __init__(self, angle, *args):
        self.angle = angle
        super().__init__(*args)
