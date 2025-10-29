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
        self, frequency: b2.Quantity, duration=DEFAULT_SOUND_DURATION, **kwargs
    ):
        self.frequency = frequency
        self.sound = b2h.Sound.tone(frequency, duration, **kwargs)


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
        self, frequency: b2.Quantity, duration=DEFAULT_SOUND_DURATION, **kwargs
    ):
        self.frequency = frequency
        self.sound = b2h.Sound.harmoniccomplex(frequency, duration, **kwargs)

class Click:
    sound: b2h.Sound

    def __init__(self, duration=DEFAULT_SOUND_DURATION, level=None, **kwargs):
        self.peak = level
        self.sound = b2h.click(duration, self.peak)


class Clicks:
    sound: b2h.Sound

    def __init__(self, duration=DEFAULT_SOUND_DURATION, click_duration = DEFAULT_CLICKS_DURATION, interval = DEFAULT_CLICKS_INTERVAL, level=None, **kwargs):

        self.peak = level

        if interval < click_duration:
            interval = click_duration
            print(f"Warning: Interval adjusted to click_duration ({click_duration} seconds) to prevent overlapping clicks.")
        
        c = b2h.click(click_duration, self.peak)
        i = b2h.silence(interval)
        p = c + i
        n = int(duration/interval)
        train = p.repeat(n)

        if level is not None:
            self.sound.level = level
            
        self.sound = train
        self.number = n



class ToneFromAngle(Tone):
    # x = ToneFromAngle(20, 20 * b2.Hz)
    angle: int

    def __init__(self, angle, *args):
        self.angle = angle
        super().__init__(*args)
