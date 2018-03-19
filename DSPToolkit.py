import math

class DSPWave(object):
    """A set of parameters and functions for
    a DSP Wave signal.
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self, a=1, f=1, c=0, d=0):
        """Initialize the wave with key components.
        
        Keyword Arguments:
            a {int} -- amplitude (default: {1})
            f {int} -- frequency in Hz (default: {1})
            c {int} -- phase shift (default: {0})
            d {int} -- vertical translation or bias (default: {0})
        """
        self.amplitude = a
        self.frequency = f
        self.phase = c          # c; left/right
        self.translation = d    # d; down/up
        self.two_pi = math.pi * 2.0 # only compute once!

        self.nyquist_rate = self.frequency * 2.0

    def generateWaveResult(self, t=0):
        """Generate f(t) result at the given t (time interval).

        Keyword Arguments:
            t {int} -- time interval (default: {0})
        """
        return self.amplitude * math.sin(self.two_pi * self.frequency * (t -    self.phase)) + self.translation

    def generateTimeSet(self, samples=64, time_interval=0.015625):
        """Generate a list of evenly-spaced time intervals
        for n samples.
        NOTE: These intervals correlate to the x-axis (t) on a graph/plot.
        
        Keyword Arguments:
            samples {int} -- n number of samples (default: {64})
            time_interval {float} -- the interval of time change per sample         (default: {0.015625})
        """
        # Ensure samples are high enough:
        if samples < 64:
            samples = 64 # could later change to self.nyquist_rate!

        time_set = []
        for n in range(1, samples + 1): # note the (1,n)
            time_set.append(n * time_interval)
        return time_set

    def generateSampleSet(self, samples=64, time_interval=0.015625):
        """Generate a list of Wave results, by computing
        wave results per evenly-spaced interval for n samples.
        NOTE: These results correlate to the y-axis on a graph/plot.

        Keyword Arguments:
            samples {int} -- n number of samples to capture (default: {64})
            time_interval {float} -- the interval of time change per sample         (default: {0.015625})
        """
        # Ensure samples are high enough:
        if samples < 64:
            samples = 64 # could later change to self.nyquist_rate!

        sample_set = []
        for n in range(1, samples + 1): # note the (1,n)
            sample_set.append(self.generateWaveResult(n * time_interval))
        return sample_set

# AUX FUNCS
def waveAddition(waveset1, waveset2):
    """Given two sample sets (lists) of DSPWave objects,
    add the wave results and compile a new sample set.
    
    Arguments:
        waveset1 {list} -- sample set of DSPWave 1
        waveset2 {list} -- sample set of DSPWave 2
    """
    complex_set = []
    waveset1len = len(waveset1)
    waveset2len = len(waveset2)

    # Choose the shorter sample set (if needed):
    if waveset1len < waveset2len:
        for s in range(waveset1len):
            complex_set.append(waveset1[s] + waveset2[s])
    elif waveset1len >= waveset2len:
        for s in range(waveset2len):
            complex_set.append(waveset1[s] + waveset2[s])
    return complex_set

def waveMultiplication(waveset1, waveset2):
    """Given two sample sets (lists) of DSPWave objects,
    multiply the wave results and compile a new sample set.
    
    Arguments:
        waveset1 {list} -- sample set of DSPWave 1
        waveset2 {list} -- sample set of DSPWave 2
    """
    complex_set = []
    waveset1len = len(waveset1)
    waveset2len = len(waveset2)

    # Choose the shorter sample set (if needed):
    if waveset1len < waveset2len:
        for s in range(waveset1len):
            complex_set.append(waveset1[s] * waveset2[s])
    elif waveset1len >= waveset2len:
        for s in range(waveset2len):
            complex_set.append(waveset1[s] * waveset2[s])
    return complex_set
