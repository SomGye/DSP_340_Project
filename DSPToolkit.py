import math
import numpy as np

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

    def generateSineWaveResult(self, t=0):
        """Generate f(t) result at the given t (time interval),
        using sine wave form:
        f(t) = a*sin(2pi*f(t-c)) + d

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

    def generateSineSampleSet(self, samples=64, time_interval=0.015625):
        """Generate a list of Sine Wave results, by computing
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
            sample_set.append(self.generateSineWaveResult(n * time_interval))
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

# Fourier stuff
# LINK: https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])

def FFT_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)
    
    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] / 2]
        X_odd = X[:, X.shape[1] / 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel() # return flattened, contiguous 1D array of elems
    