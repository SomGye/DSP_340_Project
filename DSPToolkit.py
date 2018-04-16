import math
import cmath # for complex nums
from copy import deepcopy
import numpy as np


# GLOBALS
TWO_PI = math.pi * 2.0 # only compute once
ONE_PI = math.pi

class DSPWave(object):
    """A set of parameters and functions for
    a DSP Wave signal.

    NOTE: The default signal is a sine wave.

    Arguments:
        a {int} -- amplitude (default: {1})
        f {int} -- frequency in Hz (default: {1})
        c {int} -- phase shift (default: {0})
        d {int} -- vertical translation or bias (default: {0})
    """
    def __init__(self, a=1, f=1, c=0, d=0):
        """Initialize the wave with key components.
        
        Keyword Arguments:
            a {int} -- amplitude (default: {1})
            f {int} -- frequency in Hz (default: {1})
            c {int} -- phase shift (default: {0})
            d {int} -- vertical translation or bias (default: {0})
        """
        # Set of wave properties
        self.amplitude = a
        self.frequency = f
        self.phase = c          # c; left/right
        self.translation = d    # d; down/up
        
        # Set of useful calcs
        # self.two_pi = math.pi * 2.0 # only compute once!
        self.two_pi = TWO_PI
        self.nyquist_rate = self.frequency * 2.0

        # TEST: new properties...
        # self.B = self.two_pi / f
        # self.C = -1.0 * c / self.B

    def generateSineWaveResult(self, t=0):
        """Generate f(t) result at the given t (time interval),
        using sine wave form:
        f(t) = a*sin(2pi*f(t-c)) + d

        Keyword Arguments:
            t {int} -- time interval (default: {0})
        """
        return self.amplitude * math.sin(self.two_pi * self.frequency * (t -    self.phase)) + self.translation

    # def generateSineWaveResult2(self, t=0):
    #     """Generate f(t) result at the given t (time interval),
    #     using sine wave form:
    #     f(t) = y = A sin(Bx + C) + D,
    #     where Amplitude: A, Frequency: 2pi/B, Phase: -C/B, Vert. Shift: D

    #     Keyword Arguments:
    #         t {int} -- time interval (default: {0})
    #     """
    #     return self.amplitude * math.sin(self.B * t + self.C) + self.translation
    
    def generateCosineWaveResult(self, t=0):
        """Generate f(t) result at the given t (time interval),
        using cosine wave form:
        f(t) = a*cos(2pi*f(t-c)) + d

        Keyword Arguments:
            t {int} -- time interval (default: {0})
        """
        return self.amplitude * math.cos(self.two_pi * self.frequency * (t -    self.phase)) + self.translation

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
        # for n in range(1, samples + 1): # note the (1,n)
        #     time_set.append(n * time_interval)
        for n in range(samples): # WAS (1, samples+1)
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
        for n in range(samples): # WAS (1,samples+1)
            sample_set.append(self.generateSineWaveResult(n * time_interval))
            # sample_set.append(self.generateSineWaveResult2(n * time_interval))
        return sample_set
    
    def generateCosineSampleSet(self, samples=64, time_interval=0.015625):
        """Generate a list of Cosine Wave results, by computing
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
        for n in range(samples): # WAS (1,samples+1)
            sample_set.append(self.generateCosineWaveResult(n * time_interval))
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

# LINK: https://rosettacode.org/wiki/Fast_Fourier_transform#Python
def fft_rosetta(x):
    N = len(x)
    if N <= 1: return x
    even = fft_rosetta(x[0::2])
    odd = fft_rosetta(x[1::2])
    T = [cmath.exp(-2j*cmath.pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]

def fft_cooley(data, direction=1):
    """Get the Fast Fourier Transform of list data,
    using a recursive form of the Cooley-Tukey Algorithm.
        NOTE: This is for one-dimensional data, with length
        equal to to a Power of Two, only!
    Arguments:
        data {list} -- A 1D list of y-data, generally from a signal/wave
    
    Keyword Arguments:
        direction {int} -- 1 for regular, -1 for inverse (default: {1})
    
    Returns:
        list -- A list of new FT data
    """
    # Check if more than 1D:
    if isinstance(data[0], list):
        raise TypeError("Data must be a 1D list!")
    # Get len and base case:
    N = len(data)
    if N <= 1:      # base case
        return data
    # Check for power of 2:
    if ((N & (N - 1)) != 0):
        raise ValueError("Length of Data MUST be a Power of 2!") # ERROR
    # Split up:
    even = fft_cooley(data[0::2])
    odd = fft_cooley(data[1::2])
    # Get e^():
    ft_exponent = [cmath.exp(direction * -2j*cmath.pi*k/N)*odd[k] for k in range(N//2)]
    # Return halves:
    return [even[k] + ft_exponent[k] for k in range(N//2)] + \
           [even[k] - ft_exponent[k] for k in range(N//2)]

def two_dim_fft(data, direction=1):
    """Get a 2D FFT of a set of data.
    
    Arguments:
        data {list} -- Given 2D data for iteration
    
    Keyword Arguments:
        direction {int} -- 1 for regular, -1 for inverse (default: {1})
    
    Returns:
        list -- A 2D list of FT values
    """
    # LINK: https://stackoverflow.com/questions/4455076/how-to-access-the-ith-column-of-a-numpy-multidimensional-array

    lenx = len(data)
    leny = len(data[0])
    # Pre-emptive check to ensure powers of 2 length/width!
    if (((lenx & (lenx - 1)) != 0) or lenx == 0) or \
         (((leny & (leny - 1)) != 0) or leny == 0):
        return False # ERROR

    # Convert to complex arrays:
    array_X = np.asarray(data, complex) # ORIG data as complex array
    array_Y = np.zeros((len(data), len(data[0]), 1), complex) # col FFT
    array_Z = np.zeros((len(data), len(data[0]), 1), complex) # col + row FFT

    # Outer loop
    for d in range(len(array_X)): # outer rows
        # Step 1 Inner Loop (Orig X -> Y): 
        # Column Based!
        temp_col = np.zeros((len(array_X), 1, 1), complex) # reset
        for c in range(len(array_X)):
            temp_col = array_X[:,c] # get all entries from column c
            temp_fft = fft_cooley(temp_col, direction)
            # Map new FFT data to column in Y
            for r in range(len(array_Y)):
                array_Y[r][c] = temp_fft[r] # map r,c to r of fft
        # Step 2 Inner Loop (Y -> Z)
        # Row Based!
        temp_row = np.zeros((len(array_Y[0]), 1, 1), complex) # reset
        for r in range(len(array_Y)):
            temp_row = array_Y[r,:] # get all entries from row r
            temp_fft = fft_cooley(temp_row, direction)
            # Map new FFT data to row in Z
            for c in range(len(array_Z[0])):
                array_Z[r][c] = temp_fft[c] # map r,c to c of fft
    # return array_Z #ORIG
    # Return final FFT array, with extra dimension squeezed out:
    return np.squeeze(array_Z)

def psd(ft_data, is_list=True):
    """Gets the Power Spectral Density (Energy) of
    a Fourier Transform.
    Assumes a list of data by default, but can
    also work with singular values.
    
    Arguments:
        ft_data {list or float} -- A list or single value, as float.
    
    Keyword Arguments:
        list {bool} -- Whether the data is a list or single value (default: {True})
    """
    N = 0
    if is_list:
        N = len(ft_data)
        psd_result = []
        temp_psd = 0+0j
        for n in range(N):
            temp_psd = complex(abs(ft_data[n]) * abs(ft_data[n]))
            psd_result.append(temp_psd)
        return psd_result
    else:
        return complex(abs(ft_data) * abs(ft_data))

#OLD...
# def FFTCooleyTukey(data, N=64, direction=1):
#     """Performs a recursive Fast Fourier Transform
#     using the Cooley-Tukey method, on a list of doubles.
#     Number of samples must be a power of two, and the
#     direction can be positive (regular) or negative (inverse).
    
#     Arguments:
#         data {list} -- a 1D list of doubles/floats/integers
    
#     Keyword Arguments:
#         N {int} -- Number of samples taken (default: {64})
#         direction {int} -- The direction of the transform; 
#         regular or inverse (default: {1})
#     """
#     theta = (-1*TWO_PI*direction)/N
#     r = N/2
#     j = complex(0, 1) # eq. to cmath.sqrt(-1)

#     # Loop 1: calculate Discrete Fourier Transform:
#     # EMULATE DO WHILE LOOP
#     i = 1
#     while i < (N-1):
#         w = math.cos(i*theta) + j * math.sin(i*theta) # w becomes complex!
#         k = 0
#         while k < (N-1):
#             u = 1
#             for m in range(r-1):
#                 t = FFTCooleyTukey(data, k+m) - FFTCooleyTukey(data, k+m+r)
