import math
import cmath # for complex nums
import numpy as np

## Author: Maxwell Crawford

# GLOBALS
TWO_PI = math.pi * 2.0 # only compute once
ONE_PI = math.pi
scale = 32767 # 16-bit unsigned short
Fs = 44100 # sampling rate
Sn = 4096 # n-samples per file
keys = '1', '2', '3', 'A', \
	'4', '5', '6', 'B', \
	'7', '8', '9', 'C', \
	'*', '0', '#', 'D' 			# Phone Keypad
F1 = [697, 770, 852, 941] 		# Left, Rows
F2 = [1209, 1336, 1477, 1633] 	# Top, Cols

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
        self.two_pi = TWO_PI
        self.nyquist_rate = self.frequency * 2.0

    def generateSineWaveResult(self, t=0):
        """Generate f(t) result at the given t (time interval),
        using sine wave form:
        f(t) = a*sin(2pi*f(t-c)) + d

        Keyword Arguments:
            t {int} -- time interval (default: {0})
        """
        return self.amplitude * math.sin(self.two_pi * self.frequency * (t -    self.phase)) + self.translation
    
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
        for n in range(samples):
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
        for n in range(samples):
            sample_set.append(self.generateSineWaveResult(n * time_interval))
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
        for n in range(samples):
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

# Fourier stuff...
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
    even = fft_cooley(data[0::2], direction)
    odd = fft_cooley(data[1::2], direction)
    # Get e^():
    ft_exponent = [cmath.exp(direction * -2j*cmath.pi*k/N)*odd[k] for k in range(N//2)]
    # Return halves:
    return [even[k] + ft_exponent[k] for k in range(N//2)] + \
           [even[k] - ft_exponent[k] for k in range(N//2)]

def fft_cooley_inv(data):
    """Uses the existing FFT Cooley-Tukey function
    to apply the inverse to its output.
    
    Returns:
        list -- inverse FT data
    """

    N = len(data)
    fft_result = fft_cooley(data)
    return [x/N for x in fft_result] # data / length

def two_dim_fft2(data, inverse=False):
    """Get a 2D FFT of a set of data.
    
    Arguments:
        data {list} -- Given 2D data for iteration
    
    Keyword Arguments:
        inverse {bool} -- False for regular, True for inverse (default: {False})
    
    Returns:
        list -- A 2D list of FT values
    """
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

    # Step 1 Inner Loop (Orig X -> Y):
    # Column Based!
    temp_col = np.zeros((len(array_X), 1, 1), complex) # reset
    for c in range(len(array_X)):
        temp_col = array_X[:, c] # get all entries from column c
        if not inverse:
            temp_fft = fft_cooley(temp_col)
        else:
            temp_fft = fft_cooley_inv(temp_col)
        # Map new FFT data to column in Y
        for r in range(len(array_Y)):
            array_Y[r][c] = temp_fft[r] # map r,c to r of fft
    # Step 2 Inner Loop (Y -> Z)
    # Row Based!
    temp_row = np.zeros((len(array_Y[0]), 1, 1), complex) # reset
    for r in range(len(array_Y)):
        temp_row = array_Y[r, :] # get all entries from row r
        if not inverse:
            temp_fft = fft_cooley(temp_row)
        else:
            temp_fft = fft_cooley_inv(temp_row)
        # Map new FFT data to row in Z
        for c in range(len(array_Z[0])):
            array_Z[r][c] = temp_fft[c] # map r,c to c of fft
    # Return final FFT array, with extra dimension squeezed out:
    return np.squeeze(array_Z)

def psd(data, is_list=True, need_FT=False):
    """Gets the Power Spectral Density (Energy) of
    a Fourier Transform.
    Assumes a list of data by default, but can
    also work with singular values.
    
    Arguments:
        data {list or float} -- A list or single value, as float.
    
    Keyword Arguments:
        is_list {bool} -- Whether the data is a list or single value (default: {True})
        need_FT {bool} -- Whether the FT is needed first (default: {False})
    """
    # Get FT first if needed:
    ft_data = []
    if need_FT:
        ft_data = fft_cooley(data)
    else:
        ft_data = data[:] # slice copy
    # Get Pulse Spectral Density:
    N = 0
    if is_list:
        N = len(ft_data)
        psd_result = []
        temp_psd = 0+0j
        for n in range(N):
            temp_psd = complex(abs(ft_data[n]) * abs(ft_data[n]))
            psd_result.append(temp_psd)
        return psd_result
    else: # scalar
        return complex(abs(ft_data) * abs(ft_data))

# DTMF stuff
def read_dtmf_data(data_filename):
    """Get integer data points from simple text file.

	Arguments:
		data_filename {string} -- text file location

	Returns:
		list -- 1D list of data points as int
	"""

    olddata = open(data_filename, 'r')
    newdata = []
    for line in olddata:
        if line != '' and line != '\n':
            newdata.append(int(line))
    olddata.close()
    return newdata

def dtmf_decoder(data):
    """Decode a Dual-Tone Multi-Frequency signal,
	given a list of signal data points, and return
	the appropriate keypad entry.

	Arguments:
		data {list} -- A list of signal data points

	Returns:
		list -- Returns a list of decoded key, frequencies found, and the orig. frequency set.
	"""
    freqsets = []
	# Loop thru left/top frequency table:
    for f1 in F1:
        for f2 in F2:
            diff = 0
			# Loop thru given data samples:
            for i in range(Sn): # no phase shift!
                p = i*1.0/Sn
                S = scale+scale*(math.sin(p*f1*TWO_PI)+math.sin(p*f2*TWO_PI)) / 2.0
                diff += abs(S-data[i]) # reference point for distortion
            freqsets.append((diff, f1, f2))
	# Get freq. set with minimum 'distortion':
    f1, f2 = min(freqsets)[1:] # determine minimal distortion, keep only freqs!
	# Cross-reference Frequency Tables:
    i, j = F1.index(f1), F2.index(f2)
    freq1, freq2 = F1[i], F2[j]
	# Decode with Key Formula (4 * i * j):
    decoded_key = keys[4*i+j]
	# Return result set:
    return [decoded_key, freq1, freq2, freqsets]

def get_switch_data(sign_data):
    """Get the frequency sign change "switch"
    data for a series of sign bins.
    
    Arguments:
        sign_data {list} -- A list of negative/positive signs from a set of Fourier transform data.
    
    Returns:
        list -- Switch data list.
    """

    switch = [0] # init 1st elem!
    for x in range(0, len(sign_data) - 1):
        # Detect change in sign:
        if sign_data[x] < sign_data[x + 1]:
            switch.append(1)
        else: # no change
            switch.append(0)
    return switch
