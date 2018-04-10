import matplotlib.pyplot as plt # for graphing
import cmath                    # for complex number funcs
from math import sin, pi
import DSPToolkit as DT
import numpy as np # TEST

def main():
    # Create wave objects
    wave1 = DT.DSPWave() # default settings
    wave2 = DT.DSPWave(1.2, 2, 0.1, 0.1)
    coswave = DT.DSPWave(1, 1, pi / 2.0, 0) # only phase shift!
    # coswave = DT.DSPWave() #TEST

    # Get wave x/y axes:
    w1x = wave1.generateTimeSet()
    w1y = wave1.generateSineSampleSet()
    w2x = wave2.generateTimeSet()
    w2y = wave2.generateSineSampleSet()
    cosx = coswave.generateTimeSet()
    cosy = coswave.generateSineSampleSet() # was sine...
    # cosy = coswave.generateCosineSampleSet()

    # Get complex waveforms:
    cwave1 = DT.waveAddition(w1y, w2y)
    cwave2 = DT.waveMultiplication(w1y, w2y)

    # Plot wave 1:
    print("\n\t * Sine Wave 1")
    plt.title('Sine Wave 1')
    plt.plot(w1x, w1y, 'b-') # blue line
    xmax = max(w1x) * 1.5
    xmin = min(w1x)
    if xmin < 0:
        xmin *= 1.5
    else:
        xmin *= -1.5
    ymax = max(w1y) * 1.5
    ymin = min(w1y)
    if ymin < 0:
        ymin *= 1.5
    else:
        ymin *= -1.5
    axis_list = [xmin, xmax, ymin, ymax]
    plt.axis(axis_list)
    plt.show()

    # Plot wave 2:
    print("\n\t * Sine Wave 2")
    plt.title('Sine Wave 2')
    plt.plot(w2x, w2y, 'r-') # red line
    xmax = max(w2x) * 1.5
    xmin = min(w2x)
    if xmin < 0:
        xmin *= 1.5
    else:
        xmin *= -1.5
    ymax = max(w2y) * 1.5
    ymin = min(w2y)
    if ymin < 0:
        ymin *= 1.5
    else:
        ymin *= -1.5
    axis_list = [xmin, xmax, ymin, ymax]
    plt.axis(axis_list)
    plt.show()

    # Plot cosine wave:
    print("\n\t * Cosine Wave")
    plt.title('Cosine Wave')
    plt.plot(cosx, cosy, 'g-') # green line
    xmax = max(cosx) * 1.5
    xmin = min(cosx)
    if xmin < 0:
        xmin *= 1.5
    else:
        xmin *= -1.5
    ymax = max(cosy) * 1.5
    ymin = min(cosy)
    if ymin < 0:
        ymin *= 1.5
    else:
        ymin *= -1.5
    axis_list = [xmin, xmax, ymin, ymax]
    plt.axis(axis_list)
    plt.show()

    # Plot wave 1 and wave 2:
    print("\n\t * Sine Waves 1 and 2")
    plt.title('Sine Waves 1 and 2')
    plt.plot(w1x, w1y, 'b-', w2x, w2y, 'r-')
    plt.show()

    # Plot complex wave 1 (add):
    print("\n\t * Complex Wave 1 - Addition")
    plt.title('Complex Wave 1 - Addition')
    if len(w1x) < len(w2x):
        plt.plot(w1x, cwave1, 'bs')
        plt.show()
    else:
        plt.plot(w2x, cwave1, 'bs')
        plt.show()

    # Plot complex wave 2 (mult):
    print("\n\t * Complex Wave 2 - Multiplication")
    plt.title('Complex Wave 2 - Multiplication')
    if len(w1x) < len(w2x):
        plt.plot(w1x, cwave2, 'bs')
        plt.show()
    else:
        plt.plot(w2x, cwave2, 'bs')
        plt.show()

    #--
    # FFT TEST
    fft_data = [26160.0, 19011.0, 18757.0, 18405.0, 17888.0, 14720.0, 14285.0,17018.0, 18014.0, 17119.0, 16400.0, 17497.0, 17846.0, 15700.0, 17636.0, 17181.0] #NEW
    print("\n\t * FFT TEST")
    fft_cwave1 = DT.fft_cooley(cwave1)
    fft_cwave1test = np.fft.fft(cwave1)
    fft_wave = DT.fft_cooley(w1y)
    fft_wavetest = np.fft.fft(w1y)

    # fft_real = [x.real for x in fft_cwave1]
    fft_real = [x.real for x in fft_wave]
    
    # plt.title('Complex Wave 1 FFT')
    # if len(w1x) < len(w2x):
    #     plt.plot(w1x, fft_real, 'go')
    #     plt.show()
    # else:
    #     plt.plot(w2x, fft_real, 'go')
    #     plt.show()

    plt.title('Wave 1 FFT')
    plt.plot(w1x, fft_real, 'go')
    plt.show()
    
    # fft_real = [x.real for x in fft_cwave1test]
    # plt.title('Complex Wave 1 FFT numpy vers')
    # if len(w1x) < len(w2x):
    #     plt.plot(w1x, fft_real, 'go')
    #     plt.show()
    # else:
    #     plt.plot(w2x, fft_real, 'go')
    #     plt.show()

    fft_real = [x.real for x in fft_wavetest]
    plt.title('Wave 1 FFT numpy vers')
    plt.plot(w1x, fft_real, 'go')
    plt.show()

    #NEW
    fft_complex = [complex(x) for x in fft_data]
    print("FFT of Verification Data: ")
    fft_verification = DT.fft_cooley(fft_complex)
    # with open("FFT_verification_TEST.txt", "w+") as f:
    #     f.write("FFT of Verification Data:\n")
    #     for d in fft_verification:
    #         line = str(d)
    #         line = line[1:-1] # get rid of ()
    #         line = line.replace('j', 'i') # convert back to std. form
    #         f.write(line + "\n")
    
    fft_complex = [complex(x) for x in w1y] #SINE WAVE1
    print("FFT of Sine Wave: ")
    fft_verification = DT.fft_cooley(fft_complex)
    # with open("FFT_sine_TEST.txt", "w+") as f:
    #     f.write("Sine Data Points:\n")
    #     for d in w1y:
    #         f.write(str(d) + "\n")
    #     f.write("FFT of Sine Wave Data:\n")
    #     for d in fft_verification:
    #         line = str(d)
    #         line = line[1:-1] # get rid of ()
    #         line = line.replace('j', 'i') # convert back to std. form
    #         f.write(line + "\n")

    #--
    # Problem 1a TEST
    # Compute, for S samples and up to k summations, the complex waves:
    S = [3, 10, 50]
    N = 512
    t1 = 1/N
    tlist = []
    for t in range(N): # generate samples
        tlist.append(t1*t)
    
    fst1 = [] # list of values -> 512 samples, each sample being k=1 thru s
    fst2 = []
    fst3 = []

    gst1 = []
    gst2 = []
    gst3 = []

    # Fs(t)
    # OUTER LOOP = t
    # for t in range(N):
    #     fst_current = 0.0 # hold sum of k->s
    #     # INNER LOOP = k->s
    #     for k in range(1, S[0]):
    #         fst_current += (sin(DT.TWO_PI*(2*k-1)*t)) / (2*k-1)
    #     fst1.append(fst_current)
    # for t in range(N):
    #     fst1_current = 0.0 # hold sum of k->s
    #     fst2_current = 0.0 # hold sum of k->s
    #     fst3_current = 0.0 # hold sum of k->s
    #     # INNER LOOP = k->s
    #     for k in range(1, S[0]+1):
    #         fst1_current += (sin(DT.TWO_PI*(2*k-1)*t)) / (2*k-1)
    #     for k in range(1, S[1]+1):
    #         fst2_current += (sin(DT.TWO_PI*(2*k-1)*t)) / (2*k-1)
    #     for k in range(1, S[2]+1):
    #         fst3_current += (sin(DT.TWO_PI*(2*k-1)*t)) / (2*k-1)
    #     fst1.append(fst1_current)
    #     fst2.append(fst2_current)
    #     fst3.append(fst3_current)

    # TEST 3
    # for k in range(1, S[0]+1):
    #     fst1_current = 0.0 # hold sum of k->s
    #     # INNER LOOP = k->s
    #     for t in range(N):
    #         fst1_current = (sin(DT.TWO_PI*(2*k-1)*tlist[t])) / (2*k-1)
    #         fst1.append(fst1_current)
    # for k in range(1, S[1]+1):
    #     fst2_current = 0.0 # hold sum of k->s
    #     # INNER LOOP = k->s
    #     for t in range(N):
    #         fst2_current = (sin(DT.TWO_PI*(2*k-1)*tlist[t])) / (2*k-1)
    #         fst2.append(fst2_current)
    # for k in range(1, S[2]+1):
    #     fst3_current = 0.0 # hold sum of k->s
    #     # INNER LOOP = k->s
    #     for t in range(N):
    #         fst3_current = (sin(DT.TWO_PI*(2*k-1)*tlist[t])) / (2*k-1)
    #         fst3.append(fst3_current)

    # TEST 4 - success!
    for t in range(N):
        fst1_current = 0.0 # hold sum of k->s
        # INNER LOOP = k->s
        for k in range(1, S[0]+1):
            fst1_current += (sin(DT.TWO_PI*(2*k-1)*tlist[t])) / (2*k-1)
        fst1.append(fst1_current)
    for t in range(N):
        fst2_current = 0.0 # hold sum of k->s
        # INNER LOOP = k->s
        for k in range(1, S[1]+1):
            fst2_current += (sin(DT.TWO_PI*(2*k-1)*tlist[t])) / (2*k-1)
        fst2.append(fst2_current)
    for t in range(N):
        fst3_current = 0.0 # hold sum of k->s
        # INNER LOOP = k->s
        for k in range(1, S[2]+1):
            fst3_current += (sin(DT.TWO_PI*(2*k-1)*tlist[t])) / (2*k-1)
        fst3.append(fst3_current)

    # for t in range(N):
    #     fst_current = 0.0 # hold sum of k->s
    #     # INNER LOOP = k->s
    #     for k in range(1, S[1]):
    #         fst_current += (sin(DT.TWO_PI*(2*k-1)*t)) / (2*k-1)
    #     fst2.append(fst_current)
    # for t in range(N):
    #     fst_current = 0.0 # hold sum of k->s
    #     # INNER LOOP = k->s
    #     for k in range(1, S[2]):
    #         # temp = sin(DT.TWO_PI*(2*k-1)) / (2*k-1)
    #         # print(temp)
    #         fst_current += (sin(DT.TWO_PI*(2*k-1)*t)) / (2*k-1)
    #     fst3.append(fst_current)
    #TEST
    # print("Fs(t) 1:")
    # with open("fst.txt", "w+") as f:
    #     f.write("Fs(t):\n")
    #     for i in range(len(fst1)):
    #         f.write(str(fst1[i]))
    #         f.write("\n")
    
    #Gs(t)
    # OUTER LOOP = t
    # for t in range(N):
    #     gst_current = 0.0 # hold sum of k->s
    #     # INNER LOOP = k->s
    #     for k in range(1, S[0]):
    #         gst_current += (sin(DT.TWO_PI*(2*k)*t)) / (2*k)
    #     gst1.append(gst_current)
    # for t in range(N):
    #     gst_current = 0.0 # hold sum of k->s
    #     # INNER LOOP = k->s
    #     for k in range(1, S[1]):
    #         gst_current += (sin(DT.TWO_PI*(2*k)*t)) / (2*k)
    #     gst2.append(gst_current)
    # for t in range(N):
    #     gst_current = 0.0 # hold sum of k->s
    #     # INNER LOOP = k->s
    #     for k in range(1, S[2]):
    #         # temp = sin(DT.TWO_PI*(2*k-1)) / (2*k)
    #         # print(temp)
    #         gst_current += (sin(DT.TWO_PI*(2*k-1)*t)) / (2*k)
    #     gst3.append(gst_current)

    for t in range(N):
        gst1_current = 0.0 # hold sum of k->s
        # INNER LOOP = k->s
        for k in range(1, S[0]+1):
            gst1_current += (sin(DT.TWO_PI*(2*k)*tlist[t])) / (2*k)
        gst1.append(gst1_current)
    for t in range(N):
        gst2_current = 0.0 # hold sum of k->s
        # INNER LOOP = k->s
        for k in range(1, S[1]+1):
            gst2_current += (sin(DT.TWO_PI*(2*k)*tlist[t])) / (2*k)
        gst2.append(gst2_current)
    for t in range(N):
        gst3_current = 0.0 # hold sum of k->s
        # INNER LOOP = k->s
        for k in range(1, S[2]+1):
            gst3_current += (sin(DT.TWO_PI*(2*k)*tlist[t])) / (2*k)
        gst3.append(gst3_current)

    # Show 1a waves as plots:
    plt.title('Problem 1a Complex Waves part i')
    plt.plot(tlist, fst1, 'r-', tlist, fst2, 'g-', tlist, fst3, 'b-') # rgb lines
    xmin = 0
    xmax = max(tlist)
    
    ymax = max(fst3)
    if ymax >= 0:
        ymax += abs(ymax / 2.0)
    # if ymax <= 0:
    #     ymax = 0.1 + abs(ymax)
    ymin = min(fst3)
    if ymin < 0:
        ymin *= 1.2
        # ymin -= ymin
    else:
        ymin *= -1.2
    axis_list = [xmin, xmax, ymin, ymax] #xmin
    plt.axis(axis_list)
    plt.show()

    plt.title('Problem 1a Complex Waves part ii')
    plt.plot(tlist, gst1, 'r-', tlist, gst2, 'g-', tlist, gst3, 'b-') # rgb lines
    xmin = 0
    xmax = max(tlist)
    
    ymax = max(gst3)
    if ymax >= 0:
        ymax += abs(ymax / 2.0)
    # if ymax <= 0:
    #     ymax = 0.1 + abs(ymax)
    ymin = min(gst3)
    if ymin < 0:
        ymin *= 1.2
        # ymin -= ymin
    else:
        ymin *= -1.2
    axis_list = [xmin, xmax, ymin, ymax] #xmin
    plt.axis(axis_list)
    plt.show()
    #--

if __name__ == "__main__":
    main()
