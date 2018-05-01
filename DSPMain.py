import matplotlib.pyplot as plt # for graphing
import cmath                    # for complex number funcs
from math import sin, pi, log, floor
import numpy as np # TEST
import cv2 # OpenCV for imgs
from collections import Counter
import DSPToolkit as DT

# Globals
EPSILON = 0.000000000000001
IMG_SIZE = 512

def main():
    # # Create wave objects
    # wave1 = DT.DSPWave() # default settings
    # wave2 = DT.DSPWave(1.2, 2, 0.1, 0.1)
    # coswave = DT.DSPWave(1, 1, pi / 2.0, 0) # only phase shift!

    # # Get wave x/y axes:
    # w1x = wave1.generateTimeSet()
    # w1y = wave1.generateSineSampleSet()
    # w2x = wave2.generateTimeSet()
    # w2y = wave2.generateSineSampleSet()
    # cosx = coswave.generateTimeSet()
    # cosy = coswave.generateSineSampleSet() # was sine...
    # # cosy = coswave.generateCosineSampleSet()

    # # Get complex waveforms:
    # cwave1 = DT.waveAddition(w1y, w2y)
    # cwave2 = DT.waveMultiplication(w1y, w2y)

    # # Plot wave 1:
    # print("\n\t * Sine Wave 1")
    # plt.title('Sine Wave 1')
    # plt.plot(w1x, w1y, 'b-') # blue line
    # xmax = max(w1x) * 1.5
    # xmin = min(w1x)
    # if xmin < 0:
    #     xmin *= 1.5
    # else:
    #     xmin *= -1.5
    # ymax = max(w1y) * 1.5
    # ymin = min(w1y)
    # if ymin < 0:
    #     ymin *= 1.5
    # else:
    #     ymin *= -1.5
    # axis_list = [xmin, xmax, ymin, ymax]
    # plt.axis(axis_list)
    # plt.show()

    # # Plot wave 2:
    # print("\n\t * Sine Wave 2")
    # plt.title('Sine Wave 2')
    # plt.plot(w2x, w2y, 'r-') # red line
    # xmax = max(w2x) * 1.5
    # xmin = min(w2x)
    # if xmin < 0:
    #     xmin *= 1.5
    # else:
    #     xmin *= -1.5
    # ymax = max(w2y) * 1.5
    # ymin = min(w2y)
    # if ymin < 0:
    #     ymin *= 1.5
    # else:
    #     ymin *= -1.5
    # axis_list = [xmin, xmax, ymin, ymax]
    # plt.axis(axis_list)
    # plt.show()

    # # Plot cosine wave:
    # print("\n\t * Cosine Wave")
    # plt.title('Cosine Wave')
    # plt.plot(cosx, cosy, 'g-') # green line
    # xmax = max(cosx) * 1.5
    # xmin = min(cosx)
    # if xmin < 0:
    #     xmin *= 1.5
    # else:
    #     xmin *= -1.5
    # ymax = max(cosy) * 1.5
    # ymin = min(cosy)
    # if ymin < 0:
    #     ymin *= 1.5
    # else:
    #     ymin *= -1.5
    # axis_list = [xmin, xmax, ymin, ymax]
    # plt.axis(axis_list)
    # plt.show()

    # # Plot wave 1 and wave 2:
    # print("\n\t * Sine Waves 1 and 2")
    # plt.title('Sine Waves 1 and 2')
    # plt.plot(w1x, w1y, 'b-', w2x, w2y, 'r-')
    # plt.show()

    # # Plot complex wave 1 (add):
    # print("\n\t * Complex Wave 1 - Addition")
    # plt.title('Complex Wave 1 - Addition')
    # if len(w1x) < len(w2x):
    #     plt.plot(w1x, cwave1, 'bs')
    #     plt.show()
    # else:
    #     plt.plot(w2x, cwave1, 'bs')
    #     plt.show()

    # # Plot complex wave 2 (mult):
    # print("\n\t * Complex Wave 2 - Multiplication")
    # plt.title('Complex Wave 2 - Multiplication')
    # if len(w1x) < len(w2x):
    #     plt.plot(w1x, cwave2, 'bs')
    #     plt.show()
    # else:
    #     plt.plot(w2x, cwave2, 'bs')
    #     plt.show()

    # #--
    # # FFT TEST
    # print("\n\t * FFT TEST")
    # fft_cwave1 = DT.fft_cooley(cwave1)
    # fft_cwave1test = np.fft.fft(cwave1)
    # fft_wave = DT.fft_cooley(w1y)
    # fft_wavetest = np.fft.fft(w1y)

    # fft_real = [x.real for x in fft_wave]

    # plt.title('Wave 1 FFT')
    # plt.plot(w1x, fft_real, 'go')
    # plt.show()

    # fft_real = [x.real for x in fft_wavetest]
    # plt.title('Wave 1 FFT numpy vers')
    # plt.plot(w1x, fft_real, 'go')
    # plt.show()

    # # FFT TEST NEW
    # print("\n\t * FFT TEST NEW")
    # fft_data = [26160.0, 19011.0, 18757.0, 18405.0, 17888.0, 14720.0, 14285.0,17018.0, 18014.0, 17119.0, 16400.0, 17497.0, 17846.0, 15700.0, 17636.0, 17181.0] #NEW
    # fft_complex = [complex(x) for x in fft_data]
    # print("FFT of Verification Data: ")
    # fft_verification = DT.fft_cooley(fft_complex)
    # for d in fft_verification:
    #     print(d)
    # print("--")
    
    # fft_complex = [complex(x) for x in w1y] #SINE WAVE1
    # print("FFT of Sine Wave: ")
    # fft_verification = DT.fft_cooley(fft_complex)
    # for d in fft_verification:
    #     print(d)
    # print("--")

    #--
    ## Problem 1a
    print("\n\t # Problem 1a...")
    # Compute, for S samples and up to k summations, the complex waves:
    S = [3, 10, 50]
    N = 512
    t1 = 1/N
    tlist = []
    for t in range(N): # generate samples
        tlist.append(t1*t)
    
    fst1 = [] # list of values -> 512 samples, each sample being k=1 thru s
    fst2 = []
    fst3 = [] # f50t

    gst1 = []
    gst2 = []
    gst3 = [] # g50t

    # fst
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
    # gst
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
    ymin = min(gst3)
    if ymin < 0:
        ymin *= 1.2
        # ymin -= ymin
    else:
        ymin *= -1.2
    axis_list = [xmin, xmax, ymin, ymax]
    plt.axis(axis_list)
    plt.show()
    #--
    ## Problem 1b
    print("\n\t # Problem 1b...")
    print("\n * Getting FTs and PSDs...")
    # Get FTs of f50t, g50t
    ft_fst3 = DT.fft_cooley(fst3)
    ft_gst3 = DT.fft_cooley(gst3)

    # Get PSDs of f50t, g50t
    psd_fst3 = DT.psd(ft_fst3)
    psd_gst3 = DT.psd(ft_gst3)

    # Convert PSD data back to real:
    psd_fst3 = [x.real for x in psd_fst3]
    psd_gst3 = [x.real for x in psd_gst3]

    # Show PSD plots:
    plt.title('PSDs f50t, g50t')
    plt.plot(tlist, psd_fst3, 'ro', tlist, psd_gst3, 'bo')
    
    xmin = -1
    xmax = 2
    ymin = 0
    ymax = max([max(psd_fst3), max(psd_gst3)]) # max of both lists
    axis_list = [xmin, xmax, ymin, ymax]
    print("\n * Showing PSD plots...")
    plt.axis(axis_list)
    plt.show()
    #--
    ## Problem 2
    print("\n\t # Problem 2...")
    v1t = []
    v2t = []
    xt = []
    yt = []
    v1f = 13
    v2f = 31
    # Get initial v signals and x, y signals:
    for x in range(N):
        v1t.append(sin(DT.TWO_PI*v1f*tlist[x]))
        v2t.append(sin(DT.TWO_PI*v2f*tlist[x]))
        xt.append(v1t[x] + v2t[x])
        yt.append(v1t[x] * v2t[x])
    # Get PSDs of x(t), y(t):
    psd_xt = DT.psd(xt, need_FT=True)
    psd_yt = DT.psd(yt, need_FT=True)

    # Convert PSD data back to real:
    psd_xt = [x.real for x in psd_xt]
    psd_yt = [x.real for x in psd_yt]

    # Show PSD plots:
    plt.title('PSDs xt, yt')
    plt.plot(tlist, psd_xt, 'ro', tlist, psd_yt, 'bo')

    xmin = -1
    xmax = 2
    ymin = 0
    ymax = max([max(psd_fst3), max(psd_gst3)]) # max of both lists
    axis_list = [xmin, xmax, ymin, ymax]
    print("\n * Showing PSD plots...")
    plt.axis(axis_list)
    plt.show()
    #--
    ## Problem 3
    print("\n\t # Problem 3...")
    N = 256
    t1 = 1/N
    tlist = []
    for t in range(N): # generate samples
        tlist.append(t1*t)
    gt = []
    gpt = [] # phase shifted by 1
    phase = 1

    # Get initial g signal and gp signal:
    for x in range(N):
        gt.append(sin(DT.TWO_PI*tlist[x]))
        gpt.append(sin(DT.TWO_PI*(tlist[x]-phase)))
    
    # Get PSDs of g, gp:
    psd_gt = DT.psd(gt, need_FT=True)
    psd_gpt = DT.psd(gpt, need_FT=True)

    # Convert PSD data back to real:
    psd_gt = [x.real for x in psd_gt]
    psd_gpt = [x.real for x in psd_gpt]

    # Get diffs in PSDs:
    psd_diffs = []
    for x in range(len(psd_gt)):
        psd_diffs.append(abs(psd_gt[x] - psd_gpt[x]))
    
    # Get max and avg diffs:
    avg_diff = sum(psd_diffs) / len(psd_diffs)
    max_diff = max(psd_diffs)

    # Display results:
    print("Phase shift was : " + str(phase))
    print("Avg Difference b/w Phase-shifted PSDs: " + str(avg_diff))
    print("Max Difference b/w Phase-shifted PSDs: " + str(max_diff))
    print("Chosen Epsilon: " + str(EPSILON))
    print("Was average diff less than epsilon? " + str(avg_diff < EPSILON))
    print("--")
    #--
    ## Problem 5 - DTMF
    print("\n\t # Problem 5...")
    tonedata1 = DT.read_dtmf_data('tonedataA1.txt')
    tonedata2 = DT.read_dtmf_data('tonedataB1.txt')
    # results1 = DT.dtmf_decoder(tonedata1)
    # results2 = DT.dtmf_decoder(tonedata2)
    # print("Tone Data 1: ")
    # print('* Decoded key is: ', results1[0])
    # print('=> which came from freqs: ' + str(results1[1]) + " and " + str(results1[2]))
    # print("=> DTMF was determined from freq. sets: ")
    # for r in results1[-1]:
    #     print(str(r)[1:-1])
    # print("--")
    # print("Tone Data 2: ")
    # print('* Decoded key is: ', results2[0])
    # print('=> which came from freqs: ' + str(results2[1]) + " and " + str(results2[2]))
    # print("=> DTMF was determined from freq. sets: ")
    # for r in results2[-1]:
    #     print(str(r)[1:-1])
    # print("--")

    # TEST - switch data...
    print("TEST switch data...")
    # Get FTs of A1, B1:
    A1 = DT.fft_cooley(tonedata1)
    B1 = DT.fft_cooley(tonedata2)

    # Convert FTs to reals:
    A1 = [x.real for x in A1]
    B1 = [x.real for x in B1]

    # Get signs of reals:
    A1 = [int(np.sign(x)) for x in A1]
    B1 = [int(np.sign(x)) for x in B1]

    # Get switch data:
    A1switch = DT.get_switch_data(A1)
    B1switch = DT.get_switch_data(B1)

    # Get freq. bin locations
    # and determine if freq. is in tolerance range:
    A1_locs = []
    B1_locs = []
    lows = [697, 770, 852, 941]
    highs = [1209, 1336, 1477, 1633]
    tolerance = 90
    A1_matches = []
    B1_matches = []
    for x in range(len(A1switch)):
        if A1switch[x] == 1:
            A1_locs.append(x + 1)
    for x in range(len(B1switch)):
        if B1switch[x] == 1:
            B1_locs.append(x + 1)
    print("A1 locations: ")
    for L in A1_locs:
        print(L)
        tempset = []
        for x in lows:
            if int(L) % x < tolerance: # multiple
                tempset.append(x)
        if tempset != list():
            A1_matches.append(tempset)
        tempset = []
        for x in highs:
            if int(L) % x < tolerance: # multiple
                tempset.append(x)
        if tempset != list():
            A1_matches.append(tempset)
    print("B1 locations: ")
    for L in B1_locs:
        print(L)
        tempset = []
        for x in lows:
            if int(L) % x < tolerance: # multiple
                tempset.append(x)
        if tempset != list():
            B1_matches.append(tempset)
        tempset = []
        for x in highs:
            if int(L) % x < tolerance: # multiple
                tempset.append(x)
        if tempset != list():
            B1_matches.append(tempset)
    print("A1 matches: ")
    for m in A1_matches:
        print(m)
    print("B1 matches: ")
    for m in B1_matches:
        print(m)
    
    # Flatten the match lists:
    A1_matches = [x for y in A1_matches for x in y]
    B1_matches = [x for y in B1_matches for x in y]

    # Count the freq. matches:
    print("Most common element in A1 matches: ")
    temp = tuple(A1_matches)
    A1_count = Counter(temp)
    print(A1_count)
    print("Most common element in B1 matches: ")
    temp = tuple(B1_matches)
    B1_count = Counter(temp)
    print(B1_count)
    # Get highest correlation of low, high freqs in A1, B1:
    A1_count2 = []
    A1_result = []
    low_done = False
    high_done = False
    for key, value in A1_count.items():
        A1_count2.append([value, key])
    A1_count2.sort(reverse=True) # descending
    for value, key in A1_count2:
        if not low_done:
            if key in lows:
                A1_result.append(key)
                low_done = True
        if not high_done:
            if key in highs:
                A1_result.append(key)
                high_done = True
        if low_done and high_done:
            break
    B1_count2 = []
    B1_result = []
    low_done = False
    high_done = False
    for key, value in B1_count.items():
        B1_count2.append([value, key])
    B1_count2.sort(reverse=True) # descending
    for value, key in B1_count2:
        if not low_done:
            if key in lows:
                B1_result.append(key)
                low_done = True
        if not high_done:
            if key in highs:
                B1_result.append(key)
                high_done = True
        if low_done and high_done:
            break
    print("A1 result: ")
    print(A1_result)
    print("B1 result: ")
    print(B1_result)
    print("--")
    #--
    ## Problem 7 - 2D FFT and Correlation
    print("\n\t # Problem 7...")
    # Create base imgs
    img1 = np.zeros((IMG_SIZE, IMG_SIZE, 1), np.uint8)
    img2 = np.zeros((IMG_SIZE, IMG_SIZE, 1), np.uint8)

    # Create copies to contain C images
    C1 = np.copy(img1) # TEST SIGNAL (RETURN)
    C2 = np.copy(img2) # PULSE

    ## Draw white dots in region R
    # NOTE: have to do manual array method!
    ## C1 - test signal
    # Iteration 1:
    # row 180, col 220 -> +140 row, +110 col
    startr = 180
    startc = 220
    endr = startr + 140
    endc = startc + 110
    for x in range(startr, endr + 1):
        for y in range(startc, endc + 1):
            C1[x][y] = 255 # white

    # Iteration 2:
    # block = 50
    block_height = 50
    block_width = 30
    startr = 180 + int(block_height / 2)
    endr = startr + 90
    startc = endc - block_width # subtract width; endc is the same!
    for x in range(startr, endr + 1):
        for y in range(startc, endc + 1):
            C1[x][y] = 0 # black

    ## C2 - pulse
    # Iteration 1:
    startr = 0
    startc = 0
    endr = 120
    endc = 30
    for x in range(startr, endr + 1):
        for y in range(startc, endc + 1):
            C2[x][y] = 255 # white

    # Iteration 2:
    block_height = 15
    block_width = 15
    startr = startr + block_height
    endr = endr - block_height
    startc = endc - block_width # endc is the same!
    for x in range(startr, endr + 1):
        for y in range(startc, endc + 1):
            C2[x][y] = 0 # black

    # Perform 2D FFT of Signal and Pulse:
    U_of_m = DT.two_dim_fft2(C1) # 2D FFT of return signal
    H_of_m = DT.two_dim_fft2(C2) # 2D FFT of pulse

    # Perform Fast Convolution of C1 and C2:
    U_and_H = np.dot(U_of_m, H_of_m)
    y_of_k = DT.two_dim_fft2(U_and_H, inverse=True)

    # Make a shell for real values of y(k):
    yreal = np.zeros((IMG_SIZE, IMG_SIZE, 1), np.float64)

    # Loop thru y of k 1
    # Get reals and scale by log:
    ymax = np.max(y_of_k[np.nonzero(y_of_k)]).real
    ymin = np.min(y_of_k[np.nonzero(y_of_k)]).real

    needFlip = False
    if ymax <= EPSILON:
        needFlip = True
    for x in range(len(y_of_k)):
        for y in range(len(y_of_k[0])):
            # Convert to Real part of Complex num:
            yreal[x][y] = y_of_k[x][y].real
            # Flip if needed:
            if needFlip:
                yreal[x][y] *= -1.0
            # Scale by log:
            if yreal[x][y] > EPSILON:
                yreal[x][y] = log(yreal[x][y])
            else:
                yreal[x][y] = 0.0 # negatives go to 0

    # Loop thru y of k 2
    # Get max and scaling factor, scale to limit,
    #  and convert to int:
    ymax = np.max(yreal[np.nonzero(yreal)])
    ymin = np.min(yreal[np.nonzero(yreal)])
    needScale = False
    # Flip if needed:
    if ymax <= EPSILON:
        yreal *= -1.0
        ymax *= -1.0
    # if ymax > 255.0 or ymax < 255.0:
    if ymax != 255.0:
        needScale = True
    limit = 255
    scaling_factor = limit / EPSILON
    if ymax > EPSILON:
        scaling_factor = limit / ymax
    for x in range(len(yreal)):
        for y in range(len(yreal[0])):
            # Scale linearly to within 8bit range:
            if needScale:
                yreal[x][y] *= scaling_factor
            # Convert to int:
            yreal[x][y] = floor(yreal[x][y])
            # Eliminate negatives:
            if yreal[x][y] < 0:
                yreal[x][y] = 0

    # Ensure correct format for y(k) img display:
    yreal = np.asarray(yreal, np.uint8)

    # Create Magnitude Img
    ymag = np.copy(yreal)

    # Get top percentile of magnitudes:
    ymax = np.max(ymag[np.nonzero(ymag)])
    top_percent = 0.7
    top_percentile = ymax * top_percent

    # Convert 1-channel to 3-channel BGR img:
    ymag = cv2.cvtColor(ymag, cv2.COLOR_GRAY2BGR)

    # Loop and paint top mags red:
    for x in range(1, IMG_SIZE-1):
        for y in range(1, IMG_SIZE-1):
            # Check for top percentile:
            if ymag[x][y][0] >= top_percentile or \
                ymag[x][y][1] >= top_percentile or \
                ymag[x][y][2] >= top_percentile: # can check any channel...
                # Paint main cell red:
                ymag[x][y][-1] = 255 # max out RED channel!
                # Also paint adjacent cells...
                ymag[x-1][y][-1] = 255
                ymag[x-1][y-1][-1] = 255
                ymag[x][y-1][-1] = 255

    # Ensure correct format for mag img display:
    ymag = np.asarray(ymag, np.uint8)
    # --
    print("\n\t * Displaying signal images...")
    # Display C1 - initial test signal
    cv2.imshow("C1 test", C1)
    cv2.waitKey(0)

    # Display C2 - initial pulse
    cv2.imshow("C2 test", C2)
    cv2.waitKey(0)

    # Display y of k:
    cv2.imwrite("y_of_k.png", yreal, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imshow("Y of k (real-only)", yreal)
    # Display y of k top magnitudes:
    cv2.imwrite("y_of_k_top.png", ymag, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imshow("Y of k top mags (real-only)", ymag)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
