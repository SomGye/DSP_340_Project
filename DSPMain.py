import matplotlib.pyplot as plt
import DSPToolkit as DT

def main():
    # Create wave objects
    wave1 = DT.DSPWave() # default settings
    wave2 = DT.DSPWave(1.2, 2, 0.1, 0.1)

    # Get wave x/y axes:
    w1x = wave1.generateTimeSet()
    w1y = wave1.generateSampleSet()
    w2x = wave2.generateTimeSet()
    w2y = wave2.generateSampleSet()

    # Get complex waveforms:
    cwave1 = DT.waveAddition(w1y, w2y)
    cwave2 = DT.waveMultiplication(w1y, w2y)

    # Plot wave 1:
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
    axis_list = [xmin, xmax, ymin, ymax] #xmin
    plt.axis(axis_list)
    plt.show()

    # Plot wave 2:
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
    axis_list = [xmin, xmax, ymin, ymax] #xmin
    plt.axis(axis_list)
    plt.show()

    # Plot wave 1 and wave 2:
    plt.plot(w1x, w1y, 'b-', w2x, w2y, 'r-')
    plt.show()

    # Plot complex wave 1 (add):
    if len(w1x) < len(w2x):
        plt.plot(w1x, cwave1, 'bs')
        plt.show()
    else:
        plt.plot(w2x, cwave1, 'bs')
        plt.show()

    # Plot complex wave 2 (mult):
    if len(w1x) < len(w2x):
        plt.plot(w1x, cwave2, 'bs')
        plt.show()
    else:
        plt.plot(w2x, cwave2, 'bs')
        plt.show()

if __name__ == "__main__":
    main()
