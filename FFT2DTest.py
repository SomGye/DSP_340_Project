import numpy as np
import DSPToolkit as DT

# TEST 2D FFT with simple data
def main():
    # Setup data
    # NOTE: Need rows, cols to be powers of 2!
    data1 = [[5,25,25,5],[25,25,25,25],[5,25,25,5],[25,25,25,25]]
    data2 = [[5,10,15,10],[5,25,45,25],[5,10,15,10],[5,25,45,25]]
    # Show base data
    np_data1 = np.asarray(data1)
    print("data1 as np array:")
    print(np_data1)
    np_data2 = np.asarray(data2)
    print("data2 as np array:")
    print(np_data2)
    print("--")
    # Show results comparison
    print("np result 1:")
    np_res1 = np.fft.fft2(data1)
    print(np_res1)
    print("my result 1:")
    res1 = DT.two_dim_fft(data1)
    print(res1)
    
    print("np result 2:")
    np_res2 = np.fft.fft2(data2)
    print(np_res2)
    print("my result 2:")
    res2 = DT.two_dim_fft(data2)
    print(res2)
    print("--")
main()