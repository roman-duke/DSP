from scipy.io import loadmat
from scipy.fft import fft, ifft, fftfreq, fftshift
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

#The Function Below reads the set of data from the FILTER.mat file"
def y():
    data = loadmat("FILTER.mat")
    return data["input_0"][0]

#Converts the input signal to the frequency domain
def freq_domain(y):
    return fft(y())

#Decorates a graph with the supplied arguments
def graph_decorator(ax1, x_label, y_label, title):
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)

#Generates a graph in either the time domain or the frequency domain
def graph(y, domain="") -> None:
    xf = [(0.5 + i)*2 for i in fftshift(fftfreq(501, 1))]
    h1 = np.abs(freq_domain(y))
    ax1 = plt.subplot()

    if domain == "time":
        x = np.arange(0, 501, 1)
        ax1.plot(x, y())
        graph_decorator(ax1, "Time", "Amplitude", "Input Signal, s[n]")
        plt.show()

    elif domain == "frequency":
        ax1.plot(xf, h1)
        graph_decorator(ax1, "Normalized Frequency", "Magnitude", "Frequency response " +\
                        "of Input Signal, |S(ω)|")
        plt.show()
        
    return h1

#Designs a filter with the following supplied arguments: filter_order, cutoff frequency and window method
def filter_design(filter_order=200, cutoff=0.02, window="boxcar", n=0) -> None:
    filter = signal.firwin(filter_order, cutoff, window=window)
    w1, h1 = signal.freqz(filter, fs=2)
    ax1 = plt.subplot()
    if n == 1:
        plt.plot(w1, filter_order/2*np.log10(abs(h1)), 'r')
        graph_decorator(ax1, 'Normalized Frequency (xπ rad/sample)', 'Magnitude Response (dB)', 
                    'Magnitude response (dB), |H(ω)|dB')
        plt.grid()
        plt.show()

    elif n == 2:
        ax1, ax2 = plt.subplot(211), plt.subplot(212)
        ax1.set_title('Frequency Response, |H(ω)| and ∠|H(ω)|')
        ax1.set_ylabel('Magnitude', color='b')        
        ax1.plot(w1, np.abs(h1), 'b')
        angles = np.unwrap(np.angle(h1))
        ax2.plot(w1, angles, 'g')
        ax2.set_ylabel('Phase (rad)', color='g')
        ax1.grid(), ax2.grid()
        ax2.set_xlabel('Normalized Frequency [xπ rad/sample]')
        plt.show()

    # if n == 3:
    #     z, p, k = signal.tf2zpk()
    # TODO: make a Pole/Zero plot of the impulse response of the filter

    elif n == 4:
        filter_time = ifft(h1)
        plt.plot(np.arange(0, filter_order, filter_order/len(filter_time)), filter_time, 'b')
        plt.show()
        # ax1 = plt.subplot()
        # ax1.stem(signal.windows.get_window("boxcar", 200))
        # graph_decorator(ax1, "[n]", "x[n]", "Time domain of the impulse response of the Filter")
        # plt.show()
        
    else: return (w1, h1)

#Filters the noise from the input signal
def filter():
    w1, h1 = (filter_design()[0][:501], filter_design()[1][:501])
    h = h1 * graph(y)
    ax1, ax2 = plt.subplot(211), plt.subplot(212)
    graph_decorator(ax1, 'Normalized Frequency', 'Magnitude', 'Frequency Response ' +\
                    'of Output Signal, |Y(ω)|')
    graph_decorator(ax2, "Time", "Amplitude", "Output Signal, y[n]")
    ax1.plot(w1, np.abs(h), 'b')
    output = fftshift(ifft(h))
    ax2.plot(np.arange(0, 501, 1), output, 'r')
    plt.show()
    

graph(y, "time")
graph(y, "frequency")
filter_design(200, 0.022, "boxcar", 1)     # Produces the graph of the magnitude dB of the Frequency Response
filter_design(200, 0.02, "boxcar", 2)     # Produces the Magnitude and Phase of the Frequency Response
# filter_design(200, 0.022, "boxcar", 3)   # TODO: fix the Pole/Zero Plot of the impulse response of the filter
filter_design(200, 0.022, "boxcar", 4)     # Converts the Frequency Response to Time domain
filter()
