import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import fft
import nfft

def compute_nfft(sample_instants, sample_values):
    N = len(sample_instants) * 4
    T = np.max(sample_instants) - np.min(sample_instants)
    sample_instants = ( sample_instants - np.min(sample_instants) ) / T - 0.5

    y = nfft.nfft_adjoint(sample_instants, sample_values, N)

    x = np.linspace(- N / (2.0 * T), N / (2.0 * T), N)

    f = nfft.nfft(sample_instants, y / N)

    return (x, np.abs(y), f, N)


def f(x):
    f = (x < -2.) * 1. + ((x >= -2.) & (x < 0.)) * 0. + ((x >= 0.) & (x < 2.)) * 1. + (x >= 2.) * 0.

    k = 1.
    f = np.sin(x * np.pi * k) + np.sin(x * np.pi * 1.3 * k)
    return f


def g(tau, dx):
    tau = tau / dx
    M = - int(tau * np.log(0.01))
    g = scipy.signal.windows.exponential(M+1, 0, tau, False)
    return g, M

def main():

    k = 5
    x = - 5. + 10.*np.random.rand(1000)
    x.sort()
    x, dx = np.linspace(-5., 5., 1000, retstep=True)
    #f = np.sin(x * np.pi * k) + np.sin(x * np.pi * 4. * k)
    #f = (x < 0.) * 2. + (x >= 0.) * 1.
    #f = (x < 0.) * 2. + (x >= 0.) * (2. - 1. * (1. - np.exp(- x / 0.1)))
    
    #freq, f_hat, f_new, N = compute_nfft(x, f)

    #tau = - (M-1) / np.log(0.01)
    #win = scipy.signal.windows.hann(100)
    win2, M = g(0.25, dx)
    f_new_fft = scipy.signal.oaconvolve(f(x), win2, mode='full')[:-M] / sum(win2)
    f_reconstruct, remainder = scipy.signal.deconvolve(f_new_fft, win2)
    f_reconstruct = f_reconstruct * sum(win2)

    f_reconstruct = scipy.signal.oaconvolve(f_reconstruct, win

    #plt.plot(x, f(x), 'o')
    #plt.plot(x, f_new, 'o')
    plt.plot(x, f(x), lw=3.)
    plt.plot(x, f_new_fft, lw=2.5)
    plt.plot(x[:-M], f_reconstruct)
    plt.show()
    
    plt.plot(fft.rfftfreq(len(f), d=x[1]-x[0]), np.abs(f_hat_fft))
    #plt.plot(freq[N // 2:], f_hat[N // 2:])
    plt.show()

if __name__ == '__main__': main()
