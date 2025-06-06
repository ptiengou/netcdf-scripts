import numpy as np
import scipy
from numba import jit
import bisect
import signal_analysis as sigana


def interpolate(x, x_step_or_list, *args, kind='quadratic'):
    if np.isscalar(x_step_or_list):
        x_inter = np.arange(x[0], x[-1], x_step_or_list)
    else:
        x_inter = x_step_or_list
    y_inter = [scipy.interpolate.interp1d(x, y, kind=kind)(x_inter) for y in args]
    y_inter_bis = y_inter
    return x_inter, *y_inter


def deconvolveVariableExpBis(x, y, tau, *args, TMA1=10., TMA2=50.):
    # x (N-array): time axis, can be unregular, if so will be interpolated
    # y (N-array): data to deconvolve
    # tau (N-array): time constants of the exponentials
    # args (N-arrays): other arrays to regrid

    #--first interpolation, on a regular time grid
    step_time_axis = np.min(np.diff(x))
    x_inter, y_inter, tau_inter, *args_inter = interpolate(x, step_time_axis, y, tau, *args, kind='linear')


    #--low-pass filtering 1 (boxcar window)
    #--the data is smoothed
    #T_moving_average_1 = 10. * step_time_axis # as in Erlich and Wendish 2015
    T_moving_average_1 = TMA1 * step_time_axis
    M_boxcar_1 = int( T_moving_average_1 / step_time_axis )
    win = scipy.signal.windows.boxcar(M_boxcar_1)
    y_averaged_1 = scipy.signal.oaconvolve(y_inter, win, mode='same') / np.sum(win)


    #--deconvolution

    #--inflate the data to have a constant tau value
    delta = np.diff(x_inter) / tau_inter[:-1]
    x_pre_deconvolve = np.full(len(delta) + 1, fill_value=0., dtype=np.float64)
    x_pre_deconvolve[1:] += np.cumsum(delta)
    step_deconvolve = np.min(delta)
    x_deconvolve, y_deconvolve = interpolate(x_pre_deconvolve, step_deconvolve, y_averaged_1, kind='quadratic')

    #--deconvolution (exponential window)
    tau_exp = 1. / step_deconvolve
    M_exp = - int(tau_exp * np.log(0.01)) # we suppose that after the exponential has reached 1 % of its value, it is zero
    win = scipy.signal.windows.exponential(M_exp+1, 0, tau_exp, False)
    #win, M_exp = [1., 0.], 1
    y_deconvolved, _ = scipy.signal.deconvolve(y_deconvolve, win)
    y_deconvolved = y_deconvolved * np.sum(win)

    x_deconvolved = x_deconvolve[:-M_exp]

    #--deflate the data to have a constant time axis
    i_equiv = np.nonzero(x_pre_deconvolve <= x_deconvolve[-M_exp-1])
    x_post_deconvolve = x_pre_deconvolve[i_equiv]
    _, y_post_deconvolve = interpolate(x_deconvolved, x_post_deconvolve, y_deconvolved, kind='quadratic')

    #--finalise
    x_final = x_inter[i_equiv]
    tau_final = tau_inter[i_equiv]
    args_final = (arg[i_equiv] for arg in args_inter)


    #--low-pass filtering 2 (boxcar window)
    #--the result is smoothed
    #T_moving_average_2 = 100. * step_time_axis # as in Erlich and Wendish 2015
    T_moving_average_2 = TMA2 * step_time_axis # too high
    M_boxcar_2 = int( T_moving_average_2 / step_time_axis )
    win = scipy.signal.windows.boxcar(M_boxcar_2)
    y_averaged_2 = scipy.signal.oaconvolve(y_post_deconvolve, win, mode='same') / np.sum(win)

    return x_final, y_averaged_2, tau_final, *args_final


def forwardEMA(x, y, tau):
    # x time
    # y data
    alpha = getAlpha(x, tau)

    y_con = y.copy()
    for i, y_val in enumerate(y[1:]):
        y_con[i+1] = alpha[i] * y_val + (1. - alpha[i]) * y_con[i]
    return y_con


def inverseEMA(x, y, tau, WSS):
    # x time
    # y data
    alpha = getAlpha(x, tau)

    y_filtered = sigana.moving_average(y, x, WSS, window='blackman')
    y_delta = np.diff(y_filtered)

    y_decon = np.zeros_like(y_filtered)
    y_decon[0] = y_filtered[0]
    y_decon[1:] = y_filtered[:-1] + y_delta / alpha

    return y_decon


def getAlpha(x, tau):
    alpha = 2. / (tau[:-1] + 1.)
    alpha = 1. - np.exp( - np.diff(x) / np.timedelta64(1, 's') / tau[:-1] )
    return alpha


def boxcar_window(time_centered, window_size_sec):
    window = np.full_like(time_centered, fill_value=1.)
    return window


def hann_window(time_centered, window_size_sec):
    window = 0.5 + 0.5 * np.cos(2. * np.pi * time_centered / window_size_sec)
    return window


def hamming_window(time_centered, window_size_sec):
    window = 0.54 + 0.46 * np.cos(2. * np.pi * time_centered / window_size_sec)
    return window


def blackman_window(time_centered, window_size_sec):
    window = 0.42 + 0.5  * np.cos(2. * np.pi * time_centered / window_size_sec) \
                  - 0.08 * np.cos(4. * np.pi * time_centered / window_size_sec)
    return window


def moving_average(data, time, window_size_sec, window='blackman'):

    window_functions = dict(boxcar=boxcar_window, hann=hann_window, hamming=hamming_window,
                            blackman=blackman_window)
    if window not in window_functions:
        raise ValueError(f"Window {window} is not recognized, "
                          "please use 'boxcar', 'hann', 'hamming' or 'blackman'.")
    window_fun = window_functions[window]

    time = (time - time[0]) / np.timedelta64(1, 's')
    smoothed_data = np.full_like(data, fill_value=np.nan)

    for idx_time, time_value in enumerate(time):
        if np.isscalar(window_size_sec):
            WSS = window_size_sec
        else:
            WSS = window_size_sec[idx_time]

        idx_left = bisect.bisect_left(time, time_value - WSS / 2.)
        idx_right = bisect.bisect_right(time, time_value + WSS / 2., lo=idx_left)
        window_data = data[idx_left:idx_right]
        window_time_centered = time[idx_left:idx_right] - time_value

        weights = window_fun(window_time_centered, WSS)
        weighted_sum = np.dot(weights, window_data)
        sum_of_weights = np.sum(weights)
        smoothed_data[idx_time] = weighted_sum / sum_of_weights

    return smoothed_data


def deconvolveVariableExp(x, y, tau, TMA, TCO):
    # x (N-array): time axis, can be unregular, if so will be interpolated
    # y (N-array): data to deconvolve
    # tau (N-array): time constants of the exponentials


    #--first interpolation, on a regular time grid
    step_time_axis = np.min(np.diff(x))
    x_inter = np.arange(x[0], x[-1], step_time_axis)
    y_inter = scipy.interpolate.interp1d(x, y, kind='quadratic')(x_inter)
    tau_inter = scipy.interpolate.interp1d(x, tau, kind='quadratic')(x_inter)
    tau_inter = np.maximum(tau_inter, np.min(tau))


    #--low-pass filtering 1 (boxcar window)
    #--the data is smoothed
    #T_moving_average_1 = 10. * step_time_axis # as in Erlich and Wendish 2015
    T_moving_average_1 = TMA * step_time_axis
    M_boxcar_1 = int( T_moving_average_1 / step_time_axis )
    win = scipy.signal.windows.boxcar(M_boxcar_1)
    y_averaged_1 = scipy.signal.oaconvolve(y_inter, win, mode='same') / np.sum(win)


    #--deconvolution

    #--inflate the data to have a constant tau value
    delta = np.diff(x_inter) / tau_inter[:-1]
    x_pre_deconvolve = np.full(len(delta) + 1, fill_value=0., dtype=np.float64)
    x_pre_deconvolve[1:] += np.cumsum(delta)
    step_deconvolve = np.min(delta)

    x_deconvolve = np.arange(x_pre_deconvolve[0], x_pre_deconvolve[-1], step_deconvolve)
    y_deconvolve = scipy.interpolate.interp1d(x_pre_deconvolve, y_averaged_1, kind='quadratic')(x_deconvolve)

    #--deconvolution (exponential window)
    tau_exp = 1. / step_deconvolve
    M_exp = - int(tau_exp * np.log(0.01)) # we suppose that after the exponential has reached 1 % of its value, it is zero
    win = scipy.signal.windows.exponential(M_exp+1, 0, tau_exp, False)
    #win, M_exp = [1., 0.], 1
    y_deconvolved, _ = scipy.signal.deconvolve(y_deconvolve, win)
    y_deconvolved = y_deconvolved * np.sum(win)

    x_deconvolved = x_deconvolve[:-M_exp]

    #--deflate the data to have a constant time axis
    i_equiv = np.nonzero(x_pre_deconvolve <= x_deconvolve[-M_exp-1])
    x_post_deconvolve = x_pre_deconvolve[i_equiv]
    y_post_deconvolve = scipy.interpolate.interp1d(x_deconvolved, y_deconvolved, kind='quadratic')(x_post_deconvolve)

    #--finalise
    x_pre_final = x_inter[i_equiv]


    #--low-pass filtering 2 (boxcar window)
    #--the result is smoothed
    #T_moving_average_2 = 100. * step_time_axis # as in Erlich and Wendish 2015
    #T_moving_average_2 = TMA2 * step_time_axis # too high
    #M_boxcar_2 = int( T_moving_average_2 / step_time_axis )
    #win = scipy.signal.windows.boxcar(M_boxcar_2)
    #y_averaged_2 = scipy.signal.oaconvolve(y_post_deconvolve, win, mode='same') / np.sum(win)
    
    freq = np.fft.rfftfreq(len(y_post_deconvolve), step_time_axis)
    i_cut_off = np.nonzero(freq > 1. / TCO)
    y_fourier = np.fft.rfft(y_post_deconvolve)
    y_fourier[i_cut_off] = 0.
    y_averaged_2 = np.fft.irfft(y_fourier)
    
    if x_pre_final.size % 2 == 1: x_pre_final = x_pre_final[1:]
    

    #--finally, remove the interpolated data points (we do not want to create artificial data)
    ind_filter = np.nonzero((x >= x_pre_final[0]) & (x <= x_pre_final[-1]))
    x_final = x[ind_filter]
    y_final = scipy.interpolate.interp1d(x_pre_final, y_averaged_2, kind='quadratic')(x_final)

    return x_final, y_final, ind_filter


def convolveVariableExp(x, y, tau):
    # x (N-array): time axis
    # y (N-array): data to convolve
    # tau (N-array): time constants of the exponentials
    # args (N-arrays): other arrays to regrid

    #--first interpolation, on a regular time grid
    step_time_axis = np.min(np.diff(x))
    x_inter = np.arange(x[0], x[-1], step_time_axis)
    y_inter = scipy.interpolate.interp1d(x, y, kind='quadratic')(x_inter)
    tau_inter = scipy.interpolate.interp1d(x, tau, kind='quadratic')(x_inter)
    tau_inter = np.maximum(tau_inter, np.min(tau))


    #--low-pass filtering 1 (boxcar window)
    #--the data is smoothed
    #T_moving_average_1 = 10. * step_time_axis # as in Erlich and Wendish 2015
    #M_boxcar_1 = int( T_moving_average_1 / step_time_axis )
    #win = scipy.signal.windows.boxcar(M_boxcar_1)
    #y_averaged_1 = scipy.signal.oaconvolve(y_inter, win, mode='same') / np.sum(win)
    y_averaged_1 = y_inter


    #--convolution

    #--inflate the data to have a constant tau value
    delta = np.diff(x_inter) / tau_inter[:-1]
    x_pre_convolve = np.full(len(delta) + 1, fill_value=0., dtype=np.float64)
    x_pre_convolve[1:] += np.cumsum(delta)
    step_convolve = np.min(delta)

    x_convolve = np.arange(x_pre_convolve[0], x_pre_convolve[-1], step_convolve)
    y_convolve = scipy.interpolate.interp1d(x_pre_convolve, y_averaged_1, kind='quadratic')(x_convolve)

    #--convolution (exponential window)
    tau_exp = 1. / step_convolve
    M_exp = - int(tau_exp * np.log(0.01)) # we suppose that after the exponential has reached 1 % of its value, it is zero
    win = scipy.signal.windows.exponential(M_exp+1, 0, tau_exp, False)
    #win, M_exp = [1., 0.], 1
    y_convolved = scipy.signal.oaconvolve(y_convolve, win, mode='valid')
    y_convolved = y_convolved / np.sum(win)

    x_convolved = x_convolve[M_exp:]

    #--deflate the data to have a constant time axis
    i_equiv = np.nonzero((x_pre_convolve >= x_convolve[M_exp]) & (x_pre_convolve <= x_convolve[-1]))
    x_post_convolve = x_pre_convolve[i_equiv]
    y_post_convolve = scipy.interpolate.interp1d(x_convolved, y_convolved, kind='quadratic')(x_post_convolve)

    #--finalise
    x_pre_final = x_inter[i_equiv]


    #--low-pass filtering 2 (boxcar window)
    #--the result is smoothed
    #T_moving_average_2 = 100. * step_time_axis # as in Erlich and Wendish 2015
    #T_moving_average_2 = 50. * step_time_axis # too high
    #M_boxcar_2 = int( T_moving_average_2 / step_time_axis )
    #win = scipy.signal.windows.boxcar(M_boxcar_2)
    #y_averaged_2 = scipy.signal.oaconvolve(y_post_convolve, win, mode='same') / np.sum(win)
    y_averaged_2 = y_post_convolve


    #--finally, remove the interpolated data points (we do not want to create artificial data)
    ind_filter = np.nonzero((x >= x_pre_final[0]) & (x <= x_pre_final[-1]))
    x_final = x[ind_filter]
    y_final = scipy.interpolate.interp1d(x_pre_final, y_averaged_2, kind='quadratic')(x_final)

    return x_final, y_final, ind_filter
