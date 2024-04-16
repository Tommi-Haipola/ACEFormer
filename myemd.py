import numpy as np

import scipy.signal as signal
from scipy import interpolate
from scipy.signal import argrelextrema

# correlation points of upper and lower envelope
def max_min_peaks(data, point_type:str = "emd"):
    assert(point_type in ["emd", "ext", "mid"])
    emd_id = {"emd": 1, "ext": 2, "mid": 3}
    point_type = emd_id[point_type]
    
    point_num = np.size(data)
    peaks_max = signal.argrelextrema(data, np.greater)[0]
    peaks_min = signal.argrelextrema(data, np.less)[0]
    
    if point_type > 1:
        peaks_max = np.concatenate(([0], peaks_max, [point_num-1]))
        peaks_min = np.concatenate(([0], peaks_min, [point_num-1]))
    
    if point_type > 2:
        _tmp = np.sort(np.concatenate(([0], peaks_max, peaks_min, [point_num-1])))
        _tmp = np.delete(_tmp, np.where(_tmp[1:] == _tmp[:-1]))
        mid_point = []
        for i in range(_tmp.shape[0] - 1):
            mid_point.append(int((_tmp[i]+_tmp[i+1]) / 2))
        
        peaks_max = np.sort(np.concatenate((peaks_max, mid_point)))
        peaks_min = np.sort(np.concatenate((peaks_min, mid_point)))
        
    peaks_max = np.delete(peaks_max, np.where(peaks_max[1:] == peaks_max[:-1]))
    peaks_min = np.delete(peaks_min, np.where(peaks_min[1:] == peaks_min[:-1]))
    
    return peaks_max, peaks_min

# cubic interpolation sampling for less than 4 points
def cubic_spline_3pts(x, y, T):
    """
    Apparently scipy.interpolate.interp1d does not support
    cubic spline for less than 4 points.
    """
    x0, x1, x2 = x
    y0, y1, y2 = y

    x1x0, x2x1 = x1 - x0, x2 - x1
    y1y0, y2y1 = y1 - y0, y2 - y1
    _x1x0, _x2x1 = 1.0 / x1x0, 1.0 / x2x1

    m11, m12, m13 = 2 * _x1x0, _x1x0, 0
    m21, m22, m23 = _x1x0, 2.0 * (_x1x0 + _x2x1), _x2x1
    m31, m32, m33 = 0, _x2x1, 2.0 * _x2x1

    v1 = 3 * y1y0 * _x1x0 * _x1x0
    v3 = 3 * y2y1 * _x2x1 * _x2x1
    v2 = v1 + v3

    M = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
    v = np.array([v1, v2, v3]).T
    k = np.array(np.linalg.inv(M).dot(v))

    a1 = k[0] * x1x0 - y1y0
    b1 = -k[1] * x1x0 + y1y0
    a2 = k[1] * x2x1 - y2y1
    b2 = -k[2] * x2x1 + y2y1

    t = T
    t1 = (T[np.r_[T < x1]] - x0) / x1x0
    t2 = (T[np.r_[T >= x1]] - x1) / x2x1
    t11, t22 = 1.0 - t1, 1.0 - t2

    q1 = t11 * y0 + t1 * y1 + t1 * t11 * (a1 * t11 + b1 * t1)
    q2 = t22 * y1 + t2 * y2 + t2 * t22 * (a2 * t22 + b2 * t2)
    q = np.append(q1, q2)

    return t, q

# cubic interpolation sampling for more than 3 points
def envelopes(data, peaks_max, peaks_min):
    point_num = len(data)
    
    if len(peaks_max) > 3:
        inp_max = interpolate.splrep(peaks_max, data[peaks_max], k=3)
        fit_max = interpolate.splev(np.arange(point_num), inp_max)
    else:
        _, fit_max = cubic_spline_3pts(peaks_max, data[peaks_max], np.arange(len(data)))

    if len(peaks_min) > 3:
        inp_min = interpolate.splrep(peaks_min, data[peaks_min], k=3)
        fit_min = interpolate.splev(np.arange(point_num), inp_min)
    else:
        _, fit_min = cubic_spline_3pts(peaks_min, data[peaks_min], np.arange(len(data)))

    return fit_max, fit_min

# determine if modal decomposition is over
def imf_judge(x: np.array, y: np.array):    
    """
    x: The sequence after decomposition.
    y: The sequence before decomposition.
    """
    if (y.max() - y.min()) != 0 and ((x - y)**2).sum() / (y.max() - y.min()) < 0.001:
        return True
    
    if not np.any(x == 0) and (((x - y) / x)**2).sum() < 0.2:
        return True
    
    if (y**2).sum() != 0 and ((x - y)**2).sum() / (y**2).sum() < 0.2:
        return True
    
    return False

# Empirical Mode Decomposition (EMD)
def emd(signal):
    origin_signal = signal.copy()
    # extrema point
    emd_peaks_max, emd_peaks_min = max_min_peaks(signal, "emd")

    # emd
    std_continue, old_std = 0, 0.0
    # envelope line
    emd_up_envelopes, emd_down_envelopes = 0, 0
    continue_time = 511

    while True:
        # number of extreme points
        if len(emd_peaks_max) < 3 or len(emd_peaks_min) < 3:
            break

        fit_max, fit_min = envelopes(signal, emd_peaks_max, emd_peaks_min)
        emd_up_envelopes, emd_down_envelopes = emd_up_envelopes + fit_max, emd_down_envelopes + fit_min
        signal_old = signal.copy()
        signal = signal - (fit_max + fit_min) / 2

        emd_peaks_max, emd_peaks_min = max_min_peaks(signal, "emd")
        pass_zero = np.sum(signal[:-1] * signal[1:] < 0)

        std = abs((fit_max + fit_min) / 2 / origin_signal).mean()
        std_continue = (std_continue << 1) & continue_time
        std_continue += 1 if abs(old_std - std) < 1e-6 else 0
        old_std = std

        if (abs(pass_zero - len(emd_peaks_max) - len(emd_peaks_min)) < 2) or imf_judge(signal, signal_old) or std_continue == continue_time:
            break

    if isinstance(emd_up_envelopes, int) and isinstance(emd_down_envelopes, int):
        return signal, signal
    return emd_up_envelopes, emd_down_envelopes

# Extrema Empirical Mode Decomposition (extemd)
def extemd(signal):
    origin_signal = signal.copy()
    # extrema point
    emd_peaks_max, emd_peaks_min = max_min_peaks(signal, "ext")

    # emd
    std_continue, old_std = 0, 0.0
    # envelope line
    emd_up_envelopes, emd_down_envelopes = 0, 0
    continue_time = 511

    while True:
        # number of extreme points
        if len(emd_peaks_max) < 3 or len(emd_peaks_min) < 3:
            break

        fit_max, fit_min = envelopes(signal, emd_peaks_max, emd_peaks_min)
        emd_up_envelopes, emd_down_envelopes = emd_up_envelopes + fit_max, emd_down_envelopes + fit_min
        signal_old = signal.copy()
        signal = signal - (fit_max + fit_min) / 2

        emd_peaks_max, emd_peaks_min = max_min_peaks(signal, "ext")
        pass_zero = np.sum(signal[:-1] * signal[1:] < 0)

        std = abs((fit_max + fit_min) / 2 / origin_signal).mean()
        std_continue = (std_continue << 1) & continue_time
        std_continue += 1 if abs(old_std - std) < 1e-6 else 0
        old_std = std

        if (abs(pass_zero - len(emd_peaks_max) - len(emd_peaks_min)) < 2) or imf_judge(signal, signal_old) or std_continue == continue_time:
            break

    if isinstance(emd_up_envelopes, int) and isinstance(emd_down_envelopes, int):
        return signal, signal
    return emd_up_envelopes, emd_down_envelopes

# Aliased Complete Ensemble Empirical Mode Decomposition (ACEEMD)
def aceemd(extsignal, midsignal, alpha = 0.5):
    origin_signal = extsignal.copy()
    # extrema point
    ext_peaks_max, ext_peaks_min = max_min_peaks(extsignal, "ext")
    mid_peaks_max, mid_peaks_min = max_min_peaks(midsignal, "mid")

    # emd
    std_continue, old_std = 0, 0.0
    # envelope line
    ext_up_envelopes, ext_down_envelopes = 0, 0
    mid_up_envelopes, mid_down_envelopes = 0, 0
    continue_time = 511

    while True:
        # 极值点个数
        if len(ext_peaks_max) < 3 or len(ext_peaks_min) < 3:
            break

        # 中点点集
        fit_max, fit_min = envelopes(midsignal, mid_peaks_max, mid_peaks_min)
        mid_up_envelopes, mid_down_envelopes = mid_up_envelopes + fit_max, mid_down_envelopes + fit_min
        midsignal = midsignal - (fit_max + fit_min) / 2

        mid_peaks_max, mid_peaks_min = max_min_peaks(midsignal, "mid")

        # 端点点集
        fit_max, fit_min = envelopes(extsignal, ext_peaks_max, ext_peaks_min)
        ext_up_envelopes, ext_down_envelopes = ext_up_envelopes + fit_max, ext_down_envelopes + fit_min
        extsignal_old = extsignal.copy()
        extsignal = extsignal - (fit_max + fit_min) / 2

        ext_peaks_max, ext_peaks_min = max_min_peaks(extsignal, "ext")

        # 判断循环
        pass_zero = np.sum(extsignal[:-1] * extsignal[1:] < 0)

        std = abs((fit_max + fit_min) / 2 / origin_signal).mean()
        std_continue = (std_continue << 1) & continue_time
        std_continue += 1 if abs(old_std - std) < 1e-6 else 0
        old_std = std

        if (abs(pass_zero - len(ext_peaks_max) - len(ext_peaks_min)) < 2) or imf_judge(extsignal, extsignal_old) or std_continue == continue_time:
            break

    if isinstance(ext_up_envelopes, int) and isinstance(mid_up_envelopes, int) and isinstance(ext_down_envelopes, int) and isinstance(mid_down_envelopes, int):
        return extsignal, midsignal
    return ext_up_envelopes * (1-alpha) + mid_up_envelopes * alpha, ext_down_envelopes * (1-alpha) + mid_down_envelopes * alpha

# Assuming noise is a sample of gaussian noise
def snr(data, noise):
  """
  This function calculates the SNR (Signal-to-Noise Ratio) by adding the first 
  Intrinsic Mode Function (IMF) of the noise to the data.

  Args:
      data: A numpy array representing the signal.

  Returns:
      A numpy array representing the data with noise added.
  """

  # Perform EMD on the noise to get IMFs
  imfs = emd(noise)  # Assuming emd is defined earlier

  # Extract the first IMF (IMF1)
  if isinstance(imfs, tuple):
      imf1 = imfs[0]
  else:
      imf1 = imfs

  # Add IMF1 to the data
  data_with_noise = data + imf1

  return data_with_noise


#calculate ACEEMD from base signal (includes snr)
def ACEEMD_Base(source_data,imf_times=40,emd_type=0,alpha=0.5):

    #  Gaussian noise
    noise_list = []
    win_len = source_data.shape[-2]
    for _ in range(imf_times // 2):
        noise = np.random.randn(win_len)
        n_up_envelopes, n_down_envelopes =  emd(noise)
        noise_list.append((n_up_envelopes + n_down_envelopes) / 2 / np.std(noise))

        n_up_envelopes, n_down_envelopes = emd(-noise)
        noise_list.append((n_up_envelopes + n_down_envelopes) / 2 / np.std(-noise))

    # emd process
    emd_result = []
    for s in range(len(source_data)):
        emd_tmp = []
        for d in range(len(source_data[s].T)):
            _data = (source_data[s].T)[d]
            # envelope line
            up_list, down_list = [], []
            # iceemd & eceemd
            if emd_type == 1 or emd_type == 2:
                for noise in noise_list:
                    _emd_data = _data.copy() + noise * snr(_data, noise)
                    up, down = emd(_emd_data) if emd_type == 1 else extemd(_emd_data)

                    up_list.append(up)
                    down_list.append(down)
            # aceemd
            else:
                for i in range(imf_times // 2):
                    _exemd_data = _data.copy() + noise_list[2*i] * snr(_data, noise_list[2*i])
                    _acemd_data = _data.copy() + noise_list[2*i+1] * snr(_data, noise_list[2*i+1])
                    up, down = aceemd(_exemd_data, _acemd_data, alpha)

                    up_list.append(up)
                    down_list.append(down)
            # denoise
            emd_tmp.append((np.array(up_list).mean(axis=0) + np.array(down_list).mean(axis=0)) / 2)
        emd_result.append(np.array(emd_tmp).T)
    return emd_result