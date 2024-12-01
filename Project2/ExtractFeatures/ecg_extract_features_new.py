import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyhrv.frequency_domain
import pyhrv.nonlinear
import pyhrv.time_domain
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from joblib import Parallel, delayed
import scipy
from scipy.signal import welch
from collections.abc import Iterable

import matplotlib

import heartpy as hp 

import pyhrv

import neurokit2 as nk
import biosppy.signals.ecg as ecg

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from astropy.timeseries import LombScargle
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, get_geometrical_features, get_poincare_plot_features, get_csi_cvi_features, get_sampen

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true', help='Train or Test set', default=False)
args = parser.parse_args()

SAMPLE_FREQ = 300

def seq_to_signal(seq):
    return np.argwhere(seq == 1)

def process_interval(first_signal, second_signal):
    if first_signal.shape[0] == 0 or second_signal.shape[0] == 0:
        return np.nan, np.nan 
    
    if first_signal.shape[0] > second_signal.shape[0]:
        reduce_axis = 0
    else:
        reduce_axis = 1
    
    diff = (second_signal.squeeze(-1)[None, :] - first_signal.squeeze(-1)[:, None]).astype(np.float32)
    diff[diff < 0] = np.inf 
    min_diff =  np.min(diff, axis=reduce_axis)
    return min_diff[min_diff != np.inf]

def interv_mstd(intervals):
    return np.mean(intervals), np.std(intervals)

def unroll_list(x):
    if isinstance(x, np.ndarray):
        while isinstance(x, np.ndarray) and len(x) > 0:
            x = x[0]
        return x if np.isscalar(x) else np.nan
    return x if np.isscalar(x) else np.nan

def signal_features(filtered, rpeaks):
    frequencies, psd = welch(filtered, fs=300, nperseg=None)
    dom_freq_id = np.argmax(psd)
    dominating_frequency = frequencies[dom_freq_id]
    psd_mode = psd[dom_freq_id]
    df_welch = frequencies[1] - frequencies[0]
    e_welch = np.sum(psd) * df_welch
    beats = ecg.extract_heartbeats(filtered, rpeaks, 300)['templates']
    mu = np.mean(beats, axis=0)
    mu_m, mu_std, mu_sk, mu_krt = np.mean(mu), np.std(mu), scipy.stats.skew(mu), scipy.stats.kurtosis(mu)
    return dominating_frequency, psd_mode, e_welch, mu_m, mu_std, mu_sk, mu_krt
   
def extract_general_hrv_features(interpolated_nn_intervals, rpeaks, filtered_scaled_data, freq): 
    time_domain_features = pyhrv.time_domain.time_domain(interpolated_nn_intervals, rpeaks, filtered_scaled_data, freq)
    welchs_features = pyhrv.frequency_domain.welch_psd(nni=interpolated_nn_intervals, rpeaks=rpeaks)
    lombd_features = pyhrv.frequency_domain.lomb_psd(nni=interpolated_nn_intervals, rpeaks=rpeaks)
    nonlinear_freq_features = pyhrv.nonlinear.nonlinear(nni=interpolated_nn_intervals, rpeaks=rpeaks)
    result_dict = dict(time_domain_features)
    result_dict.update(dict(welchs_features))
    result_dict.update(dict(lombd_features))
    result_dict.update(dict(nonlinear_freq_features))
    return result_dict

def process_ecg(ecg_signal, freq):
    ecg_signal = ecg_signal.dropna().to_numpy(dtype='float32')
    filtered_rm_baseline_wd = hp.remove_baseline_wander(ecg_signal, freq)
    filtered_scaled_data = hp.scale_data(filtered_rm_baseline_wd)

    ts, filtered, rpeaks, template_ts, template, heart_rate_ts, heart_rate = ecg.ecg(signal=filtered_scaled_data, sampling_rate=freq, show=False)
    signals, waves = nk.ecg_delineate(filtered,
                                        rpeaks, 
                                        sampling_rate=freq, 
                                        method="dwt", 
                                        show=False)

    ECG_P_Peaks, ECG_P_Onsets, ECG_P_Offsets, ECG_Q_Peaks, ECG_R_Onsets, ECG_R_Offsets, ECG_S_Peaks, ECG_T_Peaks, ECG_T_Onsets, ECG_T_Offsets = \
        seq_to_signal(signals['ECG_P_Peaks']), seq_to_signal(signals['ECG_P_Onsets']), seq_to_signal(signals['ECG_P_Offsets']), seq_to_signal(signals['ECG_Q_Peaks']), \
            seq_to_signal(signals['ECG_R_Onsets']), seq_to_signal(signals['ECG_R_Offsets']), seq_to_signal(signals['ECG_S_Peaks']), seq_to_signal(signals['ECG_T_Peaks']), \
                seq_to_signal(signals['ECG_T_Onsets']), seq_to_signal(signals['ECG_T_Offsets'])
    # interval based features
    PR_inter_mean, PR_inter_std = interv_mstd(process_interval(ECG_P_Onsets, ECG_R_Onsets))
    PR_seg_mean, PR_seg_std = interv_mstd(process_interval(ECG_P_Offsets, ECG_R_Onsets))
    QRS_comp_mean, QRS_comp_std = interv_mstd(process_interval(ECG_R_Onsets, ECG_R_Offsets))
    QT_inter_mean, QT_inter_std = interv_mstd(process_interval(ECG_R_Onsets, ECG_T_Offsets))
    ST_seg_mean, ST_seg_std = interv_mstd(process_interval(ECG_R_Offsets, ECG_T_Onsets))
    QS_seg_mean, QS_seg_std = interv_mstd(process_interval(ECG_Q_Peaks, ECG_S_Peaks))
    PT_inter_mean, PT_inter_std = interv_mstd(process_interval(ECG_P_Peaks, ECG_T_Peaks))
    mean_hrt = np.mean(np.array(heart_rate))
    RR_inter = rpeaks[1:] - rpeaks[:-1] # these are our raw RR intervals
    RR_inter_mean, RR_inter_std = np.mean(RR_inter), np.std(RR_inter)
    domfreq, psdmode, ewelch, mum, mustd, musk, mukrt = signal_features(filtered, rpeaks)
    
    # This remove outliers from signal
    
    RR_inter = RR_inter.astype(np.float32)
    outliers_mask = IsolationForest(contamination=0.1).fit_predict(RR_inter.reshape(-1, 1))
    RR_inter[outliers_mask.flatten() == -1] = np.nan
    # This replace outliers nan values with linear interpolation --> This is what we will use instead of the rr intervals
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=RR_inter, 
                                                    interpolation_method="linear")
    
    # This remove ectopic beats from signal
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik", verbose='False')
    # This replace ectopic beats nan values with linear interpolation --> This is what we will use instead of the nn intervals
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
    interpolated_nn_intervals =  np.array(interpolated_nn_intervals)
    interpolated_nn_intervals = interpolated_nn_intervals[~np.isnan(interpolated_nn_intervals)]

    # Extract Base Features
    features = []
    features += [PR_inter_mean, PR_inter_std]
    features += [PR_seg_mean, PR_seg_std]
    features += [QRS_comp_mean, QRS_comp_std]
    features += [QT_inter_mean, QT_inter_std]
    features += [ST_seg_mean, ST_seg_std]
    features += [QS_seg_mean, QS_seg_std]
    features += [PT_inter_mean, PT_inter_std]
    features += [mean_hrt]
    features += [RR_inter_mean, RR_inter_std]
    features += [domfreq, psdmode, ewelch, mum, mustd, musk, mukrt]

    len_features = len(features)
    features = np.array(features).reshape(1, len_features)
    
    base_cols = ['PRinterm', 'PRinterstd', 'PRsegm', 'PRsegstd', 'QRSmean', 'QRSstd', 'QTinterm', \
        'QTinterstd', 'STsegm', 'STsegstd', 'QSsegm', 'QSsegstd', 'PTinterm', 'PTinterstd', 'HeartRatem', 'RRmean', 'RRstd', \
            'DOMFREQ', 'PSDMODE', 'EWELCH', 'MUM', 'MUSTD', 'MUSK', 'MUKRT']
    
    features_base_df = pd.DataFrame(features, columns=base_cols, index=[0])
    
    # general hrv features
    dict_hrv_specific = extract_general_hrv_features(interpolated_nn_intervals, rpeaks, filtered_scaled_data, freq)
    
    # filter out the features    
    figure_or_str_columns = [key for key in dict_hrv_specific.keys() if (isinstance(dict_hrv_specific[key], matplotlib.pyplot.Figure) or isinstance(dict_hrv_specific[key], str) or dict_hrv_specific[key] is None)]

    for key in figure_or_str_columns:
        del dict_hrv_specific[key]
        
    range_columns = [key for key in dict_hrv_specific.keys() if isinstance(dict_hrv_specific[key], range)]

    for key in range_columns:
        dict_hrv_specific[key+'min'] = dict_hrv_specific[key][0]
        dict_hrv_specific[key+'max'] = dict_hrv_specific[key][-1]
        del dict_hrv_specific[key]
        
    return_tuple_columns = [key for key in dict_hrv_specific.keys() if type(dict_hrv_specific[key]).__name__ == "ReturnTuple"]

    for key in return_tuple_columns:
        ulf, vlf, lf, hf = dict_hrv_specific[key]
        dict_hrv_specific[key + 'vlf' + '0'], dict_hrv_specific[key + 'vlf' + '1'] = vlf[0], vlf[1]
        dict_hrv_specific[key + 'lf' + '0'], dict_hrv_specific[key + 'lf' + '1'] = lf[0], lf[1]
        dict_hrv_specific[key + 'hf' + '0'], dict_hrv_specific[key + 'hf' + '1'] = hf[0], hf[1]
        del dict_hrv_specific[key]

    iterable_columns = [key for key in dict_hrv_specific.keys() if isinstance(dict_hrv_specific[key], Iterable)]

    for key in iterable_columns:
        for i, element in enumerate(dict_hrv_specific[key]):
                dict_hrv_specific[key+str(i)] = element
        del dict_hrv_specific[key]
        
    hrv_specific_features = pd.DataFrame(dict_hrv_specific, index=[0])
    
    # Extract HP Features
    wd, m = hp.process(filtered_scaled_data, freq, bpmmin=0, bpmmax=300)
    hp_features = pd.DataFrame(m, index=[0])
    
    # Extract time domain Features
    time_domain_features = get_time_domain_features(interpolated_nn_intervals)
    td_features = pd.DataFrame(time_domain_features, index=[0])
    
    # Extract geometrics Features
    geometrical_features = get_geometrical_features(interpolated_nn_intervals)
    gm_features = pd.DataFrame(geometrical_features, index=[0])
    
    # Extract Point Care Features
    pointcare_features = get_poincare_plot_features(interpolated_nn_intervals)
    pt_features = pd.DataFrame(pointcare_features, index=[0])
    
    # Extract CSI Features
    csi_cvi_features = get_csi_cvi_features(interpolated_nn_intervals)
    csi_features = pd.DataFrame(csi_cvi_features, index=[0])
    
    # Extract Sampen Features
    sampen = get_sampen(interpolated_nn_intervals)
    smp_features = pd.DataFrame(sampen, index=[0])
    
    # Remove duplicate columns
    all_features_non_hrv = pd.concat([features_base_df, hp_features, td_features, gm_features, pt_features, csi_features, smp_features, hrv_specific_features], axis=1)
    mask = ~all_features_non_hrv.columns.duplicated()
    all_features_non_hrv = all_features_non_hrv.loc[:, mask]
    
    try:
        df_sig, info = nk.ecg_process(filtered_scaled_data, sampling_rate=freq)
        interval_related = nk.ecg_intervalrelated(df_sig, sampling_rate=freq)
        hrv_columns = interval_related.columns
        hrv_features = np.apply_along_axis(unroll_list, arr=np.array(interval_related), axis=0)
        hrv_features[np.logical_or(hrv_features == np.inf, hrv_features == -np.inf)] = np.nan
        hrv_features = pd.DataFrame(hrv_features.reshape(1, hrv_features.shape[0]), columns=hrv_columns, index=[0])
        features_concat = pd.concat([all_features_non_hrv, hrv_features], axis=1)
    except Exception as e:
        features_concat = all_features_non_hrv


    return features_concat

def divide_into_ranges(n, step):
    return [(start, min(start + step, n)) for start in range(0, n, step)]
    
def process_all_ecg(signals, freq=SAMPLE_FREQ, n_jobs=-1):    
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(process_ecg)(signals.iloc[i, :], freq) for i in range(signals.shape[0]))
    
    if results:
        return pd.concat(results, axis=0, ignore_index=True)
    else:
        return None
    
if args.train:
    data_train = pd.read_csv('train.csv', header=0, index_col='id')

    X = data_train.drop(columns=['y'])
    y = data_train.loc[:, 'y']
    
    partials = []

    for run in divide_into_ranges(X.shape[0], 1000):    
        start, end = run[0], run[-1]
        X_, y_ = X.iloc[start:end, :], y.iloc[start:end]
        partial_feature_df = process_all_ecg(X_)
        partial_feature_df = partial_feature_df.dropna(axis=1, how='all')
        partial_feature_df.reset_index(drop=True, inplace=True)
        y_.reset_index(drop=True, inplace=True)
        extracted_df = pd.concat([y_, partial_feature_df], ignore_index=True, axis=1)
        extracted_df.columns = ['y']+partial_feature_df.columns.to_list()
        partials.append(extracted_df)

    resulting_features = pd.concat(partials, axis=0, ignore_index=True)
    resulting_features.to_csv('train_features.csv', index_label='id')
else:
    print('TEST')
    data_test = pd.read_csv('test.csv', header=0, index_col='id')
    
    X = data_test
    
    partials = []
    
    for run in divide_into_ranges(X.shape[0], 1000):
        start, end = run[0], run[-1]
        X_ = X.iloc[start:end, :]
        partial_feature_df = process_all_ecg(X_)
        partial_feature_df = partial_feature_df.dropna(axis=1, how='all')
        partials.append(partial_feature_df)
        
    resulting_features = pd.concat(partials, axis=0, ignore_index=True)
    resulting_features.to_csv('test_features.csv', index_label='id')

