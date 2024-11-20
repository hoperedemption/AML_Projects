import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from joblib import Parallel, delayed
import scipy
from scipy.signal import welch

import heartpy as hp 

import neurokit2 as nk
import biosppy.signals.ecg as ecg

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from astropy.timeseries import LombScargle
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, get_geometrical_features, get_poincare_plot_features, get_csi_cvi_features, get_sampen

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
    
    # Extract HP Features
    wd, m = hp.process(filtered_scaled_data, freq, bpmmin=0, bpmmax=300)
    del m['sdnn']
    del m['sdsd']
    del m['rmssd']
    del m['sd1']
    del m['sd2']
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
    
    all_features_non_hrv = pd.concat([features_base_df, hp_features, td_features, gm_features, pt_features, csi_features, smp_features], axis=1)
    
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
    
def process_all_ecg(signals, freq=300, n_jobs=-1):    
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(process_ecg)(signals.iloc[i, :], freq) for i in range(signals.shape[0]))
    
    if results:
        return pd.concat(results, axis=0, ignore_index=True)
    else:
        return None
    
data_train = pd.read_csv('train.csv', header=0, index_col='id')

X = data_train.drop(columns=['y'])
y = data_train.loc[:, 'y']

feature_df = process_all_ecg(X)
feature_df = feature_df.dropna(axis=1, how='all')
feature_cols = feature_df.columns
feature_df.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
extracted_df = pd.concat([y, feature_df], ignore_index=True, axis=1)
extracted_df.columns = ['y']+feature_cols.to_list()
extracted_df = feature_df
extracted_df.to_csv('train_features.csv')