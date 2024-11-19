import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import neurokit2 as nk
import biosppy.signals.ecg as ecg

from joblib import Parallel, delayed

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

def process_ecg(ecg_signal, freq, hrv_columns):
    ecg_signal = ecg_signal.dropna().to_numpy(dtype='float32')
    ts, filtered, rpeaks, template_ts, template, heart_rate_ts, heart_rate = ecg.ecg(signal=ecg_signal, sampling_rate=freq, show=False)
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
    mean_hrt = np.mean(np.array(heart_rate))
    RR_inter = rpeaks[1:] - rpeaks[:-1]
    RR_inter_mean, RR_inter_std = np.mean(RR_inter), np.std(RR_inter)

    features = []
    features += [PR_inter_mean, PR_inter_std]
    features += [PR_seg_mean, PR_seg_std]
    features += [QRS_comp_mean, QRS_comp_std]
    features += [QT_inter_mean, QT_inter_std]
    features += [ST_seg_mean, ST_seg_std]
    features += [QS_seg_mean, QS_seg_std]
    features += [mean_hrt]
    features += [RR_inter_mean, RR_inter_std]

    features = np.array(features)

    cols = ['PRinterm', 'PRinterstd', 'PRsegm', 'PRsegstd', 'QRSmean', 'QRSstd', 'QTinterm', \
        'QTinterstd', 'STsegm', 'STsegstd', 'QSsegm', 'QSsegstd', 'HeartRatem', 'RRmean', 'RRstd']+hrv_columns

    try:
        df_sig, info = nk.ecg_process(filtered, sampling_rate=300)
        hrv_features = np.apply_along_axis(unroll_list, arr=np.array(nk.ecg_intervalrelated(df_sig, sampling_rate=300)), axis=0)
        hrv_features[np.logical_or(hrv_features == np.inf, hrv_features == -np.inf)] = np.nan
        features_concat = np.concatenate([features, hrv_features])
        df = pd.DataFrame(features_concat[None, :], columns=cols)
    except Exception:
        df = pd.DataFrame(np.append(features, [np.nan] * (len(cols) - features.shape[0]))[None, :], columns=cols)

    return df
    
def process_all_ecg(signals, freq=300, n_jobs=-1):
    ts, filtered, rpeaks, template_ts, template, heart_rate_ts, heart_rate = ecg.ecg(signal=signals.iloc[0, :].dropna().to_numpy(dtype='float32'), sampling_rate=freq, show=False)
    df, info = nk.ecg_process(filtered, sampling_rate=300)
    hrv_columns = nk.ecg_intervalrelated(df, sampling_rate=300).columns
    hrv_columns = hrv_columns.to_list()
    
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(process_ecg)(signals.iloc[i, :], freq, hrv_columns) for i in range(signals.shape[0]))
    
    if results:
        return pd.concat(results, axis=0, ignore_index=True)
    else:
        return None
    
data_train = pd.read_csv('train.csv', header=0, index_col='id')

X = data_train.drop(columns=['y'])
y = data_train.loc[:, 'y']

feature_df = process_all_ecg(X)
feature_df = feature_df.dropna(axis=1, how='all')
feature_df.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
extracted_df = pd.concat([y, feature_df], ignore_index=True, axis=1)

extracted_df.to_csv('features_extracted.csv')