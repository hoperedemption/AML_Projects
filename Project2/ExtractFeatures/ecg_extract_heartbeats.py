import neurokit2 as nk
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import biosppy.signals.ecg as ecg
from joblib import Parallel, delayed
import time

data = pd.read_csv('test.csv', index_col='id')
# y = data['y']
# data.drop(columns=['y'], inplace=True)

def extract_heartbeat(signal, freq):
    signal = signal[~np.isnan(signal)]
    r_peaks = ecg.engzee_segmenter(signal, freq)['rpeaks']
    if len(r_peaks) >= 2:
        beats = ecg.extract_heartbeats(signal, r_peaks, freq)['templates']

        if len(beats) != 0:
            return np.mean(beats, axis=0) 
    
    return None 

def process_all_signals(signals, freq=300, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(extract_heartbeat)(signals[i], freq) for i in range(signals.shape[0]))
    
    if results:
        mus = results
        mask = [result is not None for result in results]
        mus = [results[i] for i in range(len(mask)) if mask[i]]
        return np.array(mus), np.array(mask)
    else:
        return None

data = data.to_numpy()
mus, mask = process_all_signals(data)  
# y = y[mask]
mus = pd.DataFrame(mus)
mus.reset_index(drop=True, inplace=True)
# y.reset_index(drop=True, inplace=True)
# transformed_data = pd.concat([y, mus], axis=1)
mus.to_csv('heartbeats_test.csv')