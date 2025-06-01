from scipy.io import loadmat
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import mne
import pandas as pd

# Directory Variables
DATA_DIR = "/Users/alejandro/VSCodeProjects/Data-Collection/raw_data/competitionData/train/"
SAVE_DIR = "./processed_data/"

# TENTATIVE MAPPING:
# Mapping will depend on our final decision on which area we want to extract
# 1. We need the name of the channel we give it (at least I think we name the channels)
# 2. We find the corresponding channel in the stanford data set
FLEX2STANFORD = {
    'AF3': 125,
    'AF4': 120,
    'F3': 122
}

# The channels we want to extract
EXTRACT = ['AF3', 'AF4', 'F3']

def extract_channels(data, channel_map, channels):
    
    """
    Extract corresponding emotiv channel data from stanford dataset
    
    input:
    data: The dataset from the given file
    channel_map: the mapping of channels
    channels: the channels we want
    
    output:
    A (number of trials) x (number of extracted channels) array
    """    
    
    indices = [channel_map[ch] for ch in channels]
    extracted_data = []

    for trial in data:
        selected_channels = trial[:, indices]
        extracted_data.append(selected_channels)

    return extracted_data


def zscore_by_block(data_list, block_ids):
    """
    normalizes the EEG data to minimize the side effects of using the EEG for too long
    
    input: 
    data_list: extracted data set
    block_ids: the trials that are done together (from original dataset)
    
    output:
    normalized data    
    
    """
    block_data = {}
    for trial, block in zip(data_list, block_ids.flatten()):
        block_data.setdefault(block, []).append(trial)

    normalized_data = []
    for trial, block in zip(data_list, block_ids.flatten()):
        stacked = np.vstack(block_data[block])
        scaler = StandardScaler()
        scaler.fit(stacked)
        normalized_data.append(scaler.transform(trial))
    
    return normalized_data

def save_eeg_as_edf(eeg_data, sfreq, channel_names, file_path):
    """
    saves the eeg data extracted as a edf file (emotiv most likely outputs edf files)
    
    input: 
    eeg_data: the eeg_data
    sfreq: the sampling frequency of the data (from data set is 20ms so sampling freq should be 50)
    channel_names: the name of the channel for the metadata
    file_path: location to save it
    
    output: 
    n/a
    """
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    raw.export(file_path, fmt='edf', overwrite=True)
    print(f"Saved EDF file to {file_path}")
    return raw
    
def save_eeg_as_csv(raw_data, channel_names, file_path):
    """
    saves the eeg data extracted as a csv file
    
    input: 
    raw_data: the raw eeg data
    channel_names: the name of the channel for the metadata
    file_path: location to save it
    
    output: 
    n/a
    """
    data, times = raw_data[:, :]
    df = pd.DataFrame(data.T, columns=channel_names)
    df.insert(0, 'time', times)  # Insert time as first column
    df.to_csv(file_path, index=False)
    print(f"Saved CSV file to {file_path}")

def save_clean_trials(trials, sentence_text, save_dir, channel_names):
    """
    saves the trials into designated folders
    
    input:
    trials: the list of data per trial (each index of trials is a dataset)
    sentence_text: list of sentences (corresponds to the trials)
    save_dir: directory to save in
    channel_names: the names of the channels
    
    output:
    n/a
    
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, trial_data in enumerate(trials):
        fname = f"trial_{i}.edf"
        edf_file_path = os.path.join(save_dir, fname)
        sfreq = 50.0
        raw_data = save_eeg_as_edf(trial_data.T, sfreq, channel_names, edf_file_path)

        csv_file_path = os.path.join(save_dir, f"trial_{i}.csv")
        save_eeg_as_csv(raw_data, channel_names, csv_file_path)

        with open(os.path.join(save_dir, f"trial_{i}_label.txt"), 'w') as f:
            sentence = sentence_text[i]
            f.write(sentence)


def preprocess_file(filepath, flex2_channels, stanford_to_flex2, save_dir):
    """
    does the data extraction and processing pipeline
    
    input:
    filepath: the input file for extraction
    flex2_channels: the channels to extract
    stanford_to_flex2: the mapping of flex to standford
    save_dir: the directory to save to
    """
    
    mat = loadmat(filepath)
    data = mat['spikePow']  # or mat['tx1']
    block_ids = mat['blockIdx']
    sentence_text = mat['sentenceText']
    
    data = data[0]
    extracted = extract_channels(data, stanford_to_flex2, flex2_channels)
    normalized = zscore_by_block(extracted, block_ids)
    save_clean_trials(normalized, sentence_text, save_dir, flex2_channels)


def main():

    files = os.listdir(DATA_DIR)
    for file in files:
        file_dir = os.path.join(SAVE_DIR, file)
        os.makedirs(file_dir, exist_ok=True)
        data = loadmat(os.path.join(DATA_DIR, files[0]))
        file_name = os.path.join(DATA_DIR, files[0])
        preprocess_file(file_name, EXTRACT, FLEX2STANFORD, file_dir)
    
if __name__ == "__main__":
    main()