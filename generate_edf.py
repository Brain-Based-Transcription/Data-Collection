import numpy as np
import pyedflib
import os

def generate_edf(duration):
    """
    generate the edf file
    """
    # Parameters
    filedir = "./generated_edf"
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    filename = f"./generated_edf/{duration}.edf"
    n_channels = 1
    sample_rate = 256  # Hz

    # Calculate integer samples (closest match)
    n_samples = int(round(sample_rate * duration))
    actual_duration = n_samples / sample_rate

    # Time vector
    time = np.arange(n_samples) / sample_rate

    # Example EEG: 10 Hz sine wave
    signal = 1000 * np.sin(2 * np.pi * 10 * time)

    # Channel metadata
    channel_info = [{
        'label': 'EEG Fpz-Cz',
        'dimension': 'uV',
        'sample_frequency': sample_rate,
        'physical_min': -2000,
        'physical_max': 2000,
        'digital_min': -32768,
        'digital_max': 32767,
        'transducer': '',
        'prefilter': ''
    }]

    # Write EDF
    with pyedflib.EdfWriter(filename, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setSignalHeaders(channel_info)
        f.setDatarecordDuration(actual_duration)
        f.writePhysicalSamples(signal.astype(np.float64))  
        
    print(f"EDF file created: {filename}")
    print(f"Requested duration: {duration:.6f}s")
    print(f"Actual duration: {actual_duration:.6f}s")
    
    return filename

