import mne
import json
import os

# test parameters (only need to change when running this script alone)
# when run with main.py it will work with other parameters
test_eeg_file = '13.202332973480225.edf'
output_dir = 'eeg_segments'
segment_duration = 0.5
test_json_file = './keyboard_events/keyboard_events_20250814_200245.json'

def segment_eeg(eeg_file, json_file, output_dir, segment_duration):
    """
    segments eeg files
    """
    json_name = os.path.basename(json_file)
    output = f"./{output_dir}/{json_name}/"
    if not os.path.exists(output):
        os.makedirs(output)

    # LOAD THE EEG FILE 
    raw = mne.io.read_raw_edf(eeg_file, preload=True)
    raw.pick_types(eeg=True)  # only keeps eeg channels

    # LOAD JSON FILE
    with open(json_file, 'r') as f:
        keylog = json.load(f)

    recording_start = keylog['recording_start'] 
    events = keylog['events']

    # FILTER ONLY KEY PRESSES 
    press_events = [e for e in events if e['event_type'] == 'press']

    # PROCESS EACH PRESS EVENT 
    for i, event in enumerate(press_events):
        # stores pressed key and relative time stamp
        key = event['key'].replace("Key.", "")  
        rel_time = event['relative_time']       

        # finds the time frame of before and after
        t_min = rel_time - segment_duration
        t_max = rel_time + segment_duration

        # if going out of bounds (overlap of keypress)
        if t_min < 0 or t_max > raw.times[-1]:
            print(f"Skipping event {i} ({key}) - out of bounds [{t_min:.2f}s to {t_max:.2f}s]")
            continue

        # create the segement 
        segment = raw.copy().crop(tmin=t_min, tmax=t_max)

        # save as fif file, could save as other file type
        out_name = f"segment_{i}_{key}.fif"
        segment.save(os.path.join(output, out_name), overwrite=True)
        print(f"Saved: {out_name}")

def main():
    segment_eeg(test_eeg_file, test_json_file, output_dir, segment_duration)
    
if __name__ == "__main__":        
    main()