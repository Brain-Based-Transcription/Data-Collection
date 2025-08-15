
import os
import json
from keyboard_logger import run_keyboard_logger
from generate_edf import generate_edf  # if you still need EDF generation
from segment_eeg import segment_eeg

# PARAMETERS
output_dir = 'eeg_segments'
segment_duration = 0.5

keyboard_events_dir = './keyboard_events'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(keyboard_events_dir, exist_ok=True)
#json_file = 'keyboard_events/keyboard_events_20250814_200245.json'

#
# Run Keyboard Logger
#
json_file = run_keyboard_logger()
print(json_file)

#
# Generate EEG (Remove once figured out how to use EEG device)
#

# finds the length of the keyboard logging file
with open(json_file, "r") as f:
    keylog = json.load(f)
events = keylog["events"]
duration = events[-1]["relative_time"]

eeg_file = generate_edf(duration)

#
# Segment EEG
#
segment_eeg(eeg_file, json_file, output_dir, segment_duration)