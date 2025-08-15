
import time
import json
import csv
from datetime import datetime
from pynput import keyboard
import signal
import sys
import os

class KeyboardRecorder:
    def __init__(self):
        self.events = []
        self.start_time = None
        self.recording = False
        self.listener = None
        
    def start_recording(self):
        """
        starts the recording process of the keyboard logger
        """
        self.start_time = time.time()
        self.recording = True
        self.events = []
        
        print(f"Recording started at: {datetime.fromtimestamp(self.start_time)}")
        print("Press ESC to stop recording")
        
        # Set up keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_press
        )
        self.listener.start()
        
    def _on_press(self, key):
        """
        handles key press (what happens when user presses key)
        """
        if not self.recording:
            return
            
        timestamp = time.time()
        relative_time = timestamp - self.start_time
        
        try:
            key_name = key.char if hasattr(key, 'char') and key.char else str(key)
        except AttributeError:
            key_name = str(key)
            
        event = {
            'event_type': 'press',
            'key': key_name,
            'relative_time': relative_time,
            'datetime': datetime.fromtimestamp(timestamp).isoformat()
        }
        
        self.events.append(event)
        print(f"Key pressed: {key_name} at {relative_time:.3f}s")
        
        # stop recording when pressing escape
        if key == keyboard.Key.esc:
            self.stop_recording()
            
        
    def stop_recording(self):
        """
        stops recording 
        """
        self.recording = False
        if self.listener:
            self.listener.stop()
        print(f"\nRecording stopped. Total events: {len(self.events)}")
        
        
    def save_events(self, filename=None):
        """
        saving the events to output files
        """
        if not self.events:
            print("No events to save!")
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keyboard_events_{timestamp}"
        
        if not os.path.exists("./keyboard_events/"):
            os.mkdir("keyboard_events")
            
        # save as json
        json_filename = f"./keyboard_events/{filename}.json"
        with open(json_filename, 'w') as f:
            json.dump({
                'recording_start': self.start_time,
                'recording_start_datetime': datetime.fromtimestamp(self.start_time).isoformat(),
                'events': self.events
            }, f, indent=2)
        print(f"Events saved to {json_filename}")
        return json_filename
        

def run_keyboard_logger():
    """
    starts the keyboard logger and returns paths to json file for main file. 
    """
    print("Keyboard Input Recorder")
    print("======================")
    print("This script will record all keyboard events with timestamps.")
    print("Press ESC to stop recording or Ctrl+C to exit.\n")
    
    recorder = KeyboardRecorder()
    recorder.start_recording()
    
    try:
        while recorder.recording:
            time.sleep(0.1)
    except KeyboardInterrupt:
        recorder.stop_recording()
    
    json_file = recorder.save_events()
    return json_file

def main():
    json_file = run_keyboard_logger()
    print(f"Logger finished. JSON: {json_file}")

if __name__ == "__main__":        
    main()