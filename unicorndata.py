# 1. Ensure UnicornPy.py and Unicorn.dll (or equivalent for your OS)
#    are in the same directory as this script, or in your Python path.
#    These files come with the g.tec Unicorn Suite.
# 2. Install other requirements: pip install numpy pandas matplotlib
# 3. Turn on Unicorn headset and plug in USB dongle.
# 4. Run: python your_script_name.py
# This will attempt to connect, record 10s of EEG data, save it to CSV, and show a plot.
# To change recording duration, modify duration_seconds in main().

# Function Summaries (Updated):
# __init__: Sets up device handle, sampling rate, EEG channel names, and data storage.
# connect: Finds and connects to an available Unicorn headset using UnicornPy.
# start_recording: Starts data acquisition, collects EEG data for a specified duration, and stores it.
# save_data: Saves the collected EEG data into a CSV file with timestamps.
# plot_data: Generates a plot of the EEG data from the different channels.
# disconnect: Stops data acquisition and safely disconnects from the headset.
# main: Example script demonstrating the connect-record-save-plot-disconnect workflow.

import UnicornPy # Official Python API from g.tec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os # For path manipulation if needed

class UnicornDataCollector:
    def __init__(self):
        self.device = None
        self.device_info = None
        try:
            self.sampling_rate = UnicornPy.SamplingRate
        except AttributeError:

            print("Warning: UnicornPy.SamplingRate not found. Using default 250 Hz. Will update on connection.")
            self.sampling_rate = 250
        
        self.eeg_channel_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
        self.eeg_channels_count = len(self.eeg_channel_names)

        self.is_recording = False
        self.data_buffer = []
        self.recording_start_time_exact = None

    def connect(self):
        """Connect to the Unicorn BCI device."""
        try:
            print("Searching for Unicorn BCI devices...")
            available_devices = UnicornPy.GetAvailableDevices(True)
            if not available_devices:
                print("No Unicorn BCI devices found. \n"
                      "Ensure the device is ON, the Bluetooth dongle is plugged in, \n"
                      "and g.tec drivers/suite are correctly installed.")
                return False

            print(f"Available devices: {available_devices}")
            device_id_to_connect = available_devices[0]
            print(f"Attempting to connect to device: {device_id_to_connect}...")

            self.device = UnicornPy.Unicorn(device_id_to_connect)
            print(f"Successfully connected to Unicorn BCI: {device_id_to_connect}")

            self.sampling_rate = self.device.GetSamplingRate()
            print(f"Device Serial: {self.device.GetDeviceSerialNumber()}")
            print(f"Acquired Channels: {self.device.GetNumberOfAcquiredChannels()}")
            print(f"Actual Sampling Rate: {self.sampling_rate} Hz")

            if self.device.GetNumberOfAcquiredChannels() < self.eeg_channels_count:
                print(f"Warning: Device acquires {self.device.GetNumberOfAcquiredChannels()} channels, "
                      f"but we expect to extract {self.eeg_channels_count} EEG channels. "
                      "Check channel configuration if data seems incorrect.")
            return True
        except UnicornPy.DeviceException as e:
            print(f"Unicorn BCI Error (Connection): {str(e)}")
            self.device = None
            return False
        except Exception as e:
            print(f"An unexpected error occurred during connection: {str(e)}")
            self.device = None
            return False

    def start_recording(self, duration_seconds=10):
        """Start recording EEG data for a specified duration."""
        if not self.device:
            print("Device not connected. Please connect first.")
            return

        try:
            print("Starting acquisition...")
            self.device.StartAcquisition(False)
            self.is_recording = True
            self.data_buffer = []
            self.recording_start_time_exact = datetime.now()

            number_of_acquired_channels = self.device.GetNumberOfAcquiredChannels()
            

            scans_to_acquire_per_call = int(self.sampling_rate / 50)
            if scans_to_acquire_per_call == 0: scans_to_acquire_per_call = 1

            receive_buffer_size_bytes = scans_to_acquire_per_call * number_of_acquired_channels * UnicornPy.FloatSize
            receive_buffer = bytearray(receive_buffer_size_bytes)

            print(f"Recording for {duration_seconds} seconds...")
            
            total_samples_to_collect = duration_seconds * self.sampling_rate
            collected_samples_count = 0
            
            recording_loop_start_time = time.perf_counter()

            while collected_samples_count < total_samples_to_collect and self.is_recording:
                self.device.GetData(scans_to_acquire_per_call, receive_buffer, receive_buffer_size_bytes)
                
                data_chunk_flat = np.frombuffer(receive_buffer, dtype=np.float32)
                # Reshape: (scans_acquired_this_call, number_of_all_channels)
                data_chunk_reshaped = data_chunk_flat.reshape(scans_to_acquire_per_call, number_of_acquired_channels)
                
                # Extract only the EEG channels (assuming they are the first N channels)
                # If your device has a different channel order, this needs adjustment.
                eeg_data_chunk = data_chunk_reshaped[:, :self.eeg_channels_count]
                
                self.data_buffer.extend(eeg_data_chunk.tolist()) # Append as list of lists
                collected_samples_count += scans_to_acquire_per_call

                # Optional: Small sleep if GetData is non-blocking or to reduce CPU load slightly,
                # but GetData is typically blocking.
                # time.sleep(0.001) # Usually not needed if GetData blocks appropriately

            recording_loop_duration = time.perf_counter() - recording_loop_start_time
            print(f"Recording loop finished. Collected {len(self.data_buffer)} EEG samples in {recording_loop_duration:.2f} seconds.")

        except UnicornPy.DeviceException as e:
            print(f"Unicorn BCI Error (Recording): {str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred during recording: {str(e)}")
        finally:
            if self.device and self.is_recording:
                print("Stopping acquisition...")
                self.device.StopAcquisition()
            self.is_recording = False
            print("Recording process completed.")


    def save_data(self, filename=None):
        """Save recorded EEG data to a CSV file."""
        if not self.data_buffer:
            print("No data to save.")
            return None

        if filename is None:
            timestamp_str = (self.recording_start_time_exact or datetime.now()).strftime("%Y%m%d_%H%M%S")
            filename = f"eeg_data_{timestamp_str}.csv"

        data_array = np.array(self.data_buffer)
        
        if data_array.shape[1] != self.eeg_channels_count:
            print(f"Warning: Data array has {data_array.shape[1]} columns, "
                  f"but {self.eeg_channels_count} EEG channel names are defined. "
                  f"Using first {data_array.shape[1]} channel names for CSV.")
            current_csv_channel_names = self.eeg_channel_names[:data_array.shape[1]]
        else:
            current_csv_channel_names = self.eeg_channel_names

        df = pd.DataFrame(data_array, columns=current_csv_channel_names)
        
        if self.recording_start_time_exact and len(df) > 0:
            timestamps = [self.recording_start_time_exact + timedelta(seconds=i/self.sampling_rate) for i in range(len(df))]
            df.insert(0, 'timestamp', timestamps)
        else:
             df.insert(0, 'timestamp', pd.NaT)

        try:
            df.to_csv(filename, index=False, float_format='%.4f')
            print(f"Data successfully saved to {filename}")
            return df
        except Exception as e:
            print(f"Error saving data to CSV: {str(e)}")
            return None

    def plot_data(self, df_eeg_data=None):
        """Plot the recorded EEG data from a DataFrame."""
        if df_eeg_data is None or df_eeg_data.empty:
            print("No data provided or data is empty for plotting.")
            return

        plot_columns = [col for col in df_eeg_data.columns if col in self.eeg_channel_names]
        if not plot_columns:
            print(f"No EEG channels ({', '.join(self.eeg_channel_names)}) found in the DataFrame to plot.")
            return

        data_to_plot = df_eeg_data[plot_columns].values
        
        time_vector = None
        if 'timestamp' in df_eeg_data.columns and pd.api.types.is_datetime64_any_dtype(df_eeg_data['timestamp']):
            try:
                timestamps_numeric = pd.to_datetime(df_eeg_data['timestamp'])
                if not timestamps_numeric.empty and timestamps_numeric.iloc[0] is not pd.NaT:
                     time_vector = (timestamps_numeric - timestamps_numeric.iloc[0]).dt.total_seconds().values
                else:
                    time_vector = np.arange(data_to_plot.shape[0]) / self.sampling_rate
            except Exception as e:
                print(f"Could not parse timestamps for plotting x-axis: {e}. Using sample numbers scaled by sampling rate.")
                time_vector = np.arange(data_to_plot.shape[0]) / self.sampling_rate
        else:
            time_vector = np.arange(data_to_plot.shape[0]) / self.sampling_rate


        plt.figure(figsize=(15, 10))
        num_channels_to_plot = data_to_plot.shape[1]
        
        channel_offsets = np.std(data_to_plot, axis=0) * 3

        for i in range(num_channels_to_plot):
            offset_value = np.sum(channel_offsets[:i])
            plt.plot(time_vector, data_to_plot[:, i] - offset_value, label=plot_columns[i])
        
        plt.title('EEG Data Recording')
        plt.xlabel(f'Time (seconds from start at {self.sampling_rate} Hz)')
        plt.ylabel('Amplitude (μV, arbitrary offset for clarity)')
        plt.yticks([])
        plt.legend(loc='upper right')
        plt.grid(True, axis='x', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def disconnect(self):
        """Stop acquisition and disconnect from the device."""
        if self.device:
            print("Attempting to disconnect from device...")
            try:
                if self.is_recording:
                    print("Ensuring acquisition is stopped before closing.")
                    self.device.StopAcquisition()
                    self.is_recording = False
                
                self.device.Close()
                print("Successfully disconnected from Unicorn BCI device.")
            except UnicornPy.DeviceException as e:
                print(f"Unicorn BCI Error (Disconnection): {str(e)}")
            except Exception as e:
                print(f"An unexpected error occurred during disconnection: {str(e)}")
            finally:
                self.device = None

def main():
    print("🦄 Unicorn BCI Data Collector Initializing...")
    print("Reminder: Ensure UnicornPy.py and the required DLL (e.g., Unicorn.dll) are accessible.")
    print("This typically means they should be in the same directory as this script,")
    print("or their location is added to your system's PATH or Python's sys.path.\n")
    
    # Example: Add SDK path if it's in a subfolder named 'unicorn_sdk'
    # sdk_path = os.path.join(os.path.dirname(__file__), 'unicorn_sdk')
    # if os.path.exists(sdk_path):
    #     import sys
    #     if sdk_path not in sys.path:
    #         sys.path.append(sdk_path)
    #     print(f"Temporarily added {sdk_path} to Python path.")

    collector = UnicornDataCollector()
    
    if collector.connect():
        try:
            recording_duration_seconds = 10 #edit if you want to record for a different amount of time
            collector.start_recording(duration_seconds=recording_duration_seconds)
            
            recorded_dataframe = collector.save_data()
            
            if recorded_dataframe is not None and not recorded_dataframe.empty:
                collector.plot_data(df_eeg_data=recorded_dataframe)
            else:
                print("Data recording or saving failed, skipping plot.")
                
        except Exception as e:
            print(f"An critical error occurred in the main execution block: {str(e)}")
        finally:
            print("Workflow finished. Cleaning up...")
            collector.disconnect()
    else:
        print("Could not connect to the Unicorn BCI device. Please check setup and try again.")
    print("Data collection script finished.")

if __name__ == "__main__":
    main()