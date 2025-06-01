#run python3 -m pip install brainflow 
#note: make sure you run with python3
#python3 -m pip install numpy pandas matplotlib pyOpenBCI

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

class OpenBCI_Cyton_BrainFlow_Recorder:
    def __init__(self):
        self.board = None
        self.sampling_rate = None  #set after initial connection
        self.eeg_channels_count = 8  # OpenBCI Cyton has 8 EEG channels
        self.data_buffer = []
        self.recording_start_time = None

    def connect(self, serial_port='COM3'):
        #connect to OpenBCI Cyton board with BrainFlow
        try:
            params = BrainFlowInputParams()
            params.serial_port = serial_port  #For USB connection (e.g., '/dev/ttyUSB0' for MacOS and Linux or 'COM3 for Windows,  change in function param')
            self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)  

            self.board.prepare_session()

            self.sampling_rate = self.board.get_sampling_rate(BoardIds.CYTON_BOARD.value, params)
            print(f"Successfully connected to OpenBCI Cyton. Sampling rate: {self.sampling_rate} Hz.")
            return True
        except Exception as e:
            print(f"Error connecting to OpenBCI Cyton: {str(e)}")
            return False

    def record_eeg_data(self, duration_seconds=10):
        #Recording EEG data for a specified duration
        if not self.board:
            print("Device not connected. Please connect first.")
            return

        try:
            print(f"Recording EEG data for {duration_seconds} seconds...")
            self.recording_start_time = datetime.now()

            #start receiving data from board
            self.board.start_stream()

            #receiving data for duration specified   
            time.sleep(duration_seconds)

            data = self.board.get_board_data()
            #add data thats been received to buffer
            self.data_buffer.append(data)

            #stopping the receival of data
            self.board.stop_stream()

            print(f"Recording completed. Collected {len(self.data_buffer)} samples.")

        except Exception as e:
            print(f"Error during data recording: {str(e)}")

    def save_data(self, filename=None):
        #Saving EEG data to a CSV file
        if not self.data_buffer:
            print("No data to save.")
            return None

        if filename is None:
            timestamp_str = self.recording_start_time.strftime("%Y-%m-%d_%H:%M:%S")
            filename = f"eeg_data_{timestamp_str}.csv"

        #formatting data to match structure for CSV output
        data_array = np.vstack(self.data_buffer)
        df = pd.DataFrame(data_array, columns=[f"Ch{i+1}" for i in range(self.eeg_channels_count)])

        # Add timestamp column
        timestamps = [self.recording_start_time + pd.to_timedelta(i / self.sampling_rate, unit='s') for i in range(len(df))]
        df.insert(0, 'timestamp', timestamps)

        try:
            df.to_csv(filename, index=False, float_format='%.4f')
            print(f"Data successfully saved to {filename}")
            return df
        except Exception as e:
            print(f"Error saving data to CSV: {str(e)}")
            return None

    def plot_data(self, df_eeg_data=None):
        #Plot the recorded EEG data
        if df_eeg_data is None or df_eeg_data.empty:
            print("No data provided or data is empty for plotting.")
            return

        time_vector = (df_eeg_data['timestamp'] - df_eeg_data['timestamp'][0]).dt.total_seconds().values
        plt.figure(figsize=(15, 10))

        for col in df_eeg_data.columns[1:]:  # Skip the timestamp column
            plt.plot(time_vector, df_eeg_data[col], label=col)

        plt.title('EEG Data Recording')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (μV)')
        plt.legend(loc='upper right')
        plt.grid(True, axis='x', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def disconnect(self):
        #Stop data acquisition and disconnect from the Cyton board
        if self.board:
            try:
                print("Working to disconnect from OpenBCI Cyton...")
                self.board.release_session()
                print("Successfully disconnected.")
            except Exception as e:
                print(f"Error during disconnection: {str(e)}")
            finally:
                self.board = None


def main():
    print("OpenBCI Cyton BrainFlow Recorder Initializing...")

    recorder = OpenBCI_Cyton_BrainFlow_Recorder()

    # Connect to the OpenBCI Cyton board (adjust the port if needed)
    if recorder.connect(serial_port='COM3'):
        try:
            # Record EEG data for 10 seconds
            recorder.record_eeg_data(duration_seconds=10)

            # Save the data and plot it
            recorded_dataframe = recorder.save_data()
            if recorded_dataframe is not None and not recorded_dataframe.empty:
                recorder.plot_data(df_eeg_data=recorded_dataframe)
            else:
                print("Data recording or saving failed, skipping plot.")

        except Exception as e:
            print(f"Error in main execution: {str(e)}")
        finally:
            print("Workflow finished. Cleaning up...")
            recorder.disconnect()
    else:
        print("Failed to connect to OpenBCI Cyton. Please check your setup and try again.")

    print("Data collection script finished.")


if __name__ == "__main__":
    main()

