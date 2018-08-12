import pyaudio
from pynput import keyboard

import pandas as pd
import numpy as np
import os
import datetime
import threading
import time
import queue
import wave
import random

"""
save_freq = the number of key presses recorded before saving to disk
mode:   Default= save samples to disk normally
        Sample=don't save to disk, return individual samples
        Continuous= ignore key events, sample continuously
"""
class KeyAudio(object):
    def __init__(self, save_freq=500, mode="Default", save_wav=False):
        print("Instantiating...")
        
        # I want to record for approximately 250ms for each keypress. This recording should center on the press event.
        # <---200ms---> KeyRelease <---50ms--->
        
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.delta_ms = 25 # Stream read size in milliseconds
        self.full_record_ms = 250 # Key press audio recording length in milliseconds
        
        self.row_size = int(self.rate / self.chunk * self.delta_ms * (1/1000))
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format = self.format, channels = self.channels, rate = self.rate, input = True, frames_per_buffer = self.chunk)
        
        self.saving = False # Saving dataframe flag
        self.running = False # Keyboard and Audio Log started flag
        self.released = True
        self.start_time = 0.0 # Time key pressed
        self.max_hold_ms = 400 # Maximum time between hold and release for recording to be valid
        self.record_window = False

        self.frame_list = []
        self.df_list = [] # Holds list of dictionaries until user saves as dataframe
        self.q = queue.Queue() # Use Queue as FIFO for recorded frames
        
        self.key_cnt = 0 # Track the number of recorded keypresses for this session
        
        self.dataset_subdir = 'DataSet/'
        self.save_cnt = 0
        self.save_freq = save_freq
        self.sample_ready = False
        self.mode = mode
        self.save_wav = save_wav
    
    def get_dev_info(self):
        return self.p.get_default_input_device_info()
	
    # Start Listening for Keyboard Presses and recording Audio
    def startListener(self):
        print("Starting listener...")
        self.running = True
        
        # Record microphone in separate thread
        mic_threads = []
        t = threading.Thread(target=self.log)
        mic_threads.append(t)
        t.start()

        self.start_key_listener()

    def start_key_listener(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()
      
    # Keyboard Press
    def on_press(self, key):
        if self.mode == "Continuous":
            return

        if self.released == True:
            self.key = key

            self.start_time = time.time()
            self.frame_list = [] # Clear list of frames
            
            self.released = False
            self.record_window = True # Add frames to frame_list when True
            
    def on_release(self, key):
        self.released = True
        
        # Escape Pressed
        if key == keyboard.Key.esc:
            self.running = False

            if self.mode == "Continuous":
                filename = self.dataset_subdir + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_c.pkl'
                self.save_dataframe(filename=filename)

            return False # Stop Key Listener
        else:
            if self.mode == "Continuous":
                return

            # Time between press and release should be less than some delta threshold
            delta_t = time.time() - self.start_time # Time between press and release
            if delta_t > self.max_hold_ms/1000:
                print('Maximum hold time exceeded')
                return

            # Don't record data when saving
            if self.saving:
                return

            print(key)

            # Keep recording audio for some delta defined after the key is pressed
            post_rel_ms = round(self.full_record_ms - delta_t*1000)
            if post_rel_ms > self.delta_ms:
                time.sleep(post_rel_ms/1000)

            self.record_window = False # Add frames to frame_list when True

            # Verify queue size
            if self.q.qsize() != round(self.full_record_ms/self.delta_ms):
                print("Error: Incorrect queue size: {}".format(self.q.qsize()))
                return

            if self.mode == "Sample":
                self.sample_ready = True
                return # Don't save to disk

            # Save data to dataframe
            if self.key_cnt % self.save_freq == self.save_freq - 1:
                print("Saving dataframe. Session key count: {}, Sample Count: {}".format(self.key_cnt, len(self.df_list)))
                filename = self.dataset_subdir + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(self.save_cnt) + '.pkl'
                self.save_dataframe(filename=filename)

            self.key_cnt += 1 # New keypress recorded

            if self.save_wav:
                for i, df in enumerate(self.df_list):
                    wav_name = "file" + str(i) + ".wav"
                    self.save_data_as_wav(df['raw'], filename=wav_name)


            
    def key_to_string(self, key):
        key_str = ""
        if str(type(key)) == "<enum 'Key'>":
            key_str = key.name # Type <enum 'Key'
        elif type(key) == str:
            key_str = key
        else:
            key_str = key.char # Type pynput.keyboard._win32.KeyCode
        return key_str
            
    def log(self):
        shift_cnt = 0
        while self.running:
            data = self.stream.read(self.chunk * self.row_size) # Raw data in byte format
            self.q.put(data)

            if self.q.qsize() > round(self.full_record_ms/self.delta_ms):
                self.q.get()

            if self.record_window or self.mode == "Continuous":
                if self.mode == "Continuous":
                    if shift_cnt % 10 == 9:
                        shift_cnt = 0
                        self.key = "continuous"
                        print(len(self.df_list))
                    else:
                        shift_cnt += 1
                        continue # Skip saving to dataframe

                frame = list(self.q.queue)  # A list of delta_ms raw byte samples
                frame_bytes = bytearray([byte for row in frame for byte in row])
                frames_int = np.frombuffer(frame_bytes, dtype=np.int16)  # convert to int16

                # Create dictionary for each sample and append to list (used later to create dataframe)
                record_sample = [{'key': self.key_to_string(self.key), 'data': frames_int, 'raw': frame_bytes,
                                  'timestamp': datetime.datetime.utcnow()}]
                self.df_list.extend(record_sample)

        # When run complete, stop stream
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
    
    def save_data_as_wav(self, data, filename="file.wav"):
        WAVE_OUTPUT_FILENAME = filename
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb') # 'wb' write only mode
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.p.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(data)
        waveFile.close()
        
    def save_dataframe(self, filename='DataSet/data.pkl'):
        self.saving = True
        df = pd.DataFrame.from_records(self.df_list)
            
        # Save to pickle file
        df.to_pickle(filename)
        
        self.df_list = [] # Clear existing list data
        
        self.saving = False
        
        self.save_cnt += 1