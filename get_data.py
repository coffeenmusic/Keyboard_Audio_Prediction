import pandas as pd
import pyaudio
import numpy as np
import os
from sklearn.preprocessing import normalize
import datetime
import threading
from time import sleep
import queue
from pynput import keyboard

pa = pyaudio.PyAudio()
print(pa.get_default_input_device_info())


class KeyAudio(object):
    def __init__(self):
        print("Instantiating...")
        
        # I want to record for approximately 300ms for each keypress. This recording should center on the press event.
        # <---150ms---> KeyPress <---150ms--->
        
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.delta_ms = 25 # Stream read size in milliseconds
        self.full_record_ms = 250 # Key press audio recording length in milliseconds
        self.post_press_ms = 150 # Recording time after key press in milliseconds
        
        self.row_size = int(self.rate / self.chunk * self.delta_ms * (1/1000))
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format = self.format, channels = self.channels, rate = self.rate, input = True, frames_per_buffer = self.chunk)
        
        self.running = False # Keyboard and Audio Log started flag
        self.released = True # Key release seen after key press flag
        
        self.frames = [] # A list of delta_ms raw byte samples
        self.df_list = [] # Holds list of dictionaries until user saves as dataframe
        self.q = queue.Queue() # Use Queue as FIFO for recorded frames
        
        self.key_cnt = 0 # Track the number of recorded keypresses for this session
    
    # Start Listening for Keyboard Presses and recording Audio
    def startListener(self):
        print("Starting listener...")
        self.running = True
        
        # Record microphone in separate thread
        threads = []
        t = threading.Thread(target=self.log)
        threads.append(t)
        t.start()

        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()
      
    # Keyboard Press
    def on_press(self, key): 
        if not(self.released):
            print("Not released")
	
        # Escape Pressed
        if key == keyboard.Key.esc:
            self.running = False
            return False # Stop Key Listener
        elif self.released: # Disallow holding hey to repeatedly record audio logs
            print(key)
            
            self.released = False
            
            sleep(self.post_press_ms/1000) # Keep recording audio for some delta defined after the key is pressed
            
            if self.q.qsize() != round(self.full_record_ms/self.delta_ms):
                print("Error: Incorrect queue size: {}".format(self.q.qsize()))
                return
            
            self.frames = list(self.q.queue)
            frame_bytes = bytearray([byte for row in self.frames for byte in row])
            frames_int = np.frombuffer(frame_bytes, dtype=np.int16) # convert to int16
            
            # Create dictionary for each sample and append to list (used later to create dataframe)
            record_sample = [{'key': self.key_to_string(key), 'data': frames_int, 'raw': frame_bytes, 'timestamp': datetime.datetime.utcnow()}]
            self.df_list.extend(record_sample)
            
            # Save data to dataframe
            if self.key_cnt % 50 == 49:
                print("Saving dataframe. Session key count: {}".format(self.key_cnt))
                self.save_dataframe()
            
            self.key_cnt += 1 # New keypress recorded
            
    def on_release(self, key):
        self.released = True
            
    def key_to_string(self, key):
        key_str = ""
        if str(type(key)) == "<enum 'Key'>":
            key_str = key.name # Type <enum 'Key'
        else:
            key_str = key.char # Type pynput.keyboard._win32.KeyCode
        return key_str
            
    def log(self):
        while self.running:
            data = self.stream.read(self.chunk * self.row_size) # Raw data in byte format
            self.q.put(data)

            if self.q.qsize() > round(self.full_record_ms/self.delta_ms):
                self.q.get()

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
        df = pd.DataFrame.from_records(self.df_list)
        
        # Get existing data and combine with new data
        if os.path.isfile(filename):
            df_saved = pd.read_pickle(filename)
            df = df_saved.append(df, ignore_index=True)
            
        # Save to pickle file
        df.to_pickle(filename)
        
        self.df_list = [] # Clear existing list data
		
		
key = KeyAudio()
key.startListener()