from keyaudio import KeyAudio

# Start Listener
key = KeyAudio(mode="Continuous", save_wav=False)
print(key.get_dev_info())
key.startListener()