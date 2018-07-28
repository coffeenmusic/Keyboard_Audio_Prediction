from keyaudio import KeyAudio
				
key = KeyAudio(save_wav=True)
print(key.get_dev_info())
key.startListener()