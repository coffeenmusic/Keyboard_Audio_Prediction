from keyaudio import KeyAudio
				
key = KeyAudio()
print(key.get_dev_info())
key.startListener()