# Keyboard Audio Predictor

### predict_key.py
Run to start predictions. Default mode is Continuous which doesn't look at the keyboard's key press/release events

To make predictions and match accuracy against key press/release events, run in Sample mode: predict_key.py Sample

### get_data.py
Run to start logging audio from key release events. Hit ESC to exit.

### background_noise.py
Run to log continous noise to add a 'continuous' class to the database. To create a 'Not a Key' class.

### Train.ipynb
Preprocess and train recorded data on a CNN DNN

### helper.py
Extra functions used for import in to main scripts. Where the mechanics of the code aren't as relevant to that main script. 

### Get Data.ipynb
Older simpler form of get_data.py, but in jupyter notebook.