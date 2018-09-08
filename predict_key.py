from keyaudio import KeyAudio
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import pickle
import time
import sys

"""
"Sample": (Default) Predict key on key press/release event
"Continuous": Keep predicting key events at a set time interval. Pass in with command line argument
"""
mode = "Continuous"
if len(sys.argv) > 1:
    mode = str(sys.argv[1])
    print(mode)

dataset_dir = 'DataSet/'

labels = pickle.load(open(os.path.join(dataset_dir, 'labels.p'), "rb"))
print(labels)

# Import a subset of the training set.
# This will be used for normalizing the sampled data
for filename in os.listdir(dataset_dir):
    if filename.endswith("test_set.pkl"):
        continue
    if filename.endswith(".pkl"):
        df_norm = pd.read_pickle(os.path.join(dataset_dir, filename))
        break


scaler = pickle.load(open(os.path.join(dataset_dir, 'scaler.p'), "rb"))

data_width = len(df_norm['data'][0])

"""
    df          Key press audio sample as dataframe
"""
def normalize_data(df):
    input_data = df['data'].values  # Convert to numpy array
    input_data = np.stack(input_data, axis=0)  # Create numpy matrix from array of arrays

    # Normalize data
    normalized_data = scaler.transform(input_data)

    normalized_data = normalized_data.reshape((normalized_data.shape[0], normalized_data.shape[1], 1))
    return normalized_data

def reset_sample():
    key.df_list = []
    key.sample_ready = False

# Load TensorFlow Graph
loader = tf.train.import_meta_graph('checkpoints/model.ckpt.meta')
with tf.Session() as sess:
    loader.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("Inputs/inputs:0")
    keep_prob = graph.get_tensor_by_name("keep_probability:0")
    predicted = graph.get_tensor_by_name("predicted:0")

    # Start Listener
    key = KeyAudio(mode=mode, save_wav=False)
    print(key.get_dev_info())
    key.startListener()

    count = 1
    correct_cnt = 0
    key_detection = False
    prediction_list = []
    while key.running:
        if key.sample_ready == True:
            df = pd.DataFrame.from_records(key.df_list) # Key press audio sample

            # Ignore keys not in the label classes
            if not(df['key'].values[0] in labels):
                print("Key skipped. Not in label set.")
                reset_sample()
                continue

            normalized_data = normalize_data(df)  # Normalized audio sample

            # Make Prediction
            feed_dict = {x: normalized_data, keep_prob: 1.0}
            prediction = sess.run(predicted, feed_dict=feed_dict).squeeze()

            if len(df) > 1:
                center = round(len(df)/2)
                predicted_key = labels[np.argmax(prediction[center])]
            else:
                predicted_key = labels[np.argmax(prediction)]

            if mode == "Sample":
                if df['key'].values[0] == predicted_key:
                    correct_cnt += 1
                print('Count: {0}, Accuracy: {1:0.2f}, Prediction: {2}'.format(count, correct_cnt/count, predicted_key))
            elif mode == "Continuous":
                if predicted_key != "continuous":
                    key_detection = True
                    prediction_list.extend(key.df_list)
                else:
                    if key_detection:
                        df_list = pd.DataFrame.from_records(prediction_list)
                        normalized_data = normalize_data(df_list)
                        feed_dict = {x: normalized_data, keep_prob: 1.0}
                        predictions = sess.run(predicted, feed_dict=feed_dict).squeeze()
                        prediction = np.sum(predictions, axis=0)
                        predicted_key = labels[np.argmax(prediction)]

                        print('Final Prediction: {0}'.format(predicted_key))
                    key_detection = False
                    prediction_list = []

            # Reset relevant class parameters
            reset_sample()
            count += 1