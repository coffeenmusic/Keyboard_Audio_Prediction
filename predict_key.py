from keyaudio import KeyAudio
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import pickle

dataset_dir = 'DataSet/'

labels = pickle.load(open(os.path.join(dataset_dir, 'labels.p'), "rb"))

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
    #df = df.append(df_norm[:2], ignore_index=True)

    input_data = df['data'].values  # Convert to numpy array
    input_data = np.stack(input_data, axis=0)  # Create numpy matrix from array of arrays

    # Normalize data
    #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.transform(input_data)

    normalized_data = normalized_data.reshape((normalized_data.shape[0], normalized_data.shape[1], 1))
    return normalized_data

# Load TensorFlow Graph
loader = tf.train.import_meta_graph('checkpoints/model.ckpt.meta')
with tf.Session() as sess:
    loader.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("Inputs/inputs:0")
    keep_prob = graph.get_tensor_by_name("keep_probability:0")
    predicted = graph.get_tensor_by_name("predicted:0")

    # Start Listener
    key = KeyAudio(mode="Sample", save_wav=False)
    print(key.get_dev_info())
    key.startListener()
    count = 1
    correct_cnt = 0
    while key.running:
        if key.sample_ready == True:
            df = pd.DataFrame.from_records(key.df_list) # Key press audio sample

            normalized_data = normalize_data(df) # Normalized audio sample
            print(normalized_data)

            # Make Prediction
            feed_dict = {x: normalized_data, keep_prob: 1.0}
            prediction = sess.run(predicted, feed_dict=feed_dict).squeeze()

            print(prediction)
            predicted_key = labels[np.argmax(prediction)]
            print('Prediction: {0}'.format(predicted_key))

            if df['key'].values[0] == predicted_key:
                correct_cnt += 1
            print('Count: {0}, Accuracy: {1:0.2f}'.format(count, correct_cnt/count))

            # Reset relevant class parameters
            key.df_list = []
            key.sample_ready = False
            count += 1