import pandas as pd
from glob import glob
import numpy as np
import os
from scipy import stats
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D
from tensorflow.python.keras.layers import Flatten

class LoadData:
    le = preprocessing.LabelEncoder()

    # Function for loading the data in the dataframe for a particular sensor
    def read_data(self, path):
        # Loading the accelerometer data from both phone and watch
        accel_list = []
        gyro_list = []
        for dir in os.listdir(path):
            for inner_dir in os.listdir(path + dir):
                # print("./raw/train/" + dir + "/" + inner_dir)
                cols = ['id', 'class', 'timestamp', 'x', 'y', 'z']
                filenames = glob(path + dir + "/" + inner_dir +  "/" + 'data_*_accel_*.txt')
                if filenames:
                    for file in filenames:
                        df = pd.read_csv(file, names=cols, header=None)
                        df['z'] = df['z'].str[:-1]
                        accel_list.append(df)

                # Loading the gyroscope data for each user in the dataset
                cols = ['id', 'class', 'timestamp', 'x', 'y', 'z']
                filenames = glob(path + dir + "/" + inner_dir + "/" + 'data_*_gyro_*.txt')
                if filenames:
                    for file in filenames:
                        df = pd.read_csv(file, names=cols, header=None)
                        df['z'] = df['z'].str[:-1]
                        gyro_list.append(df)

        # Joining the data to have six features for each user
        accel_data = pd.concat(accel_list, ignore_index=True)
        gyro_data = pd.concat(gyro_list, ignore_index=True)
        print("Success!!!\n\n")
        return accel_data.join(gyro_data, rsuffix='_1')

    # Function for data preprocessing
    def preprocess_data(self, data):
        # Dropping the na records
        data.dropna(axis=0, how='any', inplace=True)
        data['label'] = self.le.fit_transform(data['class'].values.ravel())

        # Next we need to normalize the data for all the axis
        data['x'] = data['x'].astype(float) / data['x'].astype(float).max()
        data['y'] = data['y'].astype(float) / data['y'].astype(float).max()
        data['z'] = data['z'].astype(float) / data['z'].astype(float).max()
        data['x_1'] = data['x_1'].astype(float) / data['x_1'].astype(float).max()
        data['y_1'] = data['y_1'].astype(float) / data['y_1'].astype(float).max()
        data['z_1'] = data['z_1'].astype(float) / data['z_1'].astype(float).max()

        # We can also round off the number to certain decimal places
        data['x'] = data['x'].round(4)
        data['y'] = data['y'].round(4)
        data['z'] = data['z'].round(4)
        data['x_1'] = data['x_1'].round(4)
        data['y_1'] = data['y_1'].round(4)
        data['z_1'] = data['z_1'].round(4)
        print("Success!!!\n\n")
        return data, self.le

    # Function for defining the model
    def create_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        print("Success!!!\n\n")
        return model

    # Function for splitting the data in segements for training
    def segments(self, data, time_steps, step, label_name):
        N_FEATURES = 6
        segments = []
        labels = []
        for i in range(0, len(data) - time_steps, step):
            xs = data['x'].values[i: i + time_steps]
            ys = data['y'].values[i: i + time_steps]
            zs = data['z'].values[i: i + time_steps]
            x1_s = data['x_1'].values[i: i + time_steps]
            y1_s = data['y_1'].values[i: i + time_steps]
            z1_s = data['z_1'].values[i: i + time_steps]

            # Retrieving the most often  used label in this segment
            label = stats.mode(data[label_name][i: i + time_steps])[0][0]
            segments.append([xs, ys, zs, x1_s, y1_s, z1_s])
            labels.append(label)

        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
        labels = np.asarray(labels)
        print("Success!!!\n\n")
        return reshaped_segments, labels
