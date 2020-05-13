from tensorflow.python.keras.optimizers import SGD, Adam

from LoadData import LoadData
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import load_model


TIME_PERIODS = 80
STEP_DISTANCE = 40

obj = LoadData()

# =============================================================================
# Function for compiling and fitting the model
def compile_fit(model, callbacks_list, x_train, y_train, x_test, y_test):
    # Compiling the model
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])

    #  Fitting the model
    batch = 100
    epochs = 10
    history = model.fit(x_train, y_train, batch_size = batch, epochs = epochs, callbacks = callbacks_list, validation_split = 0.2, verbose = 2)

    # Saving the model
    model.save_weights('max_features_weights.h5')
    model.save('max_features_keras.h5')

    # Predicting on the test data
    y_pred = model.predict(x_test)

    # We now need to fetch the class with the highest probability from the predictions
    y_pred_final = np.argmax(y_pred, axis = 1)
    y_test_final = np.argmax(y_test, axis = 1)

    # Evaluating the model on the test set
    test_acc = accuracy_score(y_test_final, y_pred_final)
    print("Success!!!\n\n")
    # print("The accuracy of the model is:-")
    # print(test_acc * 100)
# ==========================================================================================


# ================
# MAIN PROCESSING
# ================

# ==========================================================================================
# Fetching data for trainig the model
print("Reading the data from the files...\n\n")
train_path = "./raw/train/"
data = obj.read_data(train_path)
print("Preprocessing the data loaded...\n\n")
data, le = obj.preprocess_data(data)

# Function for splitting the data in the train and the test set for a particular class
# We are keeping the data for 17 people in the train set and rest in the test set
train_data = data[data["id"] < 1616]
test_data = data[data["id"] > 1615]

# Splitting the data in segments
print("Splitting the train data into segments for training...\n\n")
x_train, y_train = obj.segments(train_data, TIME_PERIODS, STEP_DISTANCE, 'label')
print("Splitting the train data into segments for training...\n\n")
x_test, y_test = obj.segments(test_data, TIME_PERIODS, STEP_DISTANCE, 'label')

# Creating the variables for training the model
num_time_periods, sensors =  x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size

# We need to optimize the shape of the input data so that it can be fed to the neural network
input_shape = (num_time_periods, sensors)
# x_train = x_train.reshape(x_train.shape[0], input_shape)
# x_test = x_test.reshape(x_test.shape[0], input_shape)

# Converting the datatypes as accepted by keras
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

# One hot encoding of the labels
print("One Hot Encoding of the labels...\n\n")
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print("Success!!!\n\n")

# Creating the model
print("Defining the model...\n\n")
model = obj.create_model(input_shape, num_classes)

# List of callback actions that are to be taken to stop the training after the model reaches an optimal point
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath = './models/max_features.h5',
        monitor = 'val_loss',
        save_best_only= True)
    # ),
    # tf.keras.callbacks.EarlyStopping(monitor = 'acc', patience = 1)
]

# Compiling and fitting the model
print("Now fitting the model and starting with the training...\n\n")
compile_fit(model, callbacks_list, x_train, y_train, x_test, y_test)
# ============================================================================================




# =============================================================================================
test_path = './raw/test/'

# Function for testing th model on the testing data provided
def testing_model():
    print("Loading the data...\n\n")
    init_data = obj.read_data(test_path)
    print("Preprocessing the data...\n\n")
    data = obj.preprocess_data(init_data)
    print("Splitting the data in segments...\n\n")
    x_test, y_test = obj.segments(data, TIME_PERIODS, STEP_DISTANCE, 'label')
    x_test = x_test.astype('float32')
    print("One Hot Encoding the test labels...\n\n")
    y_test = to_categorical(y_test, 18)
    print("Success!!!\n\n")
    print("Loading the pre-trained model...\n\n")
    model = load_model('./max_features_keras.h5')
    print("Success!!!\n\n")
    print("Printing the model...\n\n")
    y_pred = model.predict(x_test)
    print("Success!!!\n\n")
    y_pred_final = np.argmax(y_pred, axis = 1)
    y_test_final = np.argmax(y_test, axis = 1)
    print("Evaluating predictions\n\n")
    test_acc = accuracy_score(y_test_final, y_pred_final)
    print("The accuracy of the model on the test data is:- ")
    print(test_acc * 100)


print("Testing the model on the test data given...\n\n")
# testing_model()
# =============================================================================================