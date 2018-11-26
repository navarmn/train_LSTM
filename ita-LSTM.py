import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint, EarlyStopping
from tensorflow import keras
import time
from sklearn import preprocessing
import os

# SET max GPU usage:

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = "0"
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))



SEQ_LEN = 100  # how long of a preceeding sequence to collect for RNN
# FUTURE_PERIOD_PREDICT = 1  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "LTC-USD"
MODEL_ARCHITECTURE = "4LSTM_2DENSE"
NEURONS = "50_50_50_50_32_4"
APPENDIX = "LOSS-MSE"
EPOCHS = 100  # how many passes through our data
BATCH_SIZE = 124  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{MODEL_ARCHITECTURE}-NEURONS-{NEURONS}-{APPENDIX}-{int(time.time())}"

if not os.path.exists(f'models/{NAME}'):
    os.system(f'mkdir models/{NAME}')

def preprocess_df(df):
    # df = df.drop("future", 1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "Health_state":  # normalize all ... except for the target itself!
            # df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            # df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.


    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        # print(i[-1])
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!
            # print(f'This is the one the gets appended: {i[-1]}')

    # random.shuffle(sequential_data)  # shuffle for good measure.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!

def train_model(X_train, y_train, X_val, y_val):
    model = Sequential()

    # model.add(CuDNNLSTM(50, batch_input_shape=(124, 100, 12), return_sequences=True, stateful=True))
    model.add(CuDNNLSTM(50, input_shape=(train_x.shape[1:]), return_sequences=True))
    # model.add(CuDNNLSTM(10, input_shape=(train_x.shape[1:])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(50))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # model.add(Dense(64, activation='tanh'))
    # model.add(Dropout(0.2))

    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.2))

    model.add(Dense(4, activation='tanh'))


    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        # loss='categorical_crossentropy',
        loss='mse',
        optimizer=opt,
        metrics=['accuracy']
    )

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("models/{}/{}.model".format(NAME, filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=100, verbose=1, mode='auto')

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard, checkpoint, early_stopping],
    )

    # Score model
    score = model.evaluate(X_val, y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save model
    model.save("models/{}/{}.model".format(NAME, NAME))

    return None

if __name__ == "__main__":
    
    main_df = pd.DataFrame() # begin empty

    labels_name = 'data_preprocessed_cleaned_labels'
    data_name = 'data_preprocessed_cleaned'

    data = pd.read_csv(f'training_datas/{data_name}.csv', index_col=0)
    labels = pd.read_csv(f'training_datas/{labels_name}.csv', index_col=0, names=['Health_state'])

    main_df = data.join(labels) 
    main_df['Health_state'] = main_df['Health_state'].sub(1)

    times = sorted(main_df.index.values)  # get the times
    last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]  # get the last 5% of the times

    validation_main_df = main_df[(main_df.index >= last_5pct)]  # make the validation data where the index is in the last 5%
    main_df = main_df[(main_df.index < last_5pct)]  # now the main_df is all the data up to the l


    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_main_df)

    # one-hot encoding on labels:

    train_y_hot = keras.utils.to_categorical(train_y, num_classes=4)
    validation_y_hot = keras.utils.to_categorical(validation_y, num_classes=4)


    print(f"train data: {len(train_x)} validation: {len(validation_x)}")
    print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
    print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

    ##
    # With one-hot-encoding on labels:
    train_model(train_x, train_y_hot, validation_x, validation_y_hot)

    # Without one-hot-encoding on labels:
    # train_model(train_x, train_y, validation_x, validation_y)

    ##
    pass

