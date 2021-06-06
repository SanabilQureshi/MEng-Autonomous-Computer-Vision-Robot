from silence_tensorflow import silence_tensorflow

silence_tensorflow()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
import GPU_RESET
import shutil
import os
import datetime
from time import *

# from time import ctime, time
GPU_RESET.reset_keras()

fdirs = ['F:/### Sanabil Dissertation/Image Dataset/Blobs/Learning/Train',
         'F:/### Sanabil Dissertation/Image Dataset/Crack/Learning/Train',
         'F:/### Sanabil Dissertation/Image Dataset/Burns/Learning/Train']

# fdirs = ['F:/### Sanabil Dissertation/Image Dataset/BMega/Learning/Train',
#          'F:/### Sanabil Dissertation/Image Dataset/BMedm/Learning/Train']

NumpyData = 'F:/### Sanabil Dissertation/Disso_Scripts/Main_Code/NumpyData'

for fdir in fdirs:
    input_X = f'{NumpyData}' + '/X-Data-' + f'{fdir[-20:-15]}.npy'
    input_y = f'{NumpyData}' + '/y-Data-' + f'{fdir[-20:-15]}.npy'

    X = np.load(input_X)
    y = np.load(input_y)
    X = X / 255.0

    EPOCHS = 10

    optimlist = ['adam', 'SGD', 'RMSprop']
    #
    # dense_layers = [1]
    # layer_sizes = [32]
    # conv_layers = [1,2]
    # optimlist = ['adam']

    conv_layer = 1
    layer_size = 124
    dense_layer = 0

    for optimchoice in optimlist:
        GPU_RESET.reset_keras()
        sleep(5)
        t = time()
        ctime(t)
        datagen = 'VGG Mar 29 20_35_01 2021'
        # NAME = "{}-{}-conv-{}-nodes-{}-dense-{}-optimizer-{}-epochs-{}".format(fdir[-20:-15],conv_layer, layer_size, dense_layer, optimchoice, EPOCHS, str(ctime(t)).replace(":","_"))
        NAME = "{}-{}-conv-{}-nodes-{}-dense-{}-optimizer-{}-epochs-{}".format(fdir[-20:-15], conv_layer, layer_size, dense_layer, optimchoice, EPOCHS,
                                                                               datagen)  # This name is completely arbitrary and only serves to satisfy post-processing conventions

        tensorboard = TensorBoard(log_dir="models/{}/{}.model".format(fdir[-20:-15], NAME))
        csv_logger = CSVLogger('./models/' + f'{fdir[-20:-15]}/' + f'{NAME}.model/' + f'{NAME}.log', append=True)

        print(NAME)

        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:], padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        metriclist = ['accuracy', 'mean_absolute_error', 'mean_absolute_percentage_error']
        checkpoint_filepath = "models/{}/{}.model".format(fdir[-20:-15], NAME)
        model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            metric=metriclist,
            mode='auto',
            save_best_only=False)

        model.compile(loss='binary_crossentropy',
                      optimizer=optimchoice,
                      metrics=metriclist,
                      )

        model.fit(X, y,
                  batch_size=64,
                  epochs=EPOCHS,
                  validation_split=0.1,
                  callbacks=[tensorboard, model_checkpoint_callback, csv_logger])

        model.save("models/{}/{}.model".format(fdir[-20:-15], NAME))

        # GPU_RESET.reset_keras()

    for optimchoice in optimlist:
        GPU_RESET.reset_keras()
        sleep(5)
        t = time()
        ctime(t)
        datagen = 'ALX Mar 29 20_35_01 2021'
        # NAME = "{}-{}-conv-{}-nodes-{}-dense-{}-optimizer-{}-epochs-{}".format(fdir[-20:-15],conv_layer, layer_size, dense_layer, optimchoice, EPOCHS, str(ctime(t)).replace(":","_"))
        NAME = "{}-{}-conv-{}-nodes-{}-dense-{}-optimizer-{}-epochs-{}".format(fdir[-20:-15], conv_layer, layer_size, dense_layer, optimchoice, EPOCHS, datagen)

        tensorboard = TensorBoard(log_dir="models/{}/{}.model".format(fdir[-20:-15], NAME))
        csv_logger = CSVLogger('./models/' + f'{fdir[-20:-15]}/' + f'{NAME}.model/' + f'{NAME}.log', append=True)

        print(NAME)

        model = Sequential()

        model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu', input_shape=X.shape[1:], kernel_initializer='GlorotNormal'))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None))

        model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same', activation='relu', kernel_initializer='GlorotNormal'))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None))

        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='GlorotNormal'))
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='GlorotNormal'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='GlorotNormal'))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None))
        model.add(Flatten())

        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1000, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        metriclist = ['accuracy', 'mean_absolute_error', 'mean_absolute_percentage_error']

        checkpoint_filepath = "models/{}/{}.model".format(fdir[-20:-15], NAME)
        model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            metric=metriclist,
            mode='auto',
            save_best_only=False)

        model.compile(loss='binary_crossentropy',
                      optimizer=optimchoice,
                      metrics=metriclist,
                      )

        model.fit(X, y,
                  batch_size=64,
                  epochs=EPOCHS,
                  validation_split=0.1,
                  callbacks=[tensorboard, model_checkpoint_callback, csv_logger])

        model.save("models/{}/{}.model".format(fdir[-20:-15], NAME))
