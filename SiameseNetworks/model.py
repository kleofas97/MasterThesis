import numpy as np
from typing import Tuple
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import cv2


def genereate_batch(path_1: str, path_2: str, batch_size: int) -> Tuple:
    """Generate batch of images from two files, where name corresponds to each other. Label is the last number in the name of the picture"""
    x = []
    labels = []
    dir_list_x1 = os.listdir(path_1)
    batch_count = 0
    while True:
        for imgp in dir_list_x1:
            label = imgp[-5]  # eg. "SampleNb_1.png, that is why [-5] is "1"
            #read img
            p1 = cv2.imread(os.path.join(path_1, imgp))
            #preprocess - color and normalization for both images
            p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
            p1 = p1 * (1. / 255.0)
            p1 = p1.reshape(p1.shape[0], p1.shape[1], 1)
            p1 = p1.astype(np.float32)
            p2 = cv2.imread(os.path.join(path_2, imgp))
            p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)
            p2 = p2 * (1. / 255.0)
            p2 = p2.reshape(p2.shape[0], p2.shape[1], 1)
            p2 = p2.astype(np.float32)
            x += [[p1, p2]]
            labels += [label]
            batch_count += 1
            if batch_count > batch_size - 1:
                apairs = np.array(x, dtype=object).astype('float32')
                alabels = np.array(labels).astype('float32')
                yield [apairs[:, 0], apairs[:, 1]], alabels
                x.clear()
                labels.clear()
                batch_count = 0


def create_base_model(input_dim):
    """Create model's siamese branch based on input dimensions size"""
    inputs = Input(shape=input_dim)
    conv_1 = Conv2D(64, (5, 5), padding="same", activation='relu', name='conv_1')(inputs)
    conv_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2 = Conv2D(128, (5, 5), padding="same", activation='relu', name='conv_2')(conv_1)
    conv_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_3')(conv_2)
    conv_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    conv_4 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_4')(conv_3)
    conv_5 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_5')(conv_4)
    conv_5 = MaxPooling2D(pool_size=(2, 2))(conv_5)

    dense_1 = Flatten()(conv_5)
    dense_1 = Dense(512, activation="relu")(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(512, activation="relu")(dense_1)
    dense_2 = Dropout(0.5)(dense_2)
    return Model(inputs, dense_2)


def built_model(input_shape):
    """create a keras model"""
    base_network = create_base_model(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    fc6 = concatenate([processed_a, processed_b])
    fc7 = Dense(1024, activation='relu')(fc6)
    fc8 = Dense(1024, activation='relu')(fc7)
    fc9 = Dense(1, activation='sigmoid')(fc8)
    model = Model([input_a, input_b], fc9)
    model.summary()
    return model


def fit_model(model, path_to_model: str, generator_train: Tuple, train_steps: int,
              generator_val: Tuple, val_steps: int, epochs: int,
              learning_rate: float, patch_size: int):
    """Fit the model"""
    callbacks = [
        EarlyStopping(patience=5, verbose=1),
        ReduceLROnPlateau(patience=1, verbose=1, monitor='val_loss'),
        ModelCheckpoint(os.path.join(path_to_model, str(patch_size) + 'px_bestmodel.h5py'),
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min'),
        CSVLogger(os.path.join(path_to_model,'log'))
    ]

    adam = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(generator_train,
                        steps_per_epoch=train_steps,
                        epochs=epochs,
                        validation_data=generator_val,
                        validation_steps=val_steps, shuffle=False,
                        callbacks=callbacks)
    del model
    model = load_model(os.path.join(path_to_model, str(patch_size) + 'px_bestmodel.h5py'))
    return model, history
