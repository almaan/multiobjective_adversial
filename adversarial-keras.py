#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
# import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import utils


def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    z_train = np.array([y % 2 == 0 for y in y_train])
    z_test = np.array([y % 2 == 0 for y in y_test])

    # convert class vectors to binary class matrices
    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)
    z_train = utils.to_categorical(z_train, 2)
    z_test = utils.to_categorical(z_test, 2)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(y_train[0:10])
    print(y_test[0:10])

    input = Input(shape=(784, ), name='mnist_input')
    x = Dense(150, activation='sigmoid')(input)
    y = Dropout(0.1)(x)
    f = Dense(50, activation='sigmoid')(y)

    g = Dense(10, activation='softmax', name='digit_pred')(f)
    h = Dense(2, activation='softmax', name='parity_pred')(f)


    model = Model(inputs=[input], outputs=[g, h])

    # note that the loss weight for the digit prediction is negative one!
    model.compile(optimizer='rmsprop',
                  loss={'digit_pred': 'categorical_crossentropy', 'parity_pred': 'binary_crossentropy'},
                  loss_weights={'digit_pred': -1., 'parity_pred': 1})

    # And trained it via:
    model.fit({'mnist_input': x_train}, {'digit_pred': y_train, 'parity_pred': z_train}, epochs=100, batch_size=32,
            verbose=1, validation_data=(x_test, {'digit_pred': y_test, 'parity_pred': z_test}))

    score = model.predict(x_test)
    print(score)

    pd.DataFrame(score[0]).to_csv("pred-digits.txt.gz", sep="\t", compression='gzip')
    pd.DataFrame(score[1]).to_csv("pred-parity.txt.gz", sep="\t", compression='gzip')
    pd.DataFrame(y_test).to_csv("test-digits.txt.gz", sep="\t", compression='gzip')
    pd.DataFrame(z_test).to_csv("test-parity.txt.gz", sep="\t", compression='gzip')

if __name__ == "__main__":
    main()
