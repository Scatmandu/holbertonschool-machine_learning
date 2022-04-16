#!/usr/bin/env python3
"""0x00-convolutional_neural_networks"""
import tensorflow as tf


def lenet5(x, y):
    """
        method that builds LeNet-5 architecture using tensorflow

        x = a tf.placeholder of shape (m, 28, 28, 1) containing
            the input images for the network

            m = the number of images

        y = a tf.placeholder of shape (m, 10) containing
            the one-hot labels for the network

            m = the number of images

        Return: a tensor for the softmax activated output
                a training operation that utilizes Adam
                    optimization (with default hyperparameters)
                a tensor for the loss of the netowrk
                a tensor for the accuracy of the network
    """
    init = tf.initializers.he_normal()
    c1 = tf.layers.Conv2D(6, kernel_size=(5, 5), padding='same',
                          activation='relu',
                          kernel_initializer=init)(x)
    p1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)
    c2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                          activation='relu',
                          kernel_initializer=init)(p1)
    p2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)
    flat = tf.layers.Flatten()(p2)
    layer1 = tf.layers.Dense(units=120, kernel_initializer=init,
                             activation='relu')(flat)
    layer2 = tf.layers.Dense(units=84, kernel_initializer=init,
                             activation='relu')(layer1)
    layer3 = tf.layers.Dense(units=10, kernel_initializer=init)(layer2)
    prediction = layer3
    softmax = tf.nn.softmax(prediction)
    loss = tf.losses.softmax_cross_entropy(y, layer3)
    accuratePred = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(accuratePred, tf.float32))
    adam = tf.train.AdamOptimizer().minimize(loss)
    return (softmax, adam, loss, accuracy)
