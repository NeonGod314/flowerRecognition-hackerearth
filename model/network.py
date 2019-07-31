from __future__ import absolute_import, division, print_function
from preprocessing import img_utils
import tensorflow as tf
import numpy as np


def initialize():
    random_normal = tf.initializers.RandomNormal()
    n_h1 = 50
    n_h2 = 25

    parameters = {

        'h_1': tf.Variable(random_normal([num_features, n_h1]), name='h_1', dtype='float32'),
        'h_2': tf.Variable(random_normal([n_h1, n_h2]), name='h_2'),
        'h_out': tf.Variable(random_normal([n_h2, num_classes]), name='h_out'),
        'b1': tf.Variable(random_normal([n_h1]), name='b_1'),
        'b2': tf.Variable(random_normal([n_h2]), name='b_2'),
        'b_out': tf.Variable(random_normal([num_classes]), name='b_out')

    }
    return parameters


def neural_net(x, parameters):
    layer_1 = tf.add(tf.matmul(x, parameters['h_1']), parameters['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, parameters['h_2']), parameters['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    ##TODO:  test it by changing the syntax
    out_layer = tf.add(tf.matmul(layer_2, parameters['h_out']), parameters['b_out'])

    return tf.nn.softmax(out_layer)


def cross_entropy(y_true, y_pred):
    y_true = tf.one_hot(y_true, num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9,1.)

    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


def accuracy(y_pred, y_true):
    correct_predictions = tf.equal(tf.argmax(y_pred,1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32), axis=-1)


def run_optimization(x, y, parameters, optimizer):

    with tf.GradientTape() as g:
        # print("1")
        # print("a: ", g.watched_variable())
        pred = neural_net(x, parameters)
        loss = cross_entropy(y_true=y, y_pred=pred)
        # print("b: ", g.watched_variables())


    # print(weights.values())
    trainable_variables = [parameters['h_1'], parameters['b1'], parameters['h_2'], parameters['b2'],
                           parameters['h_out'], parameters['b_out']]
    gradients = g.gradient(loss, trainable_variables)
    # print("b4: ", parameters['h_1'].shape)
    # print("now: ", gradients[0].shape)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    # print("c: ", dir(g))
    # print("a4: ", parameters['h_1'].shape)#, parameters['b1'], parameters['h_2'], parameters['b2'], parameters['h_out'], parameters['b_out'])



if __name__ == '__main__':

    train_img_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train'
    train_class_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train.csv'

    test_img_path = '/Users/shubham/Documents/projects/data/flowerRecognition/test'
    test_class_path = '/Users/shubham/Documents/projects/data/flowerRecognition/test.csv'
    #
    train_data, train_label = img_utils.load_data(train_img_path, train_class_path)
    # test_data, test_label = img_utils.load_data(test_img_path, test_class_path)

    num_features = 64 * 64 * 3
    n_h1 = 50
    n_h2 = 25
    num_classes = 102

    display_step = 100

    optimizer = tf.optimizers.SGD(learning_rate=0.01)
    train_data = tf.convert_to_tensor(train_data, dtype='float32')

    for i in range(1000):
        parameters = initialize()
        run_optimization(train_data, train_label, parameters, optimizer)

        if i % display_step == 0:
            print("hey")
            pred = neural_net(train_data, parameters)
            loss = cross_entropy(train_label, pred)
            acc = accuracy(pred, train_label)
            print(("step: %i, loss: %f, accuracy: %f" % (i, loss, acc)))






