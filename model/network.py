from __future__ import absolute_import, division, print_function
from preprocessing import img_utils
import tensorflow as tf
from utils import utils


def initialize(num_features, num_classes):

    n_h1 = 50
    n_h2 = 25

    parameters = {

        'h_1': tf.Variable(random_normal([num_features, n_h1]), name='h_1', dtype='float32'),
        'h_2': tf.Variable(random_normal([n_h1, n_h2]), name='h_2'),
        'h_out': tf.Variable(random_normal([n_h2, num_classes]), name='h_out'),
        'b1': tf.Variable(random_normal([1, n_h1]), name='b_1'),
        'b2': tf.Variable(random_normal([1, n_h2]), name='b_2'),
        'b_out': tf.Variable(random_normal([1, num_classes]), name='b_out')

    }

    return parameters


def neural_net(x, parameters):
    layer_1 = tf.add(tf.matmul(x, parameters['h_1']), parameters['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, parameters['h_2']), parameters['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    out_layer = tf.add(tf.matmul(layer_2, parameters['h_out']), parameters['b_out'])

    return tf.nn.softmax(out_layer)


def cross_entropy(y_true, y_pred, n_classes):
    y_true = tf.one_hot(y_true, n_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)

    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


def accuracy(y_pred, y_true):
    y_pred = tf.reshape(y_pred, [-1, 2])
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))

    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32), axis=-1)


def run_optimization(x, y, parameters, optimizer):

    with tf.GradientTape() as g:
        pred = neural_net(x, parameters)
        loss = cross_entropy(y_true=y, y_pred=pred, n_classes=num_classes)

    gradients = g.gradient(loss, list(parameters.values()))
    optimizer.apply_gradients(zip(gradients, list(parameters.values())))


if __name__ == '__main__':

    train_img_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train'
    train_class_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train.csv'

    test_img_path = '/Users/shubham/Documents/projects/data/flowerRecognition/test'
    test_class_path = '/Users/shubham/Documents/projects/data/flowerRecognition/test.csv'

    train_data, train_label, test_data, test_label = img_utils.load_data(train_img_path, train_class_path)

    num_features = 64 * 64 * 3
    n_h1 = 50
    n_h2 = 25
    num_classes = 2
    learning_rate = 0.0001

    display_step = 300

    random_normal = tf.initializers.RandomNormal()

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

    train_data = tf.convert_to_tensor(train_data, dtype='float32')
    test_data = tf.convert_to_tensor(test_data, dtype='float32')

    train_data_batches, train_label_batches = utils.mini_batches(train_data, train_label, 32)
    parameters = initialize(num_features, num_classes)

    for i in range(30000):

        for train_data_batch, train_label_batch in zip(train_data_batches, train_label_batches):

            run_optimization(train_data_batch, train_label_batch, parameters, optimizer)

        if i % display_step == 0:
            print("EPOCH:: ", i)
            pred = neural_net(train_data, parameters)
            loss = cross_entropy(train_label, pred, n_classes=num_classes)

            acc = accuracy(pred, train_label)

            print(("step: %i, loss: %f, accuracy: %f" % (i, loss, acc)))

            pred = neural_net(test_data, parameters)
            loss = cross_entropy(test_label, pred, n_classes=num_classes)
            acc = accuracy(pred, test_label)
            print(("step: %i, loss: %f, accuracy: %f" % (i, loss, acc)))
