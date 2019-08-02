import numpy as np
from preprocessing import img_utils
import tensorflow as tf


def mini_batches(x, y, batch_size):

    n_mini_batch = x.shape[0] // batch_size
    extra_batch = x.shape[0] % batch_size

    mini_batches_x = []
    mini_batches_y = []

    for i in range(n_mini_batch):
        mini_batches_x.append([x[i * batch_size:i * batch_size + batch_size, :, :, :]])
        mini_batches_y.append([y[i * batch_size:i * batch_size + batch_size]])

    mini_batches_x.append([x[n_mini_batch * batch_size:n_mini_batch * batch_size + extra_batch, :, :, :]])
    mini_batches_y.append([y[n_mini_batch * batch_size:n_mini_batch * batch_size + extra_batch]])

    return mini_batches_x, mini_batches_y


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    X = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_weights(cf1, cf2, df, num_classes):
    random_normal = tf.initializers.RandomNormal()
    parameters = {
        'wc1': tf.Variable(random_normal([5, 5, 3, cf1]), name='wc1'),
        'wc2': tf.Variable(random_normal([5, 5, cf1, cf2]), name='wc2'),
        'wd1': tf.Variable(random_normal([7*7*64, df]), name='wd1'),
        'out_w': tf.Variable(random_normal([df, num_classes]), name='out_w'),
        'bc1': tf.Variable(random_normal([cf1]), name='bc1'),
        'bc2': tf.Variable(tf.zeros([cf2]), name='bc2'),
        'bd1': tf.Variable(tf.zeros([df]), name='bd1'),
        'out_b': tf.Variable(tf.zeros([num_classes]), name='out_b')
    }
    print("parameters: ", parameters)

    return  parameters


def cross_entropy(y_pred, y_true, num_classes):
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)

    return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred)))


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))

    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


def conv_net(x, parameters):
    x = tf.reshape(x, [-1,28,28,3])

    conv_1 = conv2d(x, W=parameters['wc1'], b=parameters['bc1'], )
    conv_1 = maxpool2d(conv_1, k=2)

    conv_2 = conv2d(conv_1, W=parameters['wc2'], b=parameters['bc2'])
    conv_2 = maxpool2d(conv_2, k=2)

    fc1 = tf.reshape(conv_2, [-1, 7*7*64])
    fc1 = tf.add(tf.matmul(fc1, parameters['wd1']), parameters['bd1'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, parameters['out_w']), parameters['out_b'])

    return tf.nn.softmax(out)


def run_optimization(x,y, parameters, num_classes, optimizer):
    with tf.GradientTape(persistent=True) as g:
        pred = conv_net(x, parameters=parameters)
        loss = cross_entropy(y_pred=pred, y_true=y, num_classes=num_classes)

    gradients =g.gradient(loss, list(parameters.values()))
    # temp = g.gradient(loss, [parameters['bc1']])
    print("g4:  ", gradients[4], "\ng5:  ", gradients[5], "\ng6:  ", gradients[6])

    optimizer.apply_gradients(zip(gradients, list(parameters.values())))


if __name__ == '__main__':

    img_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train'
    class_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train.csv'

    data, labels = img_utils.load_data(img_path, class_path)

    x_batches, y_batches = mini_batches(data, labels, 128)

    print(len(x_batches))
