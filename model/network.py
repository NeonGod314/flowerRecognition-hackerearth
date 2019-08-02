from __future__ import absolute_import, division, print_function
from preprocessing import img_utils
import tensorflow as tf
from utils import utils


if __name__ == '__main__':

    train_img_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train'
    train_class_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train.csv'

    test_img_path = '/Users/shubham/Documents/projects/data/flowerRecognition/test'
    test_class_path = '/Users/shubham/Documents/projects/data/flowerRecognition/test.csv'

    train_data, train_label, test_data, test_label = img_utils.load_data(train_img_path, train_class_path)

    num_classes = 2
    learning_rate = 0.001

    display_step = 50

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

    train_data = tf.convert_to_tensor(train_data, dtype='float32')
    test_data = tf.convert_to_tensor(test_data, dtype='float32')

    train_data_batches, train_label_batches = utils.mini_batches(train_data, train_label, 20)

    parameters = utils.conv_weights(cf1=32, cf2=64, df=1024, num_classes=num_classes)

    for i in range(3000):

        for train_data_batch, train_label_batch in zip(train_data_batches, train_label_batches):
            # print(len(train_data_batch), len(train_label_batch))
            # print("before optim: ", train_data_batch[0].shape, train_label_batch[0].shape)

            utils.run_optimization(train_data_batch, train_label_batch, parameters, num_classes=num_classes, optimizer=optimizer)

        if i % display_step == 0:
            print("EPOCH:: ", i)
            pred = utils.conv_net(train_data, parameters)
            loss = utils.cross_entropy(pred, train_label, num_classes=num_classes)

            acc = utils.accuracy(pred, train_label)

            print(("step: %i, loss: %f, accuracy: %f" % (i, loss, acc)))

            pred = utils.conv_net(test_data, parameters)
            loss = utils.cross_entropy(pred, test_label, num_classes=num_classes)
            acc = utils.accuracy(pred, test_label)
            print(("step: %i, loss: %f, accuracy: %f" % (i, loss, acc)))
