import numpy as np
from preprocessing import img_utils


def mini_batches(x, y, batch_size):
    mini_batch_size = x.shape[0]/batch_size
    print(mini_batch_size)


if __name__ == '__main__':

    img_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train'
    class_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train.csv'

    data, labels = img_utils.load_data(img_path, class_path)

    mini_batches(data, labels, 1024)