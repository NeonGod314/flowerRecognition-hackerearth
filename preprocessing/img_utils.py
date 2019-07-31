import cv2
import glob
import pandas as pd
import numpy as np


def load_data(img_path, label_path):

    files = glob.glob(img_path+'/*.jpg')
    label_df = pd.read_csv(label_path)

    label_df = label_df.sort_values('image_id', axis=0)
    assert label_df['image_id'].count() == len(files)

    x = []
    y = label_df['category'].values.reshape(-1)
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, (64, 64)).reshape(-1)
        # img = img/255.0
        x.append(img)

    x = np.asarray(x)
    x = x/255.
    print(x.shape, y.shape)

    assert x.shape[0] == y.shape[0]

    return x, y


if __name__ == '__main__':
    img_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train'
    class_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train.csv'

    load_data(img_path, class_path)

