import cv2
import glob
import pandas as pd
import numpy as np
import os


def load_data(img_path, label_path):

    files = glob.glob(img_path+'/*.jpg')
    label_df = pd.read_csv(label_path)

    label_df = label_df.sort_values('image_id', axis=0)
    # assert label_df['image_id'].count() == len(files)

    x = []
    y = []
    # y = label_df['category'].values.reshape(-1)




    for file in files:

        file_label = label_df[label_df['image_id'] == int(os.path.basename(file).rstrip('.jpg'))]
        label = file_label['category'].values[0]

        if label == 2 or label == 1:
            img = cv2.imread(file)
            img = cv2.resize(img, (64, 64)).reshape(-1)
            # img = img/255.0
            x.append(img)
            if label == 2:
                y.append(1)
            else:
                y.append(0)

    x = np.asarray(x)
    y = np.asarray(y)
    x = x/255.
    print(x.shape, y.shape)

    assert x.shape[0] == y.shape[0]

    return x[:170, :], y[:170], x[170:, :], y[170:]

#
# if __name__ == '__main__':
#     img_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train'
#     class_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train.csv'
#
#     load_data(img_path, class_path)
#
#
# import cv2
# import glob
# import pandas as pd
# import numpy as np
# import os
#
#
# def load_data(img_path, label_path):
#
#     files = glob.glob(img_path+'/*.jpg')
#     files.sort()
#
#
#
#     label_df = pd.read_csv(label_path)
#     label_df = label_df.sort_values('image_id', axis=0)
#
#     file_label = label_df[label_df['image_id'] == int(os.path.basename(file).rstrip('.jpg'))]
#     label = file_label['category'].values[0]
#
#     if label == 2 or label == 1:
#
#     # assert label_df['image_id'].count() == len(files)
#
#
#     x = []
#     y = []
#
#     for file in files:
#
#         file_label = label_df[label_df['image_id']==int(os.path.basename(file).rstrip('.jpg'))]
#         label = file_label['category'].values[0]
#
#         if label == 2 or label == 1:
#             # print(label)
#             img = cv2.imread(file)
#             img = cv2.resize(img, (64, 64)).reshape(-1)
#
#             y.append(label)
#             x.append(img)
#
#     x = np.asarray(x)
#     y = np.asarray(y)
#     x = x/255.
#     x = x.reshape(12288,-1)
#     y = y.reshape(-1)
#     print(x.shape, y.shape)
#     print(set(y))
#
#     # assert x.shape[0] == y.shape[0]
#
#     return x, y
#
#
# if __name__ == '__main__':
#     img_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train'
#     class_path = '/Users/shubham/Documents/projects/data/flowerRecognition/train.csv'
#
#     load_data(img_path, class_path)
