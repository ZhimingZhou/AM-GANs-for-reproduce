import scipy.misc as sc
import os
import numpy as np
from matplotlib import pyplot as plt

file_dir = '../dataset/tiny-imagenet-200/val/'
all_label_dir = os.listdir(file_dir)

first_flag = True
label = 1
for label_dir in all_label_dir:
    imgs_of_current_label = os.listdir(file_dir + label_dir + '/images')
    for image_name in imgs_of_current_label:
        img_dir = file_dir + label_dir + '/images/' + image_name
        img = np.expand_dims(sc.imread(img_dir), axis=3)
        if first_flag:
            output_img_matrix = img
            output_label_matrix = np.zeros(shape=1)
            first_flag = False
        else:
            if img.shape[2] == 1:
                img = np.expand_dims(np.repeat(img, 3, axis=2),axis=3)
                print img.shape
                print output_img_matrix.shape
                output_img_matrix = np.concatenate((output_img_matrix, img), axis=3)
                output_label_matrix = np.concatenate((output_label_matrix, np.zeros(shape=1)+1000), axis=0)
            else:
                output_img_matrix = np.concatenate((output_img_matrix, img), axis=3)
                output_label_matrix = np.concatenate((output_label_matrix, np.zeros(shape=1)), axis=0)
    label += 1
    #if label > 2:
    #    break

np.save('tiny-imagenet_image_matrix_test.npy',output_img_matrix)
np.save('tiny-imagenet_label_matrix_test.npy',output_label_matrix)

if False:

    data_X = np.load('../dataset/tiny/tiny-imagenet_image_matrix.npy').astype(float)
    data_Y = np.load('../dataset/tiny/tiny-imagenet_label_matrix.npy').astype(int)
    data_X = data_X.transpose([3, 0, 1, 2])
    data_Y -= 1
    data_X = (data_X[:, ::2, ::2, :] + data_X[:, 1::2, ::2, :] + data_X[:, ::2, 1::2, :] + data_X[:, 1::2, 1::2, :]) / 4.
    # data_X[:, ::2, ::2, :]
    data_X = (data_X - 127.5) / 128.0

    np.save('../dataset/tiny/tiny-imagenet_image_matrix_32_2.npy', data_X)
    np.save('../dataset/tiny/tiny-imagenet_label_matrix_32_2.npy', data_Y)
    exit(0)
