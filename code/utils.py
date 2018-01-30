from __future__ import division
import os
import math
import pprint
import scipy.misc
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
#import msgpack
#import pickle

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def mean(x):
    if x is None or len(x) == 0:
        return 0
    return np.mean(x).__float__()

def std(x):
    return np.std(x).__float__()

def dotK(x, y):
    xn = x / np.sqrt(np.sum(x*x, 1, keepdims=True))
    yn = y / np.sqrt(np.sum(y*y, 1, keepdims=True))
    return np.sum(xn*yn, 1)

def get_cross_entropy(prob, ref=None):
    n = prob.shape[-1]
    if ref is None:
        ref = np.ones_like(prob) / n
    return -np.sum(ref * np.log(prob * (1.0 - n * 1e-8) + 1e-8), len(prob.shape)-1)

def remove(path):
    if os.path.exists(path):
        os.remove(path)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def removedirs(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)

def get_image(image_path, iCenterCropSize, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), iCenterCropSize, is_crop, resize_w)

def save_images(images, size, path):
    if images.shape[3] == 1:
        images = np.concatenate([images, images, images], 3)
    return scipy.misc.toimage(merge(images, size), cmin=-1, cmax=1).save(path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imresize(image, resize=1):
    h, w = image.shape[0], image.shape[1]
    img = np.zeros((h * resize, w * resize, image.shape[2]))
    for i in range(h*resize):
        for j in range(w*resize):
            img[i, j] = image[i//resize, j//resize]
    return img

def merge(images, size, resize=3):
    h, w = images.shape[1] * resize, images.shape[2] * resize
    img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = imresize(image, resize)

    return img

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    h, w = x.shape[:2]

    if crop_w is None:
        crop_w = crop_h
    if crop_h == 0:
        crop_h = crop_w = min(h, w)

    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def batch_resize(images, newHeight, newWidth):
    images_resized = np.zeros([images.shape[0], newHeight, newWidth, 3])
    for idx, image in enumerate(images):
        if (images.shape[3] == 1):
            image_3c = []
            image_3c.append(image)
            image_3c.append(image)
            image_3c.append(image)
            image = np.concatenate(image_3c, 2)
        images_resized[idx] = scipy.misc.imresize(image, [newHeight, newWidth], 'bilinear')
    return images_resized

def clip_truncated_normal(mean, stddev, shape, minval=None, maxval=None):
    if minval == None:
        minval = mean - 2 * stddev
    if maxval == None:
        maxval = mean + 2 * stddev
    return np.clip(np.random.normal(mean, stddev, shape), minval, maxval)

def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return (np.array(cropped_image) - 127.5) / 128

def inverse_transform(images):
    return (images+1.)/2.

def plot(data, legend, save_path, miny=None, maxy=None):

    if len(data)<=3:
        plt_nx = 1
        plt_ny = len(data)
    elif len(data) == 4:
        plt_ny = 2
        plt_nx = 2
    else:
        plt_ny = 3
        plt_nx = int(np.ceil(len(data) / 3.0))

    #plt.figure(figsize=(plt_ny*6, plt_nx*4))

    plt.clf()
    for i in range(len(data)):
        plt.subplot(plt_nx, plt_ny, i+1)
        plt.plot(data[i])
        #if miny is not None and maxy is not None:
        #    plt.ylim(miny[i], maxy[i])
        plt.ylabel(legend[i])
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)

class Tweet(object):
    def __init__(self, text=None, userId=None, timestamp=None, location=None):
        self.text = text
        self.userId = userId
        self.timestamp = timestamp
        self.location = location

    def toJSON(self):
        return json.dumps(self.__dict__)

    @classmethod
    def fromJSON(cls, data):
        return cls(**json.loads(data))

    ##def toMessagePack(self):
        return msgpack.packb(self.__dict__)

    #@classmethod
    #def fromMessagePack(cls, data):
    #    return cls(**msgpack.unpackb(data))
