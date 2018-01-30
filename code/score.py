from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import sys
import tarfile

import numpy as np
import tensorflow as tf

from utils import makedirs, removedirs, remove


import time
from msssim import MultiScaleSSIM
from utils import *

class MSSSIM:

    def _tf_fspecial_gauss(self, size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """

        radius = size // 2
        offset = 0.0
        start, stop = -radius, radius + 1
        if size % 2 == 0:
            offset = 0.5
            stop -= 1

        x_data, y_data = np.mgrid[offset + start:stop, offset + start:stop]  # -size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        assert len(x_data) == size

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / tf.reduce_sum(g)

    def tf_ssim(self, img1, img2, L=2, mean_metric=True, filter_size=11, sigma=1.5):

        _, height, width, _ = img1.shape.as_list()

        size = min(filter_size, height, width)
        sigma = size / 11.0 * sigma

        window = self._tf_fspecial_gauss(size, sigma)  # window shape [size, size]

        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        def filter(img):
            return tf.concat([tf.nn.conv2d(img[:,:,:,0:1], window, strides=[1, 1, 1, 1], padding='VALID'),tf.nn.conv2d(img[:,:,:,1:2], window, strides=[1, 1, 1, 1], padding='VALID'),tf.nn.conv2d(img[:,:,:,2:3], window, strides=[1, 1, 1, 1], padding='VALID')], 3)

        mu1 = filter(img1)
        mu2 = filter(img2)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = filter(img1 * img1)- mu1_sq
        sigma2_sq = filter(img2 * img2) - mu2_sq
        sigma12 = filter(img1 * img2) - mu1_mu2
        sigma12 = tf.abs(sigma12)

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = v1 / v2
        ssim = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1) * cs

        if mean_metric:
            ssim = tf.reduce_mean(ssim)
            cs = tf.reduce_mean(cs)

        return ssim, cs

    def tf_ms_ssim(self, img1, img2, mean_metric=True, level=5):

        #assert level >= 1 and level <= 5
        weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
        #weight = weight[-level:] / tf.reduce_sum(weight[-level:])
        #weight = tf.ones([level], dtype=tf.float32) / level
        #weight = tf.Print(weight, [weight])

        mssim = []
        mcs = []
        for l in range(level):
            ssim, cs = self.tf_ssim(img1, img2, mean_metric=True)
            mssim.append(ssim)
            mcs.append(cs)
            img1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            img2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # list to tensor of dim D+1
        mssim = tf.stack(mssim, axis=0)
        mcs = tf.stack(mcs, axis=0)

        value = (tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) * (mssim[level - 1] ** weight[level - 1]))

        if mean_metric:
            value = tf.reduce_mean(value)

        return value

    def __init__(self):

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.ms_ssim_graph = tf.Graph()
        self.ms_ssim_sess = tf.Session(config=config, graph=self.ms_ssim_graph)

        with self.ms_ssim_graph.as_default():
            self.image1 = tf.placeholder(tf.float32, shape=[1, 32, 32, 3])
            self.image2 = tf.placeholder(tf.float32, shape=[1, 32, 32, 3])
            self.msssim_index = self.tf_ms_ssim(self.image1, self.image2)

        self.ref_scores = None

    def msssim_ratio(self, classes_score, ref_scores):
        if ref_scores == None:
            return np.mean(np.asarray(classes_score) / classes_score)
        return np.mean([min(x, 1) for x in np.asarray(ref_scores) / classes_score])

    def msssim(self, class_images, num_classes, count):

        class_images += 1

        if False:
            classes_score = []
            for i in range(num_classes):
                scores = []
                for x1 in range(count//2):
                    x2 = (x1 + count) % len(class_images)
                    score = self.ms_ssim_sess.run(self.msssim_index, feed_dict={self.image1: class_images[count * i + x1:count * i + x1 + 1], self.image2: class_images[count * i + x2:count * i + x2 + 1]})
                    #score = MultiScaleSSIM(class_images[count * i + x1:count * i + x1 + 1], class_images[(count * i + x2) % len(class_images): (count * i + x2 + 1) % len(class_images)], 2)
                    scores.append(score)
                classes_score.append(scores)
            #print([std(scores) for scores in classes_score])
            print([mean(scores) for scores in classes_score])

        classes_score = []
        for i in range(num_classes):
            scores = []
            for x1 in range(count//2):
                x2 = count - x1 - 1
                score = self.ms_ssim_sess.run(self.msssim_index, feed_dict={self.image1: class_images[count * i + x1:count * i + x1 + 1], self.image2: class_images[count * i + x2:count * i + x2 + 1]})
                #score = MultiScaleSSIM(class_images[count * i + x1:count * i + x1 + 1], class_images[count * i + x2:count * i + x2 + 1], 2)
                scores.append(score)
            classes_score.append(scores)
        #print([std(scores) for scores in classes_score])
        print([mean(scores) for scores in classes_score])

        return [mean(scores) for scores in classes_score], self.msssim_ratio([mean(scores) for scores in classes_score], self.ref_scores)

    def set_ref_images(self, ref_imags, num_classes, count):
        self.ref_scores = self.msssim(ref_imags, num_classes, count)[0]
        print('ref_score: ' + np.array2string(np.asarray(self.ref_scores), formatter={'float_kind': lambda x: "%.5f" % x}))

    def set_ref_scores(self, ref_scores):
        self.ref_scores = ref_scores
        print('ref_score: ' + np.array2string(np.asarray(self.ref_scores), formatter={'float_kind': lambda x: "%.5f" % x}))

def inception_score_KL(preds):
    preds = preds + 1e-18
    inps_avg_preds = np.mean(preds, 0, keepdims=True)
    inps_KLs = np.sum(preds * (np.log(preds) - np.log(inps_avg_preds)), 1)
    inception_score = np.exp(np.mean(inps_KLs))
    return inception_score


def inception_score_H(preds): # inception_score_KL == inception_score_H
    preds = preds + 1e-18
    inps_avg_preds = np.mean(preds, 0)
    H_per = np.mean(-np.sum(preds * np.log(preds), 1))
    H_avg = -np.sum(inps_avg_preds * np.log(inps_avg_preds), 0)
    return np.exp(H_avg - H_per), H_per, H_avg


def inception_score_split_std(icp_preds, split_n=10):
    icp_preds = icp_preds + 1e-18
    scores = []
    for i in range(split_n):
        part = icp_preds[(i * icp_preds.shape[0] // split_n):((i + 1) * icp_preds.shape[0] // split_n), :]
        scores.append(inception_score_H(part)[0])
    return np.mean(scores), np.std(scores), scores


def am_score(preds, ref_preds):

    icp_score, icp_per, icp_avg = inception_score_H(preds) # icp with accordingly pre-trained classifer

    preds = preds + 1e-18
    avg_preds = np.mean(preds, 0)
    am_per = np.mean(-np.sum(preds * np.log(preds), 1))
    am_avg = -np.sum(ref_preds * np.log(avg_preds / ref_preds), 0)
    #am_score = am_per - am_avg
    am_score = am_per + am_avg

    return am_score, am_per, am_avg, icp_score, icp_per, icp_avg, avg_preds


def am_score_split_std(preds, ref_preds, split_n=10):
    scores = []
    for i in range(split_n):
        part = preds[(i * preds.shape[0] // split_n):((i + 1) * preds.shape[0] // split_n), :]
        scores.append(am_score(part, ref_preds)[0])
    return np.mean(scores), np.std(scores), scores


class InceptionScore:

    def __init__(self):

        self.batch_size = 50 # It does not effect the accuracy. Small batch size need less memory while bit slower

        self.inception_softmax_w = None
        self.inception_softmax_b = None

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.inception_graph = tf.Graph()
        self.inception_sess = tf.Session(config=config, graph=self.inception_graph)

        self._init_inception()

        self.count = 0

    def get_preds(self, inps):
        inps_s = np.array(inps) * 128.0 + 128

        icp_preds_w = []
        icp_preds_b = []
        f_batches = int(math.ceil(float(inps_s.shape[0]) / float(self.batch_size)))
        for i in range(f_batches):
            inp = inps_s[(i * self.batch_size): min((i + 1) * self.batch_size, inps_s.shape[0])]
            pred_w, pred_b = self.inception_sess.run([self.inception_softmax_w, self.inception_softmax_b], {'ExpandDims:0': inp})
            icp_preds_w.append(pred_w)
            icp_preds_b.append(pred_b)
            # sys.stdout.write("\rInception score evaluating: %.2f" % (float(i) / f_batches))
        # sys.stdout.write("\n")
        icp_preds_w = np.concatenate(icp_preds_w, 0)
        icp_preds_b = np.concatenate(icp_preds_b, 0)

        return icp_preds_w, icp_preds_b

    def get_inception_score(self, inps, split_n=10):  #10000 is quite good, 50000 suggested

        icp_preds_w, icp_preds_b = self.get_preds(inps)

        print('inception_score_split_w:' + ' '.join('%.2f' % score for score in inception_score_split_std(icp_preds_w)[2]))
        print('inception_score_split_b:' + ' '.join('%.2f' % score for score in inception_score_split_std(icp_preds_b)[2]))

        return inception_score_split_std(icp_preds_w, split_n)[:2] + inception_score_split_std(icp_preds_b, split_n)[:2] + inception_score_H(icp_preds_w) + inception_score_H(icp_preds_b)

    def _init_inception(self):

        MODEL_DIR = '../pretrained_model/inception/'
        DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

        makedirs(MODEL_DIR)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb')):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))

            import urllib
            filepath, _ = urllib.urlretrieve(DATA_URL, filepath, _progress)
            print('\nSuccesfully downloaded', filename, os.stat(filepath).st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)

        with self.inception_graph.as_default():
            with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')

        # Works with an arbitrary minibatch size.
        pool3 = self.inception_sess.graph.get_tensor_by_name('pool_3:0')
        ops = self.inception_sess.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o._shape = tf.TensorShape(new_shape)

        w = self.inception_graph.get_tensor_by_name("softmax/weights:0")
        output = tf.matmul(tf.squeeze(pool3), w)
        self.inception_softmax_w = tf.nn.softmax(output)

        b = self.inception_graph.get_tensor_by_name("softmax/biases:0")
        output = tf.add(output, b)
        self.inception_softmax_b = tf.nn.softmax(output)

class AMScore:

    def __init__(self, cfg):

        self.cfg = cfg

        self.am_model = None

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.am_graph = tf.Graph()
        self.am_sess = tf.Session(config=config, graph=self.am_graph)

        self.ref_am_preds = []

        self._init_AMScore()

    def set_ref(self, refs):

        if len(self.ref_am_preds) == 0:

            if 'toy' in self.cfg.sDataSet:

                self.batch_size = 128

                if True:
                    self.ref_am_preds = np.ones(8) / 8.0  # calculate from data will get very similar result
                else:
                    refs_preds = []
                    f_batches = int(math.ceil(float(refs.shape[0]) / float(self.batch_size)))
                    for i in range(f_batches):
                        ref = refs[(i * self.batch_size):min((i + 1) * self.batch_size, refs.shape[0])]
                        ref_pred = self.am_sess.run(self.am_model.prediction, feed_dict={self.am_model.input: ref})
                        refs_preds.append(ref_pred)
                        sys.stdout.write(".")
                        sys.stdout.flush()
                        if i % 100 == 99: sys.stdout.write("\n")
                    sys.stdout.write("\n")
                    refs_preds = np.concatenate(refs_preds, 0)
                    self.ref_am_preds = np.mean(refs_preds, 0)
            else:

                assert self.cfg.sDataSet == 'cifar10' or self.cfg.sDataSet == 'tiny'
                self.ref_am_preds = np.mean(self.get_preds(refs), 0)

    def get_am_score(self, inps):

        am_preds = []

        if self.cfg.sDataSet == 'cifar10' or self.cfg.sDataSet == 'tiny':
            am_preds = self.get_preds(inps)
        else:
            assert 'toy' in self.cfg.sDataSet
            f_batches = inps.shape[0] // self.batch_size
            for i in range(f_batches):
                inp = inps[(i * self.batch_size): min((i + 1) * self.batch_size, inps.shape[0])]
                pred = self.am_sess.run(self.am_model.prediction, feed_dict={self.am_model.input: inp})
                am_preds.append(pred)
                if i % 10 == 0: sys.stdout.write(".")
            sys.stdout.write("\n")
            am_preds = np.concatenate(am_preds, 0)

        print('am_score_split:' + ' '.join('%.2f' % score for score in am_score_split_std(am_preds, self.ref_am_preds)[2]))

        return am_score(am_preds, self.ref_am_preds)

    def _init_AMScore(self):

        if 'toy' in self.cfg.sDataSet:

            from am_model.toyc import ToyClassifier

            with self.am_graph.as_default():
                self.am_model = ToyClassifier()
                self.am_model.build_model(8, cfg=self.cfg)
                saver = tf.train.Saver()

            ckpt_state = tf.train.get_checkpoint_state('../pretrained_model/toyc/')
            if ckpt_state and ckpt_state.model_checkpoint_path:
                tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
                saver.restore(self.am_sess, ckpt_state.model_checkpoint_path)

        else:

            with self.am_graph.as_default():
                import argparse

                from am_model.dense_net import DenseNet
                from data_provider.utils import get_data_provider_by_name

                train_params_cifar = {
                    'batch_size': 64,
                    'n_epochs': 300,
                    'initial_learning_rate': 0.1,
                    'reduce_lr_epoch_1': 150,  # epochs * 0.5
                    'reduce_lr_epoch_2': 225,  # epochs * 0.75
                    'validation_set': True,
                    'validation_split': None,  # None or float
                    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
                    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
                    'save_path': '../dataset/',
                }

                train_params_svhn = {
                    'batch_size': 64,
                    'n_epochs': 40,
                    'initial_learning_rate': 0.1,
                    'reduce_lr_epoch_1': 20,
                    'reduce_lr_epoch_2': 30,
                    'validation_set': True,
                    'validation_split': None,  # you may set it 6000 as in the paper
                    'shuffle': True,  # shuffle dataset every epoch or not
                    'normalization': 'divide_255',
                }

                def get_train_params_by_name(name):
                    if name in ['C10', 'C10+', 'C100', 'C100+', 'tiny']:
                        return train_params_cifar
                    if name == 'SVHN':
                        return train_params_svhn

                parser = argparse.ArgumentParser()
                parser.add_argument(
                    '--train', action='store_true',
                    help='Train the model')
                parser.add_argument(
                    '--test', action='store_true',
                    help='Test model for required dataset if pretrained model exists.'
                         'If provided together with `--train` flag testing will be'
                         'performed right after training.')
                parser.add_argument(
                    '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
                    default='DenseNet-BC',
                    help='What type of model to use')
                parser.add_argument(
                    '--growth_rate', '-k', type=int, choices=[12, 24, 40],
                    default=40,
                    help='Grows rate for every layer, '
                         'choices were restricted to used in paper')
                parser.add_argument(
                    '--depth', '-d', type=int, choices=[40, 100, 190, 250],
                    default=40,
                    help='Depth of whole network, restricted to paper choices')
                parser.add_argument(
                    '--dataset', '-ds', type=str,
                    choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN', 'tiny'],
                    default='C10+',
                    help='What dataset should be used')
                parser.add_argument(
                    '--total_blocks', '-tb', type=int, default=3, metavar='',
                    help='Total blocks of layers stack (default: %(default)s)')
                parser.add_argument(
                    '--keep_prob', '-kp', type=float, metavar='',
                    help="Keep probability for dropout.")
                parser.add_argument(
                    '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
                    help='Weight decay for optimizer (default: %(default)s)')
                parser.add_argument(
                    '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
                    help='Nesterov momentum (default: %(default)s)')
                parser.add_argument(
                    '--reduction', '-red', type=float, default=0.5, metavar='',
                    help='reduction Theta at transition layer for DenseNets-BC models')

                parser.add_argument(
                    '--logs', dest='should_save_logs', action='store_true',
                    help='Write tensorflow logs')
                parser.add_argument(
                    '--no-logs', dest='should_save_logs', action='store_false',
                    help='Do not write tensorflow logs')
                parser.set_defaults(should_save_logs=True)

                parser.add_argument(
                    '--saves', dest='should_save_model', action='store_true',
                    help='Save model during training')
                parser.add_argument(
                    '--no-saves', dest='should_save_model', action='store_false',
                    help='Do not save model during training')
                parser.set_defaults(should_save_model=True)

                parser.add_argument(
                    '--renew-logs', dest='renew_logs', action='store_true',
                    help='Erase previous logs for model if exists.')
                parser.add_argument(
                    '--not-renew-logs', dest='renew_logs', action='store_false',
                    help='Do not erase previous logs for model if exists.')
                parser.set_defaults(renew_logs=True)

                args = parser.parse_args(args=[])

                if self.cfg.sDataSet == 'tiny':
                    args.dataset = 'tiny'

                if not args.keep_prob:
                    if args.dataset in ['C10', 'C100', 'SVHN']:
                        args.keep_prob = 0.8
                    else:
                        args.keep_prob = 1.0
                if args.model_type == 'DenseNet':
                    args.bc_mode = False
                    args.reduction = 1.0
                elif args.model_type == 'DenseNet-BC':
                    args.bc_mode = True

                args.test = True
                if not args.train and not args.test:
                    print("You should train or test your network. Please check params.")
                    exit()

                model_params = vars(args)

                # some default params dataset/architecture related
                train_params = get_train_params_by_name(args.dataset)
                self.data_provider = get_data_provider_by_name(args.dataset, train_params)
                self.model = DenseNet(data_provider=self.data_provider, **model_params)

            if self.cfg.sDataSet == 'tiny':
                self.model.restore_model('../pretrained_model/densenet/DenseNet-BC_growth_rate=40_depth=40_dataset_tiny/model.chkpt')
            else:
                self.model.restore_model('../pretrained_model/densenet/DenseNet-BC_growth_rate=40_depth=40_dataset_C10+/model.chkpt')

    def get_preds(self, images, labels=None):

        from data_provider.cifar import CifarDataSet

        if labels == None:
            if self.cfg.sDataSet == 'tiny':
                labels = np.random.rand(images.shape[0], 200)
            else:
                labels = np.random.rand(images.shape[0], 10)

        self.data_provider.test = CifarDataSet(
            images=images, labels=labels,
            shuffle=None, n_classes=self.data_provider.n_classes,
            normalization='by_chanels',
            augmentation=False)

        predictions, _, _ = self.model.test(self.data_provider.test, batch_size=200)

        return predictions


class AMScore3:

    def __init__(self, cfg):

        self.cfg = cfg

        self.am_model = None

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.am_graph = tf.Graph()
        self.am_sess = tf.Session(config=config, graph=self.am_graph)

        self.ref_am_preds = []

        self._init_AMScore()

    def set_ref(self, refs):

        if len(self.ref_am_preds) == 0:

            if 'toy' in self.cfg.sDataSet:

                self.batch_size = 128

                if True:
                    self.ref_am_preds = np.ones(8) / 8.0  # calculate from data will get very similar result
                else:
                    refs_preds = []
                    f_batches = int(math.ceil(float(refs.shape[0]) / float(self.batch_size)))
                    for i in range(f_batches):
                        ref = refs[(i * self.batch_size):min((i + 1) * self.batch_size, refs.shape[0])]
                        ref_pred = self.am_sess.run(self.am_model.prediction, feed_dict={self.am_model.input: ref})
                        refs_preds.append(ref_pred)
                        sys.stdout.write(".")
                        sys.stdout.flush()
                        if i % 100 == 99: sys.stdout.write("\n")
                    sys.stdout.write("\n")
                    refs_preds = np.concatenate(refs_preds, 0)
                    self.ref_am_preds = np.mean(refs_preds, 0)
            else:

                assert self.cfg.sDataSet == 'cifar10'
                self.batch_size = 100

                refs_preds = []
                f_batches = int(math.ceil(float(refs.shape[0]) / float(self.batch_size)))
                for i in range(f_batches):
                    ref = refs[(i * self.batch_size):min((i + 1) * self.batch_size, refs.shape[0])]
                    ref_pred = self.am_sess.run(self.am_model.predictions, feed_dict={self.image_holder: ref, self.label_holder: np.zeros([100, 10])})
                    refs_preds.append(ref_pred)
                    if i % 100 == 99: sys.stdout.write("\n")
                    sys.stdout.write(".")
                sys.stdout.write("\n")
                refs_preds = np.concatenate(refs_preds, 0)
                self.ref_am_preds = np.mean(refs_preds, 0)

    def get_preds(self, inps):

        am_preds = []
        f_batches = inps.shape[0] // self.batch_size
        for i in range(f_batches):
            inp = inps[(i * self.batch_size): min((i + 1) * self.batch_size, inps.shape[0])]
            if self.cfg.sDataSet == 'cifar10':
                pred = self.am_sess.run(self.am_model.predictions,
                                        feed_dict={self.image_holder: inp, self.label_holder: np.zeros([100, 10])})
            else:
                assert 'toy' in self.cfg.sDataSet
                pred = self.am_sess.run(self.am_model.prediction, feed_dict={self.am_model.input: inp})
            am_preds.append(pred)
            if i % 10 == 0: sys.stdout.write(".")
        sys.stdout.write("\n")
        am_preds = np.concatenate(am_preds, 0)

        return am_preds

    def get_am_score(self, inps):

        am_preds = self.get_preds(inps)

        return am_score(am_preds, self.ref_am_preds)

    def _init_AMScore(self):

        if 'toy' in self.cfg.sDataSet:

            from am_model.toyc import ToyClassifier

            with self.am_graph.as_default():
                self.am_model = ToyClassifier()
                self.am_model.build_model(8, cfg=self.cfg)
                saver = tf.train.Saver()

            ckpt_state = tf.train.get_checkpoint_state('../pretrained_model/toyc/')
            if ckpt_state and ckpt_state.model_checkpoint_path:
                tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
                saver.restore(self.am_sess, ckpt_state.model_checkpoint_path)

        else:

            from am_model import resnet_model

            depth = 3
            image_size = 32
            batch_size = 100
            num_classes = 10

            with self.am_graph.as_default():
                hps = resnet_model.HParams(batch_size=batch_size, num_classes=num_classes, min_lrn_rate=0.0001, lrn_rate=0.1, num_residual_units=5, use_bottleneck=False, weight_decay_rate=0.0002, relu_leakiness=0.1, optimizer='mom')

                self.image_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, depth], name='image')
                self.label_holder = tf.placeholder(tf.float32, [batch_size, 10], name='label')

                self.am_model = resnet_model.ResNet(hps, self.image_holder, self.label_holder, 'evl')
                self.am_model.build_graph()
                saver = tf.train.Saver()

            MODEL_DIR = '../pretrained_model/cifar10/'
            tarfilename = 'cifar10_pretrained.tar.gz'
            tarfilepath = os.path.join(MODEL_DIR, tarfilename)

            def download_model(tarfilepath):
                DATA_URL = 'https://goo.gl/mSJpRX' #python download not work.
                remove(tarfilepath)
                removedirs(MODEL_DIR)
                makedirs(MODEL_DIR)

                def _progress(count, block_size, total_size):
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % (tarfilename, float(count * block_size) / float(total_size) * 100.0))

                cmd = ['curl', DATA_URL, '-o', tarfilepath]
                import subprocess
                subprocess.call(cmd)
                #import urllib
                #tarfilepath, _ = urllib.urlretrieve(DATA_URL, tarfilepath, _progress)
                print('\nSuccesfully downloaded', tarfilename, os.stat(tarfilepath).st_size, 'bytes.')
                tarfile.open(tarfilepath, 'r:gz').extractall(MODEL_DIR)

            ckpt_name = 'model.ckpt-805241'
            if not os.path.exists(MODEL_DIR + ckpt_name + '.index'):
                download_model(tarfilepath)

            try:
                tf.logging.info('Loading checkpoint %s', MODEL_DIR + ckpt_name)
                saver.restore(self.am_sess, MODEL_DIR + ckpt_name)
            except Exception as e:
                print('Failed: ' + str(e))
                download_model(tarfilepath)
                tf.logging.info('Loading checkpoint %s', MODEL_DIR + ckpt_name)
                saver.restore(self.am_sess, MODEL_DIR + ckpt_name)


