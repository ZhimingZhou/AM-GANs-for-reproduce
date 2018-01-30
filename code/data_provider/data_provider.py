import os
import numpy as np
from glob import glob
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from utils import *

class DataProvider(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.data = []
        self.data_X = []
        self.data_Y = []
        self.test_X = []
        self.test_Y = []
        self.num_classes = 1

        if cfg.sDataSet == 'mnist':
            self.data_X, self.data_Y, self.test_X, self.test_Y = self.load_mnist(useX32=cfg.Use32_MNIST)
            self.num_classes = 10
        elif cfg.sDataSet == 'cifar10':
            self.data_X, self.data_Y, self.test_X, self.test_Y = self.load_cifar10()
            self.num_classes = 10
        elif 'toy' in cfg.sDataSet:
            if 'cov' in cfg.sDataSet:
                self.load_toy_data = self.load_toy_data_cov
            elif 'weight' in cfg.sDataSet:
                self.load_toy_data = self.load_toy_data_weight
            self.data_X, self.data_Y, self.test_X, self.test_Y = self.load_toy_data(add_middle=('middle' in cfg.sDataSet))
            self.num_classes = 8 + ('middle' in cfg.sDataSet)
            self.toyf = None
            self.toy_logp = open(cfg.sTestCaseDir + '/toy_-logp.txt', cfg.logModel)
            self.plot_generated_toy_batch(self.data_X[:100000], -1, cfg.sSampleDir)
        elif cfg.sDataSet == 'tiny':
            data_X = np.load('../dataset/tiny/tiny-imagenet_image_matrix_32_2.npy')
            data_Y = np.load('../dataset/tiny/tiny-imagenet_label_matrix_32_2.npy').astype(int)
            self.data_X = []
            self.data_Y = []
            for i in range(len(data_Y)):
                if data_Y[i]<1000:
                    self.data_X.append(data_X[i])
                    self.data_Y.append(data_Y[i])
            self.data_X = np.asarray(self.data_X)
            self.data_Y = np.asarray(self.data_Y)
            self.num_classes = 200
        else:
            self.data = glob(os.path.join('../dataset/', cfg.sDataSet, '*.png'))
            self.num_classes = 1

        if not cfg.bUseClassLabel:
            self.num_classes = cfg.iUnlableClass
            self.data_Y = self.data_Y * 0

        self.data_X_ref = self.data_X
        self.data_X_lab, self.data_Y_lab, self.data_X_unl, self.data_Y_unl = self.part_data(cfg)
        print('data: %i, lab: %i, unl: %i, test: %i' % (len(self.data_X), len(self.data_Y_lab), len(self.data_Y_unl), len(self.test_X)), ', data batchs: ', len(self.data_X) / cfg.iBatchSize)

        if cfg.bAugment and cfg.sDataSet == 'cifar10':
            print('augmenting data for training: flipping')
            self.data_X, self.data_Y = self.augment(self.data_X, self.data_Y)
            self.data_X_lab, self.data_Y_lab = self.augment(self.data_X_lab, self.data_Y_lab)
            self.data_X_unl, self.data_Y_unl = self.augment(self.data_X_unl, self.data_Y_unl)
        print('data: %i, lab: %i, unl: %i, test: %i' % (len(self.data_X), len(self.data_Y_lab), len(self.data_Y_unl), len(self.test_X)), ', data batchs: ', len(self.data_X) / cfg.iBatchSize)

        #print('augmenting data for training: duplicating')
        #self.data_X, self.data_Y = self.duplicate(self.data_X, self.data_Y, cfg.iBatchSize * 1000)
        #self.data_X_lab, self.data_Y_lab = self.duplicate(self.data_X_lab, self.data_Y_lab, cfg.iBatchSize * 1000)
        #self.data_X_unl, self.data_Y_unl = self.duplicate(self.data_X_unl, self.data_Y_unl, cfg.iBatchSize * 1000)
        #print('data: %i, lab: %i, unl: %i, test: %i' % (len(self.data_X), len(self.data_Y_lab), len(self.data_Y_unl), len(self.test_X)), ', data batchs: ', len(self.data_X) / cfg.iBatchSize)

    def duplicate(self, datax, datay, min_count):

        if len(datax) == 0:
            return datax, datay

        dataxs = []
        datays = []
        for i in range(int(np.ceil(float(min_count) / len(datax)))):
            dataxs.append(datax)
            datays.append(datay)

        return np.concatenate(dataxs, axis=0), np.concatenate(datays, axis=0)

    def part_data(self, cfg):

        count = cfg.iNumLabelData

        if count < len(self.data_X):

            select_X_lab = []
            select_Y_lab = []

            remained_X_unl = []
            remained_Y_unl = []

            for j in range(self.num_classes):
                select_X_lab.append(self.data_X[self.data_Y == j][:count])
                select_Y_lab.append(self.data_Y[self.data_Y == j][:count])
                remained_X_unl.append(self.data_X[self.data_Y == j][count:])
                remained_Y_unl.append(self.data_Y[self.data_Y == j][count:])

            select_X_lab = np.concatenate(select_X_lab, axis=0)
            select_Y_lab = np.concatenate(select_Y_lab, axis=0)
            remained_X_unl = np.concatenate(remained_X_unl, axis=0)
            remained_Y_unl = np.concatenate(remained_Y_unl, axis=0)

            return select_X_lab, select_Y_lab, remained_X_unl, remained_Y_unl

        else:
            if cfg.bUseUnlabel:
                return np.copy(self.data_X), np.copy(self.data_Y), np.copy(self.data_X), np.copy(self.data_Y)
            else:
                return np.copy(self.data_X), np.copy(self.data_Y), [], []

    def get_noise_count(self, cfg, count, ordered100=False):

        def get_noise():
            if cfg.bUseUniformZ:
                sample_z = np.random.np.random.uniform(-1, 1, size=(count, cfg.iDimsZ))
            else:
                sample_z = np.random.normal(size=(count, cfg.iDimsZ))
                if cfg.bNormalizeZ:
                    sample_z /= np.linalg.norm(sample_z, axis=1, keepdims=True)
            return sample_z

        sample_z = get_noise()

        if cfg.bPredefined:
            labels_index = np.random.randint(0, self.num_classes, size=[count])

            if ordered100 and count>=100:
                labels_index[:100] = np.asarray([0,1,2,3,4,5,6,7,8,9] * 10).reshape([10, 10]).transpose([1, 0]).reshape(-1)
                sample_z[:100,:] = np.tile(sample_z[:10, :], [10, 1])

            labels_vec = np.zeros([count, self.num_classes])
            labels_vec[range(count), labels_index] += 1.0
            sample_z = np.concatenate([sample_z, labels_vec], 1)

        return sample_z

    def get_noise_batch(self, cfg):
        return self.get_noise_count(cfg, cfg.iBatchSize)

    def shuffle(self, data_x, data_y):
        seed = np.random.randint(10000)
        np.random.seed(seed)
        np.random.shuffle(data_x)
        np.random.seed(seed)
        np.random.shuffle(data_y)

    def load_data_classes(self, label, count):
        image = []
        for i in range(len(self.data_X)):
            if self.data_Y[i] == label:
                image.append(self.data_X[i])
            if len(image) == count:
                break
        return np.asarray(image)

    def load_data_count(self, count):
        assert len(self.data_X) >= count
        #if len(self.data_X) < count: duplecate
        return self.data_X[:count]

    def load_unlabel_batch(self, iBatchSize, idx):

        if iBatchSize > len(self.data_X_unl):
            return None, None

        idx %= len(self.data_X_unl) // iBatchSize

        if idx == 0:
            self.shuffle(self.data_X_unl, self.data_Y_unl)

        batch_images = self.data_X_unl[idx * iBatchSize:(idx + 1) * iBatchSize]
        batch_labels = self.data_Y_unl[idx * iBatchSize:(idx + 1) * iBatchSize]

        if self.num_classes == 1:
            batch_labels = np.zeros([iBatchSize], dtype=np.int32)

        return batch_images, batch_labels

    def load_label_batch(self, iBatchSize, idx):

        if iBatchSize > len(self.data_X_lab):
            return None, None

        idx %= len(self.data_X_lab) // iBatchSize

        if idx == 0:
            self.shuffle(self.data_X_lab, self.data_Y_lab)

        batch_images = self.data_X_lab[idx * iBatchSize:(idx + 1) * iBatchSize]
        batch_labels = self.data_Y_lab[idx * iBatchSize:(idx + 1) * iBatchSize]

        if self.num_classes == 1:
            batch_labels = np.zeros([iBatchSize], dtype=np.int32)

        return batch_images, batch_labels

    def load_test_batch(self, iBatchSize, idx):

        if iBatchSize > len(self.test_X):
            return None, None

        idx %= len(self.test_X) // iBatchSize

        if idx == 0:
            self.shuffle(self.test_X, self.test_Y)

        batch_images = self.test_X[idx * iBatchSize:(idx + 1) * iBatchSize]
        batch_labels = self.test_Y[idx * iBatchSize:(idx + 1) * iBatchSize]

        if self.num_classes == 1:
            batch_labels = np.zeros([iBatchSize], dtype=np.int32)

        return batch_images, batch_labels

    def load_cifar10(self):

        def download_cifar10(data_dir):
            import os, sys, urllib, tarfile
            DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

            dest_directory = data_dir
            makedirs(dest_directory)

            filename = DATA_URL.split('/')[-1]
            filepath = os.path.join(dest_directory, filename)

            remove(filepath)
            removedirs(dest_directory + '/cifar-10-batches-py/')

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))

            filepath, _ = urllib.urlretrieve(DATA_URL, filepath, _progress)
            print('\nSuccesfully downloaded', filename, os.stat(filepath).st_size, 'bytes.')

            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

        def unpickle(file):
            import cPickle
            fo = open(file, 'rb')
            dict = cPickle.load(fo)
            fo.close()
            return dict

        data_dir = '../dataset/cifar-10-batches-py/'
        if not os.path.exists(data_dir):
            download_cifar10('../dataset/')

        try:
            trfilenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in range(1, 6)]
            tefilenames = [os.path.join(data_dir, 'test_batch')]

            data_X = []
            data_Y = []

            test_X = []
            test_Y = []

            for files in trfilenames:
                dict = unpickle(files)
                data_X.append(dict.get('data'))
                data_Y.append(dict.get('labels'))

            for files in tefilenames:
                dict = unpickle(files)
                test_X.append(dict.get('data'))
                test_Y.append(dict.get('labels'))

            data_X = np.concatenate(data_X, 0)
            data_X = np.reshape(data_X, [-1, 3, 1024])
            data_X = np.reshape(data_X, [-1, 3, 32, 32])
            data_X = np.transpose(data_X, [0, 2, 3, 1])
            data_X = (data_X - 127.5) / 128.0
            data_Y = np.concatenate(data_Y, 0)
            data_Y = np.reshape(data_Y, [len(data_Y)]).astype(np.int32)

            test_X = np.concatenate(test_X, 0)
            test_X = np.reshape(test_X, [-1, 3, 1024])
            test_X = np.reshape(test_X, [-1, 3, 32, 32])
            test_X = np.transpose(test_X, [0, 2, 3, 1])
            test_X = (test_X - 127.5) / 128.0
            test_Y = np.concatenate(test_Y, 0)
            test_Y = np.reshape(test_Y, [len(test_Y)]).astype(np.int32)

            return data_X, data_Y, test_X, test_Y

        except Exception as e:
            print('Failed: ' + str(e))
            download_cifar10(data_dir)
            return self.load_cifar10()

    def augment(self, data_X, data_Y):

        if len(data_X) > 0:
            data_X_f = np.fliplr(data_X)
            data_Y_f = data_Y
            return np.concatenate([data_X, data_X_f], 0), np.concatenate([data_Y, data_Y_f], 0)
        else:
            return data_X, data_Y

    def load_mnist(self, useX32=True):

        def download_mnist(data_dir):

            import subprocess
            removedirs(data_dir)
            makedirs(data_dir)

            url_base = 'http://yann.lecun.com/exdb/mnist/'
            file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
            for file_name in file_names:
                url = (url_base + file_name).format(**locals())
                print(url)
                out_path = os.path.join(data_dir, file_name)
                cmd = ['curl', url, '-o', out_path]
                subprocess.call(cmd)
                cmd = ['gzip', '-d', out_path]
                subprocess.call(cmd)

        data_dir = '../dataset/mnist/'
        if not os.path.exists(data_dir):
            download_mnist(data_dir)

        try:
            fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

            fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            trY = loaded[8:].reshape((60000)).astype(np.float)

            fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

            fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            teY = loaded[8:].reshape((10000)).astype(np.float)

            trY = np.asarray(trY)
            teY = np.asarray(teY)

            if useX32:
                trX32 = np.zeros([len(trX), 32, 32, 1])
                trX32[:, 2:30, 2:30] = trX
                trX = trX32

                teX32 = np.zeros([len(teX), 32, 32, 1])
                teX32[:, 2:30, 2:30] = teX
                teX = teX32

            trY = (np.reshape(trY, [len(trY)])).astype(np.int32)
            teY = (np.reshape(teY, [len(teY)])).astype(np.int32)

            return (trX - 127.5) / 128.0, trY, (teX - 127.5) / 128.0, teY

        except Exception as e:
            print('Failed: ' + str(e))

            download_mnist(data_dir)
            return self.load_mnist(useX32)

    def load_toy_data_cov(self, n_mixture=8, std=0.01, radius=1.0, pts_per_mixture=1000000, add_middle=False):

        thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
        self.xs, self.ys = radius * np.sin(thetas), radius * np.cos(thetas)
        self.cov = std * np.eye(2)
        self.std = []
        self.weight = []

        X = np.zeros(((n_mixture + 0) * pts_per_mixture, 2))
        Y = np.zeros(((n_mixture + 0) * pts_per_mixture))

        for i in range(n_mixture):
            mean = np.array([self.xs[i], self.ys[i]])
            pts = np.random.multivariate_normal(mean, self.cov if i%2==0 else self.cov / 4, pts_per_mixture)
            X[i * pts_per_mixture: (i + 1) * pts_per_mixture, :] = pts
            Y[i * pts_per_mixture: (i + 1) * pts_per_mixture] = i
            self.std.append(std if i%2==0 else std / 4)
            self.weight.append(1.0/n_mixture)

        if add_middle:
            mean = np.array([0, 0])
            pts = np.random.multivariate_normal(mean, self.cov, pts_per_mixture)
            X[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture, :] = pts
            Y[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture] = n_mixture

        self.shuffle(X, Y)

        return X, Y.astype(np.int32), np.copy(X), np.copy(Y).astype(np.int32)

    def load_toy_data_weight(self, n_mixture=8, std=0.01, radius=1.0, pts_per_mixture=1000000, add_middle=False):

        self.std = []
        thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
        self.xs, self.ys = radius * np.sin(thetas), radius * np.cos(thetas)
        self.cov = std * np.eye(2)
        self.std = []
        self.weight = []

        X = np.zeros((5 * pts_per_mixture, 2))
        Y = np.zeros((5 * pts_per_mixture))
        tolsamples = 0
        for i in range(n_mixture):
            mean = np.array([self.xs[i], self.ys[i]])
            cursamples = pts_per_mixture if i % 2 == 0 else pts_per_mixture // 4
            pts = np.random.multivariate_normal(mean, self.cov, cursamples)
            X[tolsamples:tolsamples + cursamples] = pts
            Y[tolsamples:tolsamples + cursamples] = i
            self.std.append(std)
            self.weight.append(1.0 / 5 if i%2 == 0 else 1.0 / 20)

        if add_middle:
            mean = np.array([0, 0])
            pts = np.random.multivariate_normal(mean, self.cov, pts_per_mixture)
            X[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture, :] = pts
            Y[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture] = n_mixture

        self.shuffle(X, Y)

        return X, Y.astype(np.int32), np.copy(X), np.copy(Y).astype(np.int32)

    def load_toy_data(self, n_mixture=8, std=0.01, radius=1.0, pts_per_mixture=1000000, add_middle=False):

        thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
        self.xs, self.ys = radius * np.sin(thetas), radius * np.cos(thetas)
        self.cov = std * np.eye(2)
        self.std = []
        self.weight = []

        X = np.zeros(((n_mixture + 0) * pts_per_mixture, 2))
        Y = np.zeros(((n_mixture + 0) * pts_per_mixture))

        for i in range(n_mixture):
            mean = np.array([self.xs[i], self.ys[i]])
            pts = np.random.multivariate_normal(mean, self.cov, pts_per_mixture)
            X[i * pts_per_mixture: (i + 1) * pts_per_mixture, :] = pts
            Y[i * pts_per_mixture: (i + 1) * pts_per_mixture] = i
            self.std.append(std)
            self.weight.append(1.0/n_mixture)

        if add_middle:
            mean = np.array([0, 0])
            pts = np.random.multivariate_normal(mean, self.cov, pts_per_mixture)
            X[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture, :] = pts
            Y[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture] = n_mixture

        self.shuffle(X, Y)

        return X, Y.astype(np.int32), np.copy(X), np.copy(Y).astype(np.int32)

    def plot_generated_toy_batch(self, X_gen, gen_iter, dir, discriminator=None, sess=None):

        xmin, xmax = -1.5, 1.5
        ymin, ymax = -1.5, 1.5
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

        if self.toyf is None:
            data = self.load_toy_data(pts_per_mixture=1000)[0]
            x = data[:, 0]
            y = data[:, 1]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x, y])
            kernel = stats.gaussian_kde(values)
            self.toyf = np.reshape(kernel(positions).T, xx.shape)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.gca()
        ax.contourf(xx, yy, self.toyf, cmap='Blues', vmin=np.percentile(self.toyf, 80), vmax=np.max(self.toyf), levels=np.linspace(0.25, 0.85, 30))

        if discriminator is not None and False:
            delta = 0.025
            xmin, xmax = -1.5, 1.5
            ymin, ymax = -1.5, 1.5
            XX, YY = np.meshgrid(np.arange(xmin, xmax, delta), np.arange(ymin, ymax, delta))
            arr_pos = np.vstack((np.ravel(XX), np.ravel(YY))).T
            ZZ = discriminator(arr_pos)
            ZZ = ZZ.reshape(XX.shape)
            ax.contour(XX, YY, ZZ, cmap="Blues", levels=np.linspace(0.25, 0.85, 10))

        logp = 0
        for i in range(X_gen.shape[0]):
            p = 0
            for k in range(len(self.xs)):
                p += np.exp(-(np.square(X_gen[i][0]-self.xs[k]) + np.square(X_gen[i][1]-self.ys[k])) / 2 / self.std[k] / self.std[k]) / np.sqrt(2*3.1415926) / self.std[k] * self.weight[k]
            logp += np.log(p+1e-8)
        logp /= -X_gen.shape[0]

        plt.scatter(X_gen[:5000, 0], X_gen[:5000, 1], s=0.5, color="coral", marker="o")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        #import matplotlib
        #font = {'size': 18}
        #matplotlib.rc('font', **font)

        plt.savefig(dir + "/toy_dataset_iter%s.pdf" % gen_iter)
        plt.clf()
        plt.close()

        if gen_iter % 1000 == 0:
            self.toy_logp.write('%d %f\n' % (gen_iter, logp))
            self.toy_logp.flush()

        print('%f,' % logp)
        return logp
