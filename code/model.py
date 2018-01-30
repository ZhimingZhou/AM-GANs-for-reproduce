from __future__ import division
from __future__ import print_function

import collections
import locale
import time
import cPickle as pickle

from tensorflow.contrib.layers.python.layers.layers import layer_norm
from tensorflow.contrib.opt.python.training.nadam_optimizer import NadamOptimizer

from ops_wn import *
from data_provider.data_provider import DataProvider
from score import *
from utils import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

debug_first_n = 0
max_checkpoint_num = 100

class DCGAN(object):

    ################################################################# Traning ##############################################################################

    def train(self, cfg):

        print('\n\ninitialization')

        data = self.data

        if 'toy' not in cfg.sDataSet:
            save_images(self.sample_images[:self.save_size * self.save_size], [self.save_size, self.save_size], '{}/train_{:02d}_{:04d}.png'.format(cfg.sSampleDir + '/fixed_noise', 0, 0))
            save_images((self.sample_images + clip_truncated_normal(0, cfg.fInputNoise + 0.00001, self.sample_images.shape))[:self.save_size * self.save_size], [self.save_size, self.save_size], '{}/noise{:02f}_{:02d}_{:04d}.png'.format(cfg.sSampleDir + '/fixed_noise', cfg.fInputNoise, 0, 0))

        self.initilization_model(cfg)

        ############################################################################################################################################

        try:
            f = open(cfg.checkpoint_dir + '/store2.pickle', 'rb')
            msssim_scores_col = pickle.load(f)
            if len(msssim_scores_col[0]) == 1:
                msssim_scores_col = msssim_scores_col[0]
            f.close()
        except:
            msssim_scores_col = []

        try:
            f = open(cfg.checkpoint_dir + '/store.pickle', 'rb')
            g_decay_col, d_decay_col, counter_col, am_score_col, icp_score_col = pickle.load(f)

            if cfg.sDataSet == 'cifar10':

                am_score_col_ = []
                for am_score in am_score_col:
                    am_score = list(am_score)
                    if abs(am_score[0] - am_score[1] - am_score[2]) > 0.01:
                        am_score[1] = am_score[4]
                        am_score[2] = am_score[2] + np.log(0.1)
                        am_score[0] = am_score[1] + am_score[2]
                    am_score_col_.append(tuple(am_score))
                am_score_col = am_score_col_

                icp_score_col_ = []
                for i, icp_score in enumerate(icp_score_col):
                    icp_score_ = list(icp_score)
                    icp_score_[10] = am_score_col[i][0]
                    icp_score_col_.append(tuple(icp_score_))
                icp_score_col = icp_score_col_

            f.close()

        except:
            g_decay_col = []
            d_decay_col = []
            counter_col = []
            am_score_col = []
            icp_score_col = []

        if len(icp_score_col) < len(msssim_scores_col):
            msssim_scores_col = []

        def collect(X, x, len):
            if isinstance(x, np.ndarray):
                if x.shape.__len__() == 1:
                    x = x.reshape((1,) + x.shape)
                return x if X is None else np.concatenate([X, x], 0)[-len:]
            else:
                return [x] if X is None else (X + [x])[-len:]

        def gen_image(fake_vecKss, generated_images, zs, z):

            generated_image, fake_vecK = self.sess.run([self.images_gen, self.fake_vecK], feed_dict={self.z: z, self.fInputNoise: fInputNoise})

            if cfg.sDataSet == 'mnist' and cfg.Use32_MNIST:
                generated_image = generated_image[:, 2:30, 2:30, :]

            if fake_vecKss is None:
                zs = z
                fake_vecKss = fake_vecK
                generated_images = generated_image
            else:
                zs = np.concatenate([zs, z])
                fake_vecKss = np.concatenate([fake_vecKss, fake_vecK])
                generated_images = np.concatenate([generated_images, generated_image])

            return fake_vecKss, generated_images, zs

        def gen_class_image(fake_vecKss, generated_images, zs, rcount, maxgen):

            class_images = []
            for i in range(self.num_classes):
                j = 0
                count = 0
                while count < rcount:
                    if generated_images is None or j >= generated_images.shape[0] - 1:
                        if generated_images is not None and generated_images.shape[0] > maxgen:
                            while count < rcount:
                                class_images.append(generated_images[np.random.randint(0, generated_images.shape[0])])
                                count += 1
                        else:
                            fake_vecKss, generated_images, zs = gen_image(fake_vecKss, generated_images, zs, data.get_noise_batch(cfg))
                    if (cfg.bPredefined and np.argmax(zs[j, -10:]) == i) or (not cfg.bPredefined and np.argmax(fake_vecKss[j]) == i):
                        class_images.append(generated_images[j])
                        count += 1
                    j += 1

            return np.asarray(class_images)

        fLrG = fLrD = 0
        test_accK = 0
        score_time = 0
        g_losses = d_losses = d_decaylosses = g_decaylosses = g_GNs = None
        fake_vec1s = fake_vecKs = real_vec1s = real_vecKs = fake_targets = real_targets = mode_vecs = None
        d_loss_ac = d_loss_gan = 0

        ############################################################################################################################################

        print('\n')
        print('optimizor starting')

        counter = load_counter = int(self.g_global_step.eval(self.sess))

        problog = open(cfg.sTestCaseDir+'/prob.txt', cfg.logModel)

        D_optimized = False
        fake_vec1 = None

        while counter <= cfg.iBatchRun:

            counter += 1

            start_time = time.time()

            fInputNoise = (cfg.fInputNoise - cfg.fInputNoiseMin) * (1.0 - counter / min(100000, cfg.iBatchRun)) ** cfg.iInputNoisePow + cfg.fInputNoiseMin

            G_optimized = False

            while not G_optimized or not D_optimized:

                for k_d in range(cfg.iTrainD if counter > cfg.iWarmD else cfg.iWarmDIterPer):

                    if mean(fake_vec1) < cfg.fLimitedD:
                        continue

                    D_optimized = True

                    batch_z = data.get_noise_batch(cfg)
                    feed_dict = {self.z: batch_z, self.fInputNoise: fInputNoise}

                    if cfg.bUseClassLabel:
                        batch_images_lab, batch_labels_lab = data.load_label_batch(cfg.iBatchSize, counter)
                    else:
                        batch_images_lab, batch_labels_lab = data.load_unlabel_batch(cfg.iBatchSize, counter)

                    feed_dict.update({self.images_lab: batch_images_lab, self.lab_image_labels: batch_labels_lab})

                    _, d_loss, d_loss_gan, d_loss_ac, d_decayloss, fLrD, \
                    fake_vec1, fake_vecK, fake_target, mode_vec, real_vec1, real_vecK, real_target\
                        = self.sess.run([self.d_optim, self.d_loss_total, self.d_loss_gan, self.d_loss_ac, self.d_decayloss, self.fLrD,
                                         self.fake_vec1, self.fake_vecK, self.fake_target, self.mode_vec, self.real_vec1, self.real_vecK, self.real_target], feed_dict=feed_dict)

                    d_losses = collect(d_losses, d_loss, 1000)
                    d_decaylosses = collect(d_decaylosses, d_decayloss, 1000)
                    mode_vecs = collect(mode_vecs, mode_vec, 1000)

                    fake_vec1s = collect(fake_vec1s, fake_vec1, 1000)
                    fake_vecKs = collect(fake_vecKs, fake_vecK, 1000)
                    fake_targets = collect(fake_targets, fake_target, 1000)

                    real_vec1s = collect(real_vec1s, real_vec1, 1000)
                    real_vecKs = collect(real_vecKs, real_vecK, 1000)
                    real_targets = collect(real_targets, real_target, 1000)

                for k_g in range(cfg.iTrainG):

                    if mean(fake_vec1) > cfg.fLimitedG:
                        continue

                    G_optimized = True

                    batch_z = data.get_noise_batch(cfg)

                    if cfg.GN or np.mod(counter, 100) == 0:

                        _, g_GN, g_loss, g_decayloss, generated_images, fLrG, fake_vec1, fake_vecK, fake_target, mode_vec \
                            = self.sess.run([self.g_optim, self.g_GN, self.g_loss_total, self.g_decayloss, self.images_gen, self.fLrG,
                                             self.fake_vec1, self.fake_vecK, self.fake_target, self.mode_vec],
                                            feed_dict={self.z: batch_z, self.fInputNoise: fInputNoise})
                        g_GNs = collect(g_GNs, g_GN, 100)

                    else:

                        _, g_loss, g_decayloss, generated_images, fLrG, fake_vec1, fake_vecK, fake_target, mode_vec \
                            = self.sess.run([self.g_optim, self.g_loss_total, self.g_decayloss, self.images_gen, self.fLrG,
                                             self.fake_vec1, self.fake_vecK, self.fake_target, self.mode_vec],
                                            feed_dict={self.z: batch_z, self.fInputNoise: fInputNoise})

                    g_losses = collect(g_losses, g_loss, 1000)
                    g_decaylosses = collect(g_decaylosses, g_decayloss, 1000)
                    mode_vecs = collect(mode_vecs, mode_vec, 1000)

                    fake_vec1s = collect(fake_vec1s, fake_vec1, 1000)
                    fake_vecKs = collect(fake_vecKs, fake_vecK, 1000)
                    fake_targets = collect(fake_targets, fake_target, 1000)

            end_time = time.time()

            ############################################################################################################################################

            if np.mod(counter, int(cfg.iIterCheckpoint // 100)) == 0:
                d_decay_col.append(mean(g_decaylosses))
                g_decay_col.append(mean(d_decaylosses))

            if np.mod(counter, cfg.iIterCheckpoint) == 0 or cfg.bLoadForEvaluation: # or (np.mod(counter, cfg.iIterCheckpoint // 10) == 0 and counter < cfg.iIterCheckpoint):

                var_list = tf.trainable_variables()

                def penalty(value, thre):
                    return float(np.sum((np.maximum(np.abs(value), thre) - thre) ** 2)) / 2

                for var in var_list:
                    value = var.eval(self.sess)
                    print(var.op.name + ' [mean:' + '%.2f,' % np.mean(np.abs(value)) + ' std:' + '%.2f,' % np.std(np.abs(value)) + ' max:' + '%.2f]' % np.max(np.abs(value)) + ' [abs10: ' + '%.2f,' % penalty(value, 10) + ' abs1: ' + '%.2f,' % penalty(value, 1) + ' abs0: ' + '%.2f]' % penalty(value, 0))

                plt.figure()
                plt.plot(np.asarray([d_decay_col, g_decay_col]).transpose([1, 0]))
                plt.legend(['d_decay_col', 'g_decay_col'])
                plt.savefig(cfg.sSampleDir + '/decay_loss.png')
                plt.close()

                if cfg.iTrainG > 0:

                    counter_col.append(counter)

                    score_start_time = time.time()

                    iSamplesEvaluate = cfg.iSamplesEvaluate // 10 if np.mod(counter, cfg.iIterCheckpoint // 10) == 0 and counter < cfg.iIterCheckpoint else cfg.iSamplesEvaluate

                    zs = None
                    fake_vecKss = None
                    generated_images = None

                    for i in range(int(np.ceil(iSamplesEvaluate / cfg.iBatchSize))):
                        fake_vecKss, generated_images, zs = gen_image(fake_vecKss, generated_images, zs, data.get_noise_batch(cfg))

                    if True and cfg.bUseClassLabel and cfg.sDataSet == 'cifar10':
                        class_images = gen_class_image(fake_vecKss, generated_images, zs, cfg.iSSIM, cfg.iSSIM * self.num_classes * 10)
                        msssim_scores = self.MSSSIM.msssim(class_images, self.num_classes, cfg.iSSIM)

                        msssim_scores_col.append(msssim_scores)
                        for i, msssim_scores in enumerate(msssim_scores_col):
                            print('[%d] ' % (counter_col[-len(msssim_scores_col) + i] // 10000) + np.array2string(np.asarray(msssim_scores[0]), formatter={'float_kind': lambda x: "%.2f" % x}) + ' final: %.2f' % msssim_scores[1])

                    if cfg.sDataSet == 'cifar10' or cfg.sDataSet == 'tiny' or 'toy' in cfg.sDataSet:

                        self.am_score.set_ref(data.data_X_ref)
                        am_score = self.am_score.get_am_score(generated_images)

                        am_score_col.append(am_score)

                        for i, am_score in enumerate(am_score_col):
                            print('[%d] ' % (counter_col[-len(am_score_col) + i] // 10000) + 'am_avg_probs: ' + ', '.join(['%.2f'] * len(am_score[-1])) % tuple(am_score[-1]))

                        for i, am_score in enumerate(am_score_col):
                            print('[%d] ' % (counter_col[-len(am_score_col) + i] // 10000) + 'am_score: %.3f, am_per: %.3f, am_avg: %.3f, am_icp_score: %.3f, am_icp_per: %.3f, am_icp_avg: %.3f' % tuple(am_score[:6]))

                        plot([[am_score[0] for am_score in am_score_col], [am_score[1] for am_score in am_score_col], [am_score[2] for am_score in am_score_col]], ['am_score', 'am_per', 'am_avg'], cfg.sSampleDir + '/am_score.png')
                        plot([[am_score[3] for am_score in am_score_col], [am_score[4] for am_score in am_score_col], [am_score[5] for am_score in am_score_col]], ['am_icp_score', 'am_icp_per', 'am_icp_avg'], cfg.sSampleDir + '/am_icp_score.png')

                    if cfg.bLoadForEvaluation or cfg.iIterCheckpoint == 1:
                        exit(0)

                    if cfg.sDataSet == 'cifar10' or cfg.sDataSet == 'tiny':

                        if cfg.sDataSet == 'cifar10':
                            icp_score = tuple(list(self.icp_score.get_inception_score(generated_images)) + list(am_score_col[-1][:-1]))
                        else:
                            icp_score = tuple(list(self.icp_score.get_inception_score(generated_images)))

                        icp_score_col.append(icp_score)

                        for i, icp_score in enumerate(icp_score_col):
                            print('[%d] ' % (counter_col[-len(icp_score_col) + i] // 10000) + 'icp_mean_std_w: %.3f, %.3f, icp_mean_std_b: %.3f, %.3f' % tuple(icp_score[:4]))

                        for i, icp_score in enumerate(icp_score_col):
                            print('[%d] ' % (counter_col[-len(icp_score_col) + i] // 10000) + 'icp_score_w: %.3f, icp_per_w: %.3f, icp_avg_w: %.3f, icp_score_b: %.3f, icp_per_b: %.3f, icp_avg_b: %.3f' % tuple(icp_score[4:10]))

                        plot([[icp_score[4] for icp_score in icp_score_col], [icp_score[5] for icp_score in icp_score_col], [icp_score[6] for icp_score in icp_score_col]], ['icp_score_w', 'icp_per_w', 'icp_avg_w'], cfg.sSampleDir + '/icp_score_w.png')
                        plot([[icp_score[7] for icp_score in icp_score_col], [icp_score[8] for icp_score in icp_score_col], [icp_score[9] for icp_score in icp_score_col]], ['icp_score_b', 'icp_per_b', 'icp_avg_b'], cfg.sSampleDir + '/icp_score_b.png')

                        if cfg.sDataSet == 'cifar10':
                            plot([[icp_score[4] for icp_score in icp_score_col], [icp_score[7] for icp_score in icp_score_col], [icp_score[10] for icp_score in icp_score_col], [icp_score[13] for icp_score in icp_score_col]], ['icp_score_w', 'icp_score_b', 'am_score', 'am_icp_score'], cfg.sSampleDir + '/the_scores.png')
                            plot([[icp_score[5] for icp_score in icp_score_col], [icp_score[8] for icp_score in icp_score_col], [icp_score[11] for icp_score in icp_score_col], [icp_score[14] for icp_score in icp_score_col]], ['icp_per_w', 'icp_per_b', 'am_per', 'am_icp_per'], cfg.sSampleDir + '/the_per_scores.png')
                            plot([[icp_score[6] for icp_score in icp_score_col], [icp_score[9] for icp_score in icp_score_col], [icp_score[12] for icp_score in icp_score_col], [icp_score[15] for icp_score in icp_score_col]], ['icp_avg_w', 'icp_avg_b', 'am_avg', 'am_icp_avg'], cfg.sSampleDir + '/the_avg_scores.png')

                    score_time = time.time() - score_start_time

                    cfg.bLoadForEvaluation = False

            if np.mod(counter, cfg.iIterCheckpoint // 10) == 0:
                f = open(cfg.checkpoint_dir + '/store.pickle', 'wb')
                pickle.dump([g_decay_col, d_decay_col, counter_col, am_score_col, icp_score_col], f)
                f.close()
                f = open(cfg.checkpoint_dir + '/store2.pickle', 'wb')
                pickle.dump(msssim_scores_col, f)
                f.close()
                self.save(cfg, int(np.ceil((counter / cfg.iIterCheckpoint))) * cfg.iIterCheckpoint)

            ############################################################################################################################################

            if np.mod(counter, cfg.iIterCheckpoint) == 0 and cfg.bUseLabel and cfg.sDataSet == 'cifar10':

                test_vec1s = test_vecKs = test_targets = None

                for idx in range(len(data.test_X) // cfg.iBatchSize):
                    batch_images_lab, batch_labels_lab = data.load_test_batch(cfg.iBatchSize, idx)

                    test_vec1, test_vecK, test_target = self.sess.run([self.real_vec1, self.real_vecK, self.real_target], feed_dict={self.images_lab: batch_images_lab, self.lab_image_labels: batch_labels_lab, self.fInputNoise: fInputNoise})

                    test_vec1s = collect(test_vec1s, test_vec1, 50000)
                    test_vecKs = collect(test_vecKs, test_vecK, 50000)
                    test_targets = collect(test_targets, test_target, 50000)

                print("[validation] t_vec1: %.3f %.3f t_maxK: %.3f %.3f t_dotK: %.3f %.3f t_accK: %.3f " % (mean(test_vec1s), std(test_vec1s), mean(np.max(test_vecKs, 1)), std(np.max(test_vecKs, 1)), mean(dotK(test_vecKs, test_targets)), std(dotK(test_vecKs, test_targets)), mean(np.argmax(test_vecKs,1)==np.argmax(test_targets,1))))

                test_accK = mean(np.argmax(test_vecKs, 1) == np.argmax(test_targets, 1))

            ############################################################################################################################################

            # plot samples
            if 'toy' in cfg.sDataSet:

                if np.mod(counter, 1000) == 0:

                    generated_images = self.sess.run(self.images_gen, feed_dict={self.z: self.sample_z[:cfg.iBatchSize]})
                    for i in range(100):
                        batch_z = data.get_noise_batch(cfg)
                        generated_image = self.sess.run(self.images_gen, feed_dict={self.z: batch_z})
                        generated_images = np.concatenate([generated_images, generated_image])

                    curent_logp = data.plot_generated_toy_batch(generated_images, counter, cfg.sSampleDir)
                    icp_score_col[-1] = curent_logp

            else:

                # fixed noise
                if np.mod(counter, 1000) == 0 or (np.mod(counter, 100) == 0 and counter < 1000) or (np.mod(counter, 10) == 0 and counter < 100):

                    zs = None
                    fake_vecKss = None
                    generated_images = None

                    for i in range(int(np.ceil(float(cfg.iSaveCount) / cfg.iBatchSize))):
                        fake_vecKss, generated_images, zs = gen_image(fake_vecKss, generated_images, zs, self.sample_z[i * cfg.iBatchSize:i * cfg.iBatchSize + cfg.iBatchSize])

                    save_images(generated_images[:self.save_size * self.save_size], [self.save_size, self.save_size],
                                '{}/train_{:02d}_{:04d}.png'.format(cfg.sSampleDir + '/fixed_noise', counter // 10000, counter % 10000))

                # random sample
                if np.mod(counter, 1000) == 0 or (np.mod(counter, 100) == 0 and counter < 1000) or (np.mod(counter, 10) == 0 and counter < 100):

                    zs = None
                    fake_vecKss = None
                    generated_images = None

                    for i in range(int(np.ceil(float(cfg.iSaveCount) / cfg.iBatchSize))):
                        fake_vecKss, generated_images, zs = gen_image(fake_vecKss, generated_images, zs, data.get_noise_batch(cfg))

                    if cfg.bPredefined:
                        ct = 0.0
                        for i in range(generated_images.shape[0]):
                            if np.argmax(zs[i, -10:]) == np.argmax(fake_vecKss[i]):
                                ct += 1
                        print('Predefined class label macth ratio: %.2f' % (ct / generated_images.shape[0]))

                    class_images = gen_class_image(fake_vecKss, generated_images, zs, cfg.iSaveCount // self.num_classes, cfg.iSaveCount * 10)

                    save_images(class_images[:(cfg.iSaveCount // self.num_classes) * self.num_classes], [cfg.iSaveCount // self.num_classes, self.num_classes],
                                '{}/train_{:02d}_{:04d}.png'.format(cfg.sSampleDir + '/class_random', counter // 10000, counter % 10000))
                    save_images(class_images[:(cfg.iSaveCount // self.num_classes) * self.num_classes], [cfg.iSaveCount // self.num_classes, self.num_classes],
                                '{}/sample.png'.format(cfg.sSampleDir))

                    plt.figure()
                    plt.ylim(0.0, 1.0)
                    plt.plot(np.asarray([np.squeeze((fake_vec1s[:real_vec1s.shape[0]])[-1000:]), np.squeeze(real_vec1s[-1000:])]).transpose([1, 0]))
                    plt.legend(['fake', 'real'], fontsize='xx-small')
                    plt.savefig(cfg.sSampleDir + '/vec1_plot.png')
                    plt.savefig('{}/vec1_plot_{:02d}_{:04d}.png'.format(cfg.sSampleDir + '/class_random/', counter // 10000, counter % 10000))
                    plt.close()

                    plt.figure()
                    plt.ylim(0.0, 1.0)
                    plt.plot(dotK(fake_vecKs[-1000:], fake_targets[-1000:]))
                    plt.savefig(cfg.sSampleDir + '/fake_dotK.png')
                    plt.savefig('{}/fake_dotK_{:02d}_{:04d}.png'.format(cfg.sSampleDir + '/class_random/', counter // 10000, counter % 10000))
                    plt.close()

                    plt.figure()
                    plt.ylim(0.0, 1.0)
                    plt.plot(dotK(real_vecKs[-1000:], real_targets[-1000:]))
                    plt.savefig(cfg.sSampleDir + '/real_dotK.png')
                    plt.savefig('{}/real_dotK_{:02d}_{:04d}.png'.format(cfg.sSampleDir + '/class_random/', counter // 10000, counter % 10000))
                    plt.close()

                    plt.figure()
                    plt.ylim(0.0, 1.0)
                    plt.plot(np.mean(mode_vecs, 0))
                    plt.savefig(cfg.sSampleDir + '/mode_vec.png')
                    plt.savefig('{}/mode_vec_{:02d}_{:04d}.png'.format(cfg.sSampleDir + '/class_random/', counter // 10000, counter % 10000))
                    plt.close()

            ############################################################################################################################################

            log_time = time.time()

            if counter - load_counter == 1:
                time1 = time.time()

            if counter - load_counter == 2:
                start_time = time1 - (time.time() - time1)

            if np.mod(counter, 10) == 0:

                the_am_score = am_score_col[-1][0] if len(am_score_col) > 0 else 0
                the_am_score_per = am_score_col[-1][1] if len(am_score_col) > 0 else 0
                the_am_icp_score = am_score_col[-1][3] if len(am_score_col) > 0 else 0
                the_am_icp_score_per = am_score_col[-1][4] if len(am_score_col) > 0 else 0
                the_icp_score = icp_score_col[-1][4] if len(icp_score_col) > 0 else 0
                the_icp_score_per = icp_score_col[-1][5] if len(icp_score_col) > 0 else 0
                the_ssim_sample = msssim_scores_col[-1][0][0] if len(msssim_scores_col) > 0 else 0
                the_ssim_ratio = msssim_scores_col[-1][1] if len(msssim_scores_col) > 0 else 0

                print("[%d, %4d] icp_sc:% .2f (%.2f) am_sc: %.2f (%.2f) am_icp_sc: %.2f (%.2f) ssim: %.2f (%.2f) klr: %.3f %.3f ndev: %.3f "
                  "f_vec1: %.3f %.3f f_maxK: %.3f %.3f f_dotK: %.3f %.3f "
                  "r_vec1: %.3f %.3f r_maxK: %.3f %.3f r_dotK: %.3f %.3f "
                  "r_accK: %.3f t_accK: %.3f "
                  "f_entp: %.3f %.3f [%.3f] r_entp: %.3f %.3f "
                  "#%s time: %.3fs [%.3fm] %.3fs %.1fm %.1fh d_loss:% .3f g_loss:% .3f g_GN: %.3f d_decay: %.3f g_decay: %.3f [d_loss_gan: %.3f d_loss_ac: %.3f]"%
                  (counter // 10000, counter % 10000, the_icp_score, the_icp_score_per, the_am_score, the_am_score_per, the_am_icp_score, the_am_icp_score_per, the_ssim_sample, the_ssim_ratio,
                   fLrG * 1000, fLrD * 1000, fInputNoise,
                   mean(fake_vec1s), std(fake_vec1s), mean(np.max(fake_vecKs, 1)), std(np.max(fake_vecKs, 1)), mean(dotK(fake_vecKs, fake_targets)), std(dotK(fake_vecKs, fake_targets)),
                   mean(real_vec1s), std(real_vec1s), mean(np.max(real_vecKs, 1)), std(np.max(real_vecKs, 1)), mean(dotK(real_vecKs, real_targets)), std(dotK(real_vecKs, real_targets)),
                   mean(np.argmax(real_vecKs,1)==np.argmax(real_targets,1)), test_accK,
                   mean(get_cross_entropy(fake_vecKs)),
                   get_cross_entropy(np.mean(fake_vecKs, 0)),
                   get_cross_entropy(np.mean(mode_vecs, 0)),
                   mean(get_cross_entropy(real_vecKs)),
                   get_cross_entropy(np.mean(real_vecKs, 0)),
                   cfg.sTestName,
                   log_time - end_time, score_time / 60, end_time - start_time, (end_time - start_time) * 10000 / 60, (end_time - start_time) * (cfg.iBatchRun - counter) / 3600,
                   mean(d_losses), mean(g_losses), mean(g_GNs), mean(d_decaylosses), mean(g_decaylosses), d_loss_gan, d_loss_ac))

                if cfg.iTrainD > 0:
                    problog.write("[%d, %4d] r_vecK: %s, r_vec1: %.3f f_vecK: %s f_vec1: %.3f mode_vec: %s mode_entropy: %.3f\n" % (
                        counter // 10000, counter % 10000,
                        np.array2string(real_vecKs[0,:], formatter={'float_kind': lambda x: "%.2f" % x}), mean(real_vec1s),
                        np.array2string(fake_vecKs[0,:], formatter={'float_kind': lambda x: "%.2f" % x}), mean(fake_vec1s),
                        np.array2string(np.mean(mode_vecs, 0), formatter={'float_kind': lambda x: "%.2f" % x}), get_cross_entropy(np.mean(mode_vecs, 0))))

            ################################################################# Ne2rk ##############################################################################

    @property
    def am_score(self):
        if self._am_score is None:
            self._am_score = AMScore(self.cfg)
        return self._am_score

    @property
    def icp_score(self):
        if self._icp_score is None:
            self._icp_score = InceptionScore()
        return self._icp_score

    @property
    def MSSSIM(self):
        if self._msssim is None:
            self._msssim = MSSSIM()
        return self._msssim

    def __init__(self, sess, cfg=None):

        self.sess = sess
        self.cfg = cfg
        self.data = DataProvider(cfg)
        self.num_classes = self.data.num_classes

        self._am_score = None
        self._icp_score = None
        self._msssim = None

        if cfg.bUseClassLabel and (cfg.sDataSet == 'cifar10' or cfg.sDataSet == 'tiny'):
            self.MSSSIM.set_ref_images(np.concatenate([self.data.load_data_classes(i, cfg.iSSIM) for i in range(self.num_classes)], 0), self.num_classes, cfg.iSSIM)

        #self.MSSSIM.set_ref_scores([0.3131934106349945, 0.32166755199432373, 0.2892302870750427, 0.2745410203933716, 0.29048576951026917, 0.2770773768424988, 0.28816020488739014, 0.28492170572280884, 0.34145477414131165, 0.30046385526657104])

        self.ref_distribution_even = tf.ones([cfg.iBatchSize, self.data.num_classes]) / self.data.num_classes # should change if use not evenly distributed data
        self.ref_avg_distribution_even = tf.ones([self.data.num_classes]) / self.data.num_classes  # should change if use not evenly distributed data

        self.linear = linear
        self.conv2d = conv2d
        self.deconv2d = deconv2d
        self.lrelu = lrelu
        self.tanh = tf.nn.tanh
        self.bn = batch_norm

        if cfg.bAddHZ:
            self.deconv2d_op = self.deconv2d
            self.deconv2d = self.deconv2d_hz

        self.generator = getattr(self, cfg.generator)
        self.discriminator = getattr(self, cfg.discriminator)

        self.build_model(cfg)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        with tf.variable_scope('optimizer/G'):
            self.g_global_step = tf.Variable(0, trainable=False, name='g_global_step')

            print('exp decay fLrG')
            self.fLrG = tf.train.exponential_decay(cfg.fLrIniG, self.g_global_step, cfg.iLrStep, cfg.fLrDecay, cfg.bLrStair)

            if cfg.oOptG == 'adam':
                self.g_opt = tf.train.AdamOptimizer(self.fLrG, beta1=cfg.fBeta1G, beta2=cfg.fBeta2G)
            elif cfg.oOptG == 'rmsprop':
                self.g_opt = tf.train.RMSPropOptimizer(self.fLrG)
            elif cfg.oOptG == 'nadam':
                self.g_opt = NadamOptimizer(self.fLrG, beta1=cfg.fBeta1G, beta2=cfg.fBeta2G)
            else:
                self.g_opt = tf.train.GradientDescentOptimizer(self.fLrG)
                assert cfg.oOptG == 'sgd'

            self.g_gv = self.g_opt.compute_gradients(self.g_loss_total, var_list=self.g_vars)
            self.g_optim = self.g_opt.apply_gradients(self.g_gv, global_step=self.g_global_step)

        with tf.variable_scope('optimizer/D'):
            self.d_global_step = tf.Variable(0, trainable=False, name='d_global_step')

            print('exp decay fLrD')
            self.fLrD = tf.train.exponential_decay(cfg.fLrIniD*2, self.d_global_step, cfg.iLrStep, cfg.fLrDecay, cfg.bLrStair)

            if cfg.oOptD == 'adam':
                self.d_opt = tf.train.AdamOptimizer(self.fLrD, beta1=cfg.fBeta1D, beta2=cfg.fBeta2D)
            elif cfg.oOptD == 'rmsprop':
                self.d_opt = tf.train.RMSPropOptimizer(self.fLrD)
            elif cfg.oOptG == 'nadam':
                self.d_opt = NadamOptimizer(self.fLrD, beta1=cfg.fBeta1D, beta2=cfg.fBeta2D)
            else:
                self.d_opt = tf.train.GradientDescentOptimizer(self.fLrD)
                assert cfg.oOptD == 'sgd'

            self.d_gv = self.d_opt.compute_gradients(self.d_loss_total, var_list=self.d_vars)
            self.d_optim = self.d_opt.apply_gradients(self.d_gv, global_step=self.d_global_step)

        for name, grads_and_vars in [('G', self.g_gv), ('D', self.d_gv)]:

            print("\n\n{} Params:".format(name))

            info = []

            total_param_count = 0
            for g, v in grads_and_vars:
                shape = v.get_shape()
                shape_str = ",".join([str(x) for x in v.get_shape()])

                param_count = 1
                for dim in shape:
                    param_count *= int(dim)
                total_param_count += param_count

                if g == None:
                    info += ["\t{} ({}) [{}] [no grad!]".format(v.name, shape_str, param_count)]
                else:
                    info += ["\t{} ({}) [{}]".format(v.name, shape_str, param_count)]

            for x in sorted(info):
                print(x)

            print("Total param count: {}\n\n".format(locale.format("%d", total_param_count, grouping=True)))

        self.saver = tf.train.Saver(max_to_keep=max_checkpoint_num)
        self.writer = tf.summary.FileWriter(cfg.sTestCaseDir + '/log', self.sess.graph)

        cfg.iSaveCount = min(int(cfg.iSaveCount) // 10 * self.num_classes, 400)
        self.save_size = int(np.sqrt(cfg.iSaveCount))

        cfg.iSaveCount = self.save_size**2
        samplecount = int(np.ceil(float(cfg.iSaveCount) / cfg.iBatchSize)) * cfg.iBatchSize

        self.sample_z = self.data.get_noise_count(cfg, samplecount, ordered100=True)
        self.sample_images = self.data.load_data_count(samplecount)

        if cfg.bUseClassLabel:
            class_images = []
            for i in range(self.num_classes):
                class_images.append(self.data.load_data_classes(i, cfg.iSaveCount // self.num_classes))
            class_images = np.concatenate(class_images)
            self.sample_images[:(cfg.iSaveCount // self.num_classes) * self.num_classes] = class_images

    def initilization_model(self, cfg):

        def initialization_wn():
            var_list = tf.global_variables()
            for var in var_list:
                self.sess.run(tf.variables_initializer([var]), feed_dict={self.z: self.sample_z[:cfg.iBatchSize], self.images_lab: self.sample_images[:cfg.iBatchSize], self.fInputNoise: cfg.fInputNoise})
                print(var.op.name)

        def initialization():
            self.sess.run(tf.global_variables_initializer(), feed_dict={self.z: self.sample_z[:cfg.iBatchSize], self.images_lab: self.sample_images[:cfg.iBatchSize], self.fInputNoise: cfg.fInputNoise})

        def do_init():
            #try:
            #    initialization()
            #except:
            initialization_wn()

        print('optimizor initialization')

        if cfg.bLoadCheckpoint:
            try:
                if self.load(cfg):
                    print(" [*] Load SUCCESS")
                else:
                    print(" [!] Load failed...")
                    do_init()
            except:
                do_init()
        else:
            do_init()

    def build_model(self, cfg):

        ############################################################################################################################################

        def per_entropy_cross_with_logits(logits, ref=None):
            if ref==None:
                ref = self.ref_distribution_even
            assert 'probs' not in logits.name
            per_sample_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ref))
            return per_sample_cross_entropy

        def avg_entropy_cross_with_probs(probs):
            probs = tf.reduce_mean(probs, 0)
            probs = probs / tf.reduce_sum(probs)
            return -tf.reduce_sum(self.ref_avg_distribution_even * tf.log(probs * (1.0 - self.data.num_classes * 1e-8) + 1e-8))

        def per_entropy_H_with_probs(probs):
            return -tf.reduce_sum(probs * tf.log(probs * (1.0 - self.data.num_classes * 1e-8) + 1e-8), 1)

        def avg_entropy_H_with_probs(probs):
            probs = tf.reduce_mean(probs, 0)
            probs = probs / tf.reduce_sum(probs)
            return -tf.reduce_sum(probs * tf.log(probs * (1.0 - self.data.num_classes * 1e-8) + 1e-8))

        def log_sum_exp(x):
            mx = tf.reduce_max(x, axis=1, keep_dims=True)
            return tf.reshape(tf.squeeze(mx) + tf.log(tf.reduce_sum(tf.exp(x - mx), 1)), [cfg.iBatchSize, 1])

        #def get_labels_vector(labels_index, num_classes):
        #    vec_flattened = tf.Variable(tf.zeros([cfg.iBatchSize*num_classes]), trainable=False)
        #    with tf.control_dependencies([vec_flattened.assign(tf.zeros([cfg.iBatchSize*num_classes]))]):
        #        idx_flattened = tf.range(0, cfg.iBatchSize) * num_classes + labels_index
        #        vec_flattened = tf.scatter_nd_add(vec_flattened, tf.reshape(idx_flattened, [cfg.iBatchSize, 1]), tf.ones([cfg.iBatchSize]) * cfg.fkSmoothed)
        #        vec_flattened += (1.0 - cfg.fkSmoothed) / num_classes
        #        labels_vec = tf.reshape(vec_flattened, [cfg.iBatchSize, num_classes])
        #        return labels_vec

        def get_2vec(label_index, f2Smoothed):
            vec = tf.one_hot(label_index, 2, on_value=cfg.f2Smoothed, off_value=1.0 - f2Smoothed)
            return vec

        def get_kvec(label_index, fkSmoothed):
            vec = tf.one_hot(label_index, self.num_classes, on_value=fkSmoothed + (1.0 - fkSmoothed) / self.num_classes, off_value=(1.0 - fkSmoothed) / self.num_classes)
            return vec

        def get_mvec_real(label_index, fkSmoothed, f2Smoothed):
            vec = tf.concat(axis=1, values=[get_kvec(label_index, fkSmoothed) * f2Smoothed, tf.ones([tf.shape(label_index)[0], 1]) * (1.0 - cfg.f2Smoothed)])
            return vec

        def get_mvec_fake(f2Smoothed):
            vec = tf.concat(axis=1, values=[tf.ones([cfg.iBatchSize, self.num_classes]) * (1.0 - f2Smoothed) / self.num_classes, tf.ones([cfg.iBatchSize, 1]) * f2Smoothed])
            return vec

        def get_sharp_kvec(klogits):
            if cfg.sharp_pow == 0:
                print('using max as target')
                max_labels = tf.to_int32(tf.arg_max(klogits, 1))
                return get_kvec(max_labels, cfg.fkSmoothed)
            elif cfg.sharp_pow > 0:
                print('using temperatured softmax as sharp operator: %f' % cfg.sharp_pow)
                return tf.nn.softmax(klogits*cfg.sharp_pow)

        def AM_target_kclass(logit_kclass):
            v = get_sharp_kvec(logit_kclass)
            target_prob = tf.Variable(trainable=False, initial_value=tf.ones_like(v), name='am')
            ass = target_prob.assign(v)
            with tf.control_dependencies([ass]):
                target_prob = tf.identity(target_prob)
            return target_prob

        ############################################################################################################################################

        self.z = tf.placeholder(tf.float32, [cfg.iBatchSize, cfg.iDimsZ + cfg.bPredefined * self.num_classes], name='z')
        self.lab_image_labels = tf.placeholder(tf.int32, [cfg.iBatchSize], name='lab_image_labels')
        self.fInputNoise = tf.placeholder(tf.float32, [], name='fInputNoise')

        self.images_gen = self.generator(self.z, cfg=cfg)
        self.images_lab = tf.placeholder(tf.float32, self.images_gen.get_shape(), name='lab_images')

        logits_kclass_gen, logit_2class_r_gen, logit_2class_f_gen = self.discriminator(self.images_gen, cfg=cfg, reuse=False, y=self.z[:, cfg.iDimsZ:])
        logits_kclass_lab, logit_2class_r_lab, logit_2class_f_lab = self.discriminator(self.images_lab, cfg=cfg, reuse=True, y=get_kvec(self.lab_image_labels, fkSmoothed=cfg.fkSmoothed))

        print('\n\n')

        if not cfg.FAKE_LOGIT:
            logit_2class_f_gen = tf.zeros([cfg.iBatchSize, 1])
            logit_2class_f_lab = tf.zeros([cfg.iBatchSize, 1])

        if cfg.LAB:
            logit_2class_r_gen = log_sum_exp(logits_kclass_gen)
            logit_2class_r_lab = log_sum_exp(logits_kclass_lab)

        logits_2class_gen = tf.concat(axis=1, values=[logit_2class_r_gen, logit_2class_f_gen])
        logits_2class_lab = tf.concat(axis=1, values=[logit_2class_r_lab, logit_2class_f_lab])

        ############################################################################################################################################

        distarget_2class_real = get_2vec(tf.zeros([cfg.iBatchSize], dtype=tf.int32), f2Smoothed=cfg.f2Smoothed)
        distarget_2class_fake = get_2vec(tf.ones([cfg.iBatchSize], dtype=tf.int32), f2Smoothed=cfg.f2Smoothed)

        if cfg.PATH:

            assert self.num_classes == 1 and cfg.f2Smoothed == 1.0
            realp = tf.sqrt(tf.random_uniform([cfg.iBatchSize, 1], minval=0.0, maxval=1.0))
            distarget_2class_real = tf.concat(axis=1, values=[realp, tf.ones([cfg.iBatchSize, 1]) - realp])

            fakep = tf.sqrt(tf.random_uniform([cfg.iBatchSize, 1], minval=0.0, maxval=1.0))
            distarget_2class_fake = tf.concat(axis=1, values=[tf.ones([cfg.iBatchSize, 1]) - fakep, fakep])

        distarget_kclass_real = get_kvec(self.lab_image_labels, fkSmoothed=cfg.fkSmoothed)
        distarget_kclass_fake = self.ref_distribution_even

        if cfg.bPredefined:
            assert cfg.sharp_pow == 0.0  # argmax
            pseudo_logit = tf.slice(self.z, [0, cfg.iDimsZ], [cfg.iBatchSize, self.num_classes])
            gentarget_kclass = AM_target_kclass(pseudo_logit)
        else:
            gentarget_kclass = AM_target_kclass(logits_kclass_gen)

        self.mode_vec = tf.reduce_mean(tf.nn.softmax(logits_kclass_gen * 10000), 0) # arg_max have no gradient. use this a an alternative
        self.mode_entropy = -tf.reduce_sum(self.ref_avg_distribution_even * tf.log(self.mode_vec * (1.0 - self.data.num_classes * 1e-8) + 1e-8))

        self.fake_target = gentarget_kclass
        self.real_target = distarget_kclass_real

        ############################################################################################################################################

        def dloss_WGAN():

            assert not cfg.LAB
            return tf.reduce_mean(logit_2class_r_gen) - tf.reduce_mean(logit_2class_r_lab)

        def dloss_GAN():

            self.d_loss_lab = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2class_lab, labels=distarget_2class_real))
            self.d_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2class_gen, labels=distarget_2class_fake))

            return self.d_loss_lab + self.d_loss_gen

        def dloss_AC():

            if cfg.bUseClassLabel:
                self.d_loss_lab_ac = tf.reduce_mean(cfg.f2Smoothed * tf.nn.softmax_cross_entropy_with_logits(logits=logits_kclass_lab, labels=distarget_kclass_real))
            else:
                self.d_loss_lab_ac = tf.reduce_mean(per_entropy_H_with_probs(tf.nn.softmax(logits_kclass_lab)))
                self.d_loss_lab_ac -= per_entropy_H_with_probs(tf.reduce_mean(tf.nn.softmax(logits_kclass_lab), 0, keep_dims=True))

            if cfg.f2Smoothed < 1.0:
                self.d_loss_gen_ac = tf.reduce_mean((1.0 - cfg.f2Smoothed) * tf.nn.softmax_cross_entropy_with_logits(logits=logits_kclass_gen, labels=distarget_kclass_fake))
            else:
                self.d_loss_gen_ac = 0

            if cfg.DIST_AC_GEN_EVEN:
                print('added d_loss_gen_ac')
                self.d_loss_gen_ac = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_kclass_gen, labels=distarget_kclass_fake))

            if cfg.DIST_AC_GEN_ACGAN:
                self.d_loss_gen_ac = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_kclass_gen, labels=gentarget_kclass))

            return self.d_loss_lab_ac + self.d_loss_gen_ac

        def dloss_GP():

            alpha = tf.random_uniform(shape=[cfg.iBatchSize, 1, 1, 1], minval=0., maxval=1.)
            differences = self.images_gen - self.images_lab
            interpolates = self.images_lab + alpha * differences
            gradients = tf.gradients(self.discriminator(interpolates, cfg=cfg, reuse=True)[1], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))  # here, reduction_indices=[1] means reduce over all HWC pixels
            gradient_penalty = 10 * tf.reduce_mean((slopes - 1.) ** 2)

            return gradient_penalty

        def dloss_DECAY():
            d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
            return tf.add_n([tf.nn.l2_loss(tf.maximum(tf.abs(var), 1) - 1) for var in d_vars])

        def gloss_WGAN():
            return -tf.reduce_mean(logit_2class_r_gen)

        def gloss_GAN():
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2class_gen, labels=distarget_2class_real))

        def gloss_AC():
            return tf.reduce_mean(cfg.f2Smoothed * tf.nn.softmax_cross_entropy_with_logits(logits=logits_kclass_gen, labels=gentarget_kclass))

        def gloss_MODE():
            return self.mode_entropy

        def gloss_DECAY():
            g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
            return tf.add_n([tf.nn.l2_loss(tf.maximum(tf.abs(var), 1) - 1) for var in g_vars])

        def gloss_GN():
            gradients = tf.slice(tf.gradients(self.images_gen, self.z)[0], [0, 0], [cfg.iBatchSize, cfg.iDimsZ])
            return tf.reduce_mean(tf.abs(gradients))

        ############################################################################################################################################

        self.d_decayloss = dloss_DECAY()
        self.d_loss_total = self.d_decayloss * cfg.DECAY_WEIGHT

        if cfg.WGAN:
            self.d_loss_gan = dloss_WGAN()
            self.d_loss_total += self.d_loss_gan
        else:
            self.d_loss_gan = dloss_GAN()
            self.d_loss_total += self.d_loss_gan

        if cfg.WGAN or cfg.GP:
            self.d_loss_total += dloss_GP()

        self.d_loss_ac = dloss_AC()
        if cfg.AC or cfg.LAB or cfg.D_AC_WEIGHT:
            if cfg.D_AC_WEIGHT:
                self.d_loss_total += self.d_loss_ac * cfg.D_AC_WEIGHT
            else:
                self.d_loss_total += self.d_loss_ac

        self.g_decayloss = gloss_DECAY()
        self.g_loss_total = self.g_decayloss * cfg.DECAY_WEIGHT

        if cfg.WGAN:
            self.g_loss_total += gloss_WGAN()
        else:
            self.g_loss_total += gloss_GAN()

        if cfg.AC:
            self.g_loss_total += gloss_AC() * cfg.G_AC_WEIGHT

        if cfg.MODE:
            self.g_loss_total += gloss_MODE()

        self.g_GN = gloss_GN()
        if cfg.GN:
            self.g_loss_total += -self.g_GN * cfg.GN_WEIGHT

        ############################################################################################################################################

        if cfg.WGAN:

            self.real_vec1 = logit_2class_r_lab
            self.fake_vec1 = logit_2class_r_gen

        else:

            self.real_vec1 = tf.nn.softmax(logits_2class_lab)[:, 0:1]
            self.fake_vec1 = tf.nn.softmax(logits_2class_gen)[:, 0:1]

        #if cfg.LAB or cfg.AC:

        self.real_vecK = tf.nn.softmax(logits_kclass_lab)
        self.fake_vecK = tf.nn.softmax(logits_kclass_gen)

    def channel_concat(self, x, y):

        x_shapes = x.get_shape().as_list()
        y_shapes = y.get_shape().as_list()
        assert y_shapes[0] == x_shapes[0]

        y = tf.reshape(y, [y_shapes[0], 1, 1, y_shapes[1]]) * tf.ones([y_shapes[0], x_shapes[1], x_shapes[2], y_shapes[1]])

        return tf.concat([x, y], 3)

    def generator(self, z, cfg):

        with tf.variable_scope('generator'):

            cts = {}
            ldc = []

            curfdim = cfg.iFilterDimG
            cursize = cfg.iImageSize // 2
            while cursize % 2 == 0 and cursize >= 8:
                cursize //= 2
                curfdim *= cfg.fDimIncreaseRate

            dims = [curfdim]
            curres = cfg.iResG
            while curres > 0:
                curres -= 1
                curfdim = max(cfg.iFilterDimD, curfdim // cfg.fDimIncreaseRate)
                dims = [curfdim] + dims

            #if cfg.bPredefined:
                #cat = self.linear(z[:, cfg.iDimsZ:], 100, cts=cts, ldc=ldc)
                #z= z[:, :cfg.iDimsZ] + cat

            h0 = z
            h0 = self.linear(h0, int(curfdim) * cursize * cursize, cts=cts, ldc=ldc)
            h0 = tf.reshape(h0, [-1, cursize, cursize, int(curfdim)])
            h0 = self.activate(h0, cfg.oActG, cfg.oBnG, cts, ldc)

            for i in range(cfg.iResG):
                h0 = self.deconv_maybe_upsize(h0, int(dims[i+1]), ksize=cfg.iKsizeG, stride=1, oUp=cfg.oUp, cts=cts, ldc=ldc)
                h0 = self.activate(h0, cfg.oActG, cfg.oBnG, cts, ldc)

            curfdim = dims[-1]

            while cfg.iImageSize // 2 != cursize:
                cursize *= 2
                curfdim /= cfg.fDimIncreaseRate
                h0 = self.deconv_maybe_upsize(h0, int(curfdim), ksize=cfg.iKsizeG, stride=2, oUp=cfg.oUp, cts=cts, ldc=ldc)
                h0 = self.activate(input=h0, oAct=cfg.oActG, oBn=cfg.oBnG, cts=cts, ldc=ldc)

            if cfg.gfirst1:
                h0 = self.deconv_maybe_upsize(h0, int(curfdim // 2), ksize=cfg.iKsizeG, stride=2, oUp=cfg.oUp, cts=cts, ldc=ldc)
                h0 = self.activate(input=h0, oAct=cfg.oActG, oBn=cfg.oBnG, cts=cts, ldc=ldc)
                h0 = self.deconv_maybe_upsize(h0, int(cfg.iDimsC), ksize=cfg.iKsizeG, stride=1, oUp=cfg.oUp, cts=cts, ldc=ldc)
                out = self.activate(input=h0, oAct='tanh', oBn='none', cts=cts, ldc=ldc)
            else:
                h0 = self.deconv_maybe_upsize(h0, int(cfg.iDimsC), ksize=cfg.iKsizeG, stride=2, oUp=cfg.oUp, cts=cts, ldc=ldc)
                out = self.activate(input=h0, oAct='tanh', oBn='none', cts=cts, ldc=ldc)

            print('\ngenerator:')
            print("\n".join(layer for layer in ldc))
            return out

    def discriminator(self, image, cfg, reuse, y=None):

        with tf.variable_scope('discriminator', reuse=reuse):

            cts = {}
            ldc = []

            h0 = image

            if cfg.fInputNoise > 0:
                h0 = noise(h0, self.fInputNoise, bAdd=True, bMulti=False, cts=cts, ldc=ldc)

            if cfg.CGAN:
                assert cfg.bPredefined
                h0 = self.channel_concat(h0, y)

            for i in range(cfg.dfirst1):
                curfdim = cfg.iFilterDimD / (cfg.fDimIncreaseRate**(cfg.dfirst1-i))
                h0 = self.conv_maybe_downsize(h0, int(curfdim), ksize=cfg.iKsizeD, stride=1, oDown=cfg.oDown, cts=cts, ldc=ldc)
                h0 = self.activate(h0, oAct=cfg.oActD, oBn=cfg.oBnD, cts=cts, ldc=ldc)
                h0 = self.noise(h0, fNoise=cfg.fLayerNoiseD, fDrop=cfg.fLayerDropoutD, cts=cts, ldc=ldc)

            curfdim = cfg.iFilterDimD / 2
            while h0.get_shape().as_list()[1] / 2 >= cfg.iMinSizeD:
                curfdim *= cfg.fDimIncreaseRate
                h0 = self.conv_maybe_downsize(h0, int(curfdim), ksize=cfg.iKsizeD, stride=2, oDown=cfg.oDown, cts=cts, ldc=ldc)
                h0 = self.activate(h0, oAct=cfg.oActD, oBn=cfg.oBnD, cts=cts, ldc=ldc)
                h0 = self.noise(h0, fNoise=cfg.fLayerNoiseD, fDrop=cfg.fLayerDropoutD, cts=cts, ldc=ldc)

            for i in range(cfg.iResD):
                h0 = self.conv_maybe_downsize(h0, int(curfdim), ksize=cfg.iKsizeD, stride=1, oDown=cfg.oDown, cts=cts, ldc=ldc)
                h0 = self.activate(h0, oAct=cfg.oActD, oBn=cfg.oBnD, cts=cts, ldc=ldc)
                h0 = self.noise(h0, fNoise=cfg.fLayerNoiseD, fDrop=cfg.fLayerDropoutD, cts=cts, ldc=ldc)

            if cfg.dfinal == 'avgpool':
                h0 = avgpool(h0, ksize=h0.get_shape().as_list()[1], stride=h0.get_shape().as_list()[1], cts=cts, ldc=ldc)
                h0 = self.noise(h0, fNoise=cfg.fLayerNoiseD, fDrop=cfg.fLayerDropoutD, cts=cts, ldc=ldc)
            elif cfg.dfinal == 'sqrpool':
                h0 = h0 * h0
                h0 = avgpool(h0, ksize=h0.get_shape().as_list()[1], stride=h0.get_shape().as_list()[1], cts=cts, ldc=ldc)
                h0 = self.noise(h0, fNoise=cfg.fLayerNoiseD, fDrop=cfg.fLayerDropoutD, cts=cts, ldc=ldc)

            h0 = tf.reshape(h0, [cfg.iBatchSize, -1])

            class_logit = self.linear(h0, self.num_classes + 2, cts=cts, ldc=ldc)

            if not reuse: print('\ndiscriminator:')
            if not reuse: print("\n".join(layer for layer in ldc))

            return class_logit[:, :self.num_classes], class_logit[:, -2:-1], class_logit[:, -1:]

    def noise(self, input, fNoise, fDrop, cts, ldc):

        if fNoise > 0:
            input = noise(input=input, stddev=fNoise, cts=cts, ldc=ldc)

        if fDrop > 0:
            input = dropout(input=input, drop_prob=fDrop, cts=cts, ldc=ldc)

        return input

    def deconv_maybe_upsize(self, input, targetdim, ksize, stride, oUp, cts, ldc):

        name = get_name('deconv_maybe_downsize', cts)
        with tf.variable_scope(name):

            input_shape = input.get_shape().as_list()

            if stride == 1:
                input = self.deconv2d(input=input, output_dim=targetdim, ksize=ksize, stride=1, cts=cts, ldc=ldc)
            else:
                if oUp == 'deconv':
                    input = self.deconv2d(input=input, output_dim=targetdim, ksize=ksize, stride=2, cts=cts, ldc=ldc)

                if oUp == 'resizen':
                    name = get_name('resizen', cts)
                    input = tf.image.resize_nearest_neighbor(input, [input_shape[1] * 2, input_shape[2] * 2], name=name)
                    input = self.deconv2d(input=input, output_dim=targetdim, ksize=ksize, stride=1, cts=cts, ldc=ldc)

                if oUp == 'resizel':
                    name = get_name('resizel', cts)
                    input = tf.image.resize_bilinear(input, [input_shape[1] * 2, input_shape[2] * 2], name=name)
                    input = self.deconv2d(input, targetdim, ksize=ksize, stride=1, cts=cts, ldc=ldc)

                if oUp == 'depth_space':
                    input = tf.depth_to_space(input, 2)
                    input = self.deconv2d(input, targetdim, ksize=ksize, stride=1, cts=cts, ldc=ldc)

                if oUp == 'phaseshift':
                    input = PhaseShiftResize(input, 2)
                    input = self.deconv2d(input, targetdim, ksize=ksize, stride=1, cts=cts, ldc=ldc)

                if oUp == 'deconvpool':
                    input = self.deconv2d(input, input_shape[-1], ksize=2, stride=2, cts=cts, ldc=ldc)
                    input = self.deconv2d(input, targetdim, ksize=ksize, stride=1, cts=cts, ldc=ldc)

        return input

    def conv_maybe_downsize(self, input, targetdim, ksize, stride, oDown, cts, ldc):

        name = get_name('conv_maybe_downsize', cts)
        with tf.variable_scope(name):

            input_shape = input.get_shape().as_list()

            if input_shape[1] < 8 or stride == 1:
                input = self.conv2d(input, targetdim, ksize=ksize, stride=1, cts=cts, ldc=ldc)
            else:
                if oDown == 'conv':
                    input = self.conv2d(input, targetdim, ksize=ksize, stride=2, cts=cts, ldc=ldc)

                if oDown == 'space_depth':
                    input = tf.space_to_depth(input, 2)
                    input = self.conv2d(input, targetdim, ksize=ksize, stride=1, cts=cts, ldc=ldc)

                if oDown == 'resizen':
                    name = get_name('resizen', cts)
                    input = self.conv2d(input, targetdim, ksize=ksize, stride=1, cts=cts, ldc=ldc)
                    input = tf.image.resize_nearest_neighbor(input, [input_shape[1] // 2, input_shape[2] // 2], name=name)

                if oDown == 'avgpool':
                    name = get_name('avgpool', cts)
                    input = self.conv2d(input, targetdim, ksize=ksize, stride=1, cts=cts, ldc=ldc)
                    input = tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

                if oDown == 'maxpool':
                    name = get_name('maxpool', cts)
                    input = self.conv2d(input, targetdim, ksize=ksize, stride=1, cts=cts, ldc=ldc)
                    input = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

        return input

    def deconv2d_residual(self, input, ksize, oAct, oBn, cts, ldc):

        name = get_name('deconv_residual', cts)
        with tf.variable_scope(name):

            act1 = self.activate(input, oAct, oBn=oBn, cts=cts, ldc=ldc)
            deconv1 = self.deconv2d(act1, input.get_shape().as_list()[-1], ksize=ksize, stride=1, cts=cts, ldc=ldc)

            output = input + deconv1

        ldc.append(str(output.get_shape().as_list()) + ' ' + name + ' ksize:' + str(ksize))
        return output

    def conv2d_residual(self, input, ksize, oAct, oBn, cts, ldc):

        name = get_name('conv_residual', cts)
        with tf.variable_scope(name):

            act1 = self.activate(input, oAct, oBn=oBn, cts=cts, ldc=ldc)
            conv1 = self.conv2d(act1, input.get_shape().as_list()[3] // 4, ksize=1, stride=1, cts=cts, ldc=ldc)

            act2 = self.activate(conv1, oAct, oBn=oBn, cts=cts, ldc=ldc)
            conv2 = self.conv2d(act2, input.get_shape().as_list()[3] // 4, ksize=ksize, stride=1, cts=cts, ldc=ldc)

            act3 = self.activate(conv2, oAct, oBn=oBn, cts=cts, ldc=ldc)
            conv3 = self.conv2d(act3, input.get_shape().as_list()[3], ksize=1, stride=1, cts=cts, ldc=ldc)

            output = input + conv3

        ldc.append(str(output.get_shape().as_list()) + ' ' + name + ' ksize:' + str(ksize) + '')
        return output

    def deconv2d_hz(self, input, output_dim, ksize, stride, cts, ldc, stddev=0.05, padding='SAME'):

        if stride > 1:
            name = get_name('hz', cts)
            with tf.variable_scope(name):
                input_shape = input.get_shape().as_list() #input_shape[3] // 4
                hz = tf.random_uniform([input_shape[0], input_shape[1], input_shape[2], 1], minval=-1.0, maxval=1.0, name=name)
                input = tf.concat(axis=3, values=[input, hz])
            ldc.append(name)

        return self.deconv2d_op(input=input, output_dim=output_dim, ksize=ksize, stride=stride, cts=cts, ldc=ldc, stddev=stddev, padding=padding)

    def activate(self, input, oAct, oBn, cts, ldc):

        if oBn == 'bn':
            input = self.bn(input, cts, ldc)
        elif oBn == 'ln':
            name = get_name('ln', cts)
            input = layer_norm(input, scope=name)
            ldc.append(name)
        else:
            assert oBn == 'none'

        name = get_name(oAct, cts)

        with tf.variable_scope(name):

            if oAct == 'elu':
                input = tf.nn.elu(input)

            elif oAct == 'relu':
                input = tf.nn.relu(input)

            elif oAct == 'lrelu':
                input = lrelu(input)

            elif oAct == 'softmax':
                input = tf.nn.softmax(input)

            elif oAct == 'tanh':
                input = tf.nn.tanh(input)

            elif oAct == 'crelu':
                input = tf.nn.crelu(input)

            else:
                assert oAct == 'none'

        ldc.append(oAct)
        return input

    def save(self, cfg=None, step=None):
        model_name = "model"
        makedirs(cfg.checkpoint_dir)
        self.saver.save(self.sess, os.path.join(cfg.checkpoint_dir, model_name), global_step=step)

    def load(self, cfg=None):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(cfg.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False