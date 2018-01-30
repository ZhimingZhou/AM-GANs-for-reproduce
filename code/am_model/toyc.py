import collections
import os

import numpy as np
import tensorflow as tf

from data_provider.data_provider import DataProvider

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("fLrIni", 0.0001, "")
tf.app.flags.DEFINE_float("fLrDecay", 1.0, "")
tf.app.flags.DEFINE_integer("iLrStep", 200000, "")

tf.app.flags.DEFINE_integer("iBatchSize", 128, "")
tf.app.flags.DEFINE_integer("iBatchRun", 300000, "")
tf.app.flags.DEFINE_integer("iIterCheckpoint", 10000, "")

############################################################################################################################################

tf.app.flags.DEFINE_string("sDataSet", "cifar10", "cifar10, mnist, toy, celebA, toyc")
tf.app.flags.DEFINE_string("sResultTag", "test", "your tag for each test case")

#tf.app.flags.DEFINE_boolean("GP", True, "")
tf.app.flags.DEFINE_boolean("WGAN", True, "")
tf.app.flags.DEFINE_boolean("LAB", False, "")
tf.app.flags.DEFINE_boolean("AC", True, "")
tf.app.flags.DEFINE_string("bPath", False, "")

tf.app.flags.DEFINE_float("sharp_pow", 0.00, "")
tf.app.flags.DEFINE_float("neg_weight", 1.00, "")

tf.app.flags.DEFINE_boolean("bPredefined", False, "")
tf.app.flags.DEFINE_boolean("bUseClassLabel", True, "")

tf.app.flags.DEFINE_float("f2Smoothed", 0.75, "")
tf.app.flags.DEFINE_float("fkSmoothed", 1.00, "")

tf.app.flags.DEFINE_string("generator", 'generator', "generator, generator_vbn, generator_mnist")
tf.app.flags.DEFINE_string("discriminator", 'discriminator', "discriminator, discriminator_mnist")

tf.app.flags.DEFINE_string("sResultDir", "../result/", "where to save the checkpoint and sample")

############################################################################################################################################

tf.app.flags.DEFINE_float("iInputNoisePow", 4, "")
tf.app.flags.DEFINE_float("fInputNoise",    0.10, "")
tf.app.flags.DEFINE_float("fInputNoiseMin", 0.10, "")
tf.app.flags.DEFINE_float("fLayerNoiseD",   0.00, "")
tf.app.flags.DEFINE_float("fLayerDropoutD", 0.05, "")

tf.app.flags.DEFINE_integer("iKsizeG", 5, "3, 4, 5")
tf.app.flags.DEFINE_integer("iKsizeD", 3, "3, 4, 5")
tf.app.flags.DEFINE_integer("iFilterDimG", 128, "")
tf.app.flags.DEFINE_integer("iFilterDimD", 64, "")
tf.app.flags.DEFINE_float("fDimIncreaseRate", 2.0, "")

tf.app.flags.DEFINE_string("oUp", 'resizen', "deconv, resizen, resizel, phaseshift, deconvpool")
tf.app.flags.DEFINE_string("oDown", 'avgpool', "conv, resizen, resizel, avgpool, maxpool, convpool")

tf.app.flags.DEFINE_boolean("bAddHZ", True, "")

tf.app.flags.DEFINE_boolean("bUseUniformZ", False, "")
tf.app.flags.DEFINE_boolean("bNormalizeZ", False, "") # set len(Z) = 1

############################################################################################################################################

tf.app.flags.DEFINE_integer("iTrainG", 1, "")
tf.app.flags.DEFINE_integer("iTrainD", 1, "")

tf.app.flags.DEFINE_float("fBeta1G", 0.5, "")
tf.app.flags.DEFINE_float("fBeta1D", 0.5, "")
tf.app.flags.DEFINE_float("fBeta2G", 0.999, "")
tf.app.flags.DEFINE_float("fBeta2D", 0.999, "")

tf.app.flags.DEFINE_string("oOptG", 'adam', "adam, rmsprop, sgd")
tf.app.flags.DEFINE_string("oOptD", 'adam', "adam, rmsprop, sgd")

tf.app.flags.DEFINE_string("oActG", 'lrelu', "relu, lrelu, elu")
tf.app.flags.DEFINE_string("oActD", 'lrelu', "relu, lrelu, elu")
tf.app.flags.DEFINE_string("oBnG", 'bn', "bn, ln, none")
tf.app.flags.DEFINE_string("oBnD", 'bn', "bn, ln, none")

############################################################################################################################################

tf.app.flags.DEFINE_integer("iDimsC", 3, "")
tf.app.flags.DEFINE_integer("iDimsZ", 100, "")
tf.app.flags.DEFINE_integer("iResG", 0, "")
tf.app.flags.DEFINE_integer("oResG", 0, "")
tf.app.flags.DEFINE_integer("oResD", 0, "")
tf.app.flags.DEFINE_integer("iResD", 0, "")

############################################################################################################################################

tf.app.flags.DEFINE_boolean("bUseNCEforG", False, "") # Negative Cross Entropy: True for log(1-D(x)), False for -log(D(x))
tf.app.flags.DEFINE_boolean("bNegativeSmooth", True, "")

tf.app.flags.DEFINE_boolean("bEntropyS", False, "")
tf.app.flags.DEFINE_boolean("bEntropyN", False, "")
tf.app.flags.DEFINE_boolean("bEntropyH", False, "")

tf.app.flags.DEFINE_float("fEntropyLossG", 0.00, "")
tf.app.flags.DEFINE_float("fEntropyLossD", 0.00, "")

tf.app.flags.DEFINE_boolean("bUseLabel", True, "")
tf.app.flags.DEFINE_boolean("bUseUnlabel", False, "")
tf.app.flags.DEFINE_integer("iNumLabelData", 400, "")

############################################################################################################################################

tf.app.flags.DEFINE_boolean("bLoadCheckpoint", True, "bLoadCheckpoint")
tf.app.flags.DEFINE_string("sEvaluateCheckpoint", "", "")
tf.app.flags.DEFINE_integer("iSamplesEvaluate", 50000, "")

tf.app.flags.DEFINE_boolean("bAugment", True, "")
tf.app.flags.DEFINE_string("sLogfileName", 'log.txt', "log file name")

tf.app.flags.DEFINE_integer("iImageSize", 32, "")
tf.app.flags.DEFINE_boolean("Use32_MNIST", True, "")

tf.app.flags.DEFINE_integer("iSaveBatch", 4, "")

tf.app.flags.DEFINE_boolean("bCropImage", True, "")
tf.app.flags.DEFINE_integer("iCenterCropSize", 108, "")

tf.app.flags.DEFINE_float("fLrPower", 0, "fLrPower == 0 --> exp decay")
tf.app.flags.DEFINE_float("fLrMin", 0.00005, "fLrMin only for fLrPower > 0, not for exp decay")

############################################################################################################################################


class ToyClassifier(object):

    def model_initilization(self, sess, cfg):

        ############################################################################################################################################
        batch_images_lab, batch_labels_lab = self.data.load_label_batch(cfg.iBatchSize, 0)

        def initialization():
            var_list = tf.global_variables()
            for var in var_list:
                sess.run(tf.variables_initializer([var]), feed_dict={self.input: batch_images_lab, self.lab_image_labels: batch_labels_lab})
                print(var.op.name)

        print('optimizor initialization')

        if cfg.bLoadCheckpoint:
            if self.load(sess, cfg):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                initialization()
        else:
            initialization()

    def load(self, sess, cfg=None):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(cfg.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def save(self, sess, cfg=None, step=None):
        model_name = "model"
        if not os.path.exists(cfg.checkpoint_dir):
            os.makedirs(cfg.checkpoint_dir)
        self.saver.save(sess, os.path.join(cfg.checkpoint_dir, model_name), global_step=step)

    def build_model(self, num_classes, cfg):

        self.input = tf.placeholder(tf.float32, [cfg.iBatchSize, 2], name='lab_images')
        self.lab_image_labels = tf.placeholder(tf.int32, [cfg.iBatchSize], name='lab_image_labels')

        with tf.variable_scope('ccccccc'):

            from ops_wn import linear as linear_wn

            cts = {}
            ldc = []

            h0 = self.input
            h0 = tf.nn.relu(linear_wn(h0, 1000, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 1000, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 1000, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 1000, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 1000, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 1000, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 1000, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 1000, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 500, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 500, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 250, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 250, cts=cts, ldc=ldc))
            h0 = tf.nn.relu(linear_wn(h0, 250, cts=cts, ldc=ldc))
            class_logit = linear_wn(h0, num_classes, cts=cts, ldc=ldc, init_scale=0.1)

            self.d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=class_logit, labels=tf.one_hot(self.lab_image_labels, num_classes))
            self.prediction = tf.nn.softmax(class_logit)
            self.top1 = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(self.prediction, self.lab_image_labels, 1)))

        with tf.variable_scope('optimizer/D'):
            self.d_global_step = tf.Variable(0, trainable=False, name='d_global_step')
            self.fLrD = tf.train.exponential_decay(cfg.fLrIni, self.d_global_step, cfg.iLrStep, cfg.fLrDecay)
            self.d_optim = tf.train.AdamOptimizer(self.fLrD, beta1=cfg.fBeta1D, beta2=cfg.fBeta2D).minimize(self.d_loss, global_step=self.d_global_step)
            self.saver = tf.train.Saver(max_to_keep=10000)

    def train(self, sess, cfg):

        self.data = DataProvider(cfg)
        counter = 0
        self.model_initilization(sess, cfg)

        test_top1_col = collections.deque(maxlen=10000)
        while True:
            batch_images_lab, batch_labels_lab = self.data.load_label_batch(cfg.iBatchSize, counter)
            _, training_top1 = sess.run([self.d_optim, self.top1], feed_dict={self.input: batch_images_lab, self.lab_image_labels: batch_labels_lab})
            counter += 1
            #print('training_top1:%f' % training_top1)

            if np.mod(counter, 100) == 0 and cfg.bUseLabel:

                batch_images_lab, batch_labels_lab = self.data.load_test_batch(cfg.iBatchSize, counter // 100)
                test_top1 = sess.run(self.top1, feed_dict={self.input: batch_images_lab, self.lab_image_labels: batch_labels_lab})

                test_top1_col.append(test_top1)
                avg_test_top1 = np.mean(test_top1_col)
                print('iter %d, test_top1:%f' % (counter, float(avg_test_top1)))

            if np.mod(counter, 5000) == 0:
                self.save(sess, cfg, counter)


def main(_):

    per_process_gpu_memory_fraction = 0.45
    gpu_memory_allow_growth = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    if gpu_memory_allow_growth:
        config.gpu_options.allow_growth = True
    elif per_process_gpu_memory_fraction < 1.0:
        config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    sess = tf.Session(config=config)


    toy = ToyClassifier()
    toy.build_model(8, cfg=FLAGS)
    toy.train(sess, cfg=FLAGS)
    exit(0)
