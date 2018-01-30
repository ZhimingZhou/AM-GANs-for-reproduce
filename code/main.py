from __future__ import print_function
import os, sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#per_process_gpu_memory_fraction = 0.45
gpu_memory_allow_growth = True

from shutil import *
import tensorflow as tf
from utils import pp, makedirs
from print_hook import PrintHook
import numpy as np
import scipy.ndimage

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("iIterCheckpoint", 10000, "")
tf.app.flags.DEFINE_integer("iSamplesEvaluate", 50000, "")
tf.app.flags.DEFINE_boolean("bLoadCheckpoint", False, "bLoadCheckpoint")
tf.app.flags.DEFINE_boolean("bLoadForEvaluation", False, "bLoadForEvaluation")

tf.app.flags.DEFINE_float("fLrIniG", 0.0004, "")
tf.app.flags.DEFINE_float("fLrIniD", 0.0004, "")
tf.app.flags.DEFINE_float("fLrDecay", 0.5, "")
tf.app.flags.DEFINE_integer("iLrStep", 100000, "")
tf.app.flags.DEFINE_boolean("bLrStair", True, "")
tf.app.flags.DEFINE_integer("iBatchRun", 1000000, "")

tf.app.flags.DEFINE_integer("iBatchSize", 100, "")
tf.app.flags.DEFINE_integer("iSSIM", 10, "")

############################################################################################################################################

tf.app.flags.DEFINE_float("fLimitedD", 0.00, "")
tf.app.flags.DEFINE_float("fLimitedG", 1.00, "")

tf.app.flags.DEFINE_string("sDataSet", "cifar10", "cifar10, mnist, toy, celebA, toyc, imagenet")
tf.app.flags.DEFINE_string("sResultTag", "test", "your tag for each test case")

tf.app.flags.DEFINE_boolean("GN", False, "")
tf.app.flags.DEFINE_boolean("GP", False, "")
tf.app.flags.DEFINE_boolean("WGAN", False, "")

tf.app.flags.DEFINE_boolean("CGAN", False, "")
tf.app.flags.DEFINE_boolean("LAB", True, "")
tf.app.flags.DEFINE_boolean("AC", True, "")
tf.app.flags.DEFINE_float("sharp_pow", 0.00, "")

tf.app.flags.DEFINE_float("D_AC_WEIGHT", 0.0, "")
tf.app.flags.DEFINE_float("G_AC_WEIGHT", 1.0, "")
tf.app.flags.DEFINE_boolean("DIST_AC_GEN_EVEN", False, "")
tf.app.flags.DEFINE_boolean("DIST_AC_GEN_ACGAN", False, "")

tf.app.flags.DEFINE_float("DECAY_WEIGHT", 0.0, "")
tf.app.flags.DEFINE_float("GN_WEIGHT", 0.0, "")

tf.app.flags.DEFINE_boolean("FAKE_LOGIT", False, "")
tf.app.flags.DEFINE_string("PATH", False, "")
tf.app.flags.DEFINE_boolean("MODE", False, "")

tf.app.flags.DEFINE_boolean("bPredefined", False, "")
tf.app.flags.DEFINE_boolean("bUseClassLabel", True, "")
tf.app.flags.DEFINE_integer("iUnlableClass", 20, "")

tf.app.flags.DEFINE_float("f2Smoothed", 1.00, "")
tf.app.flags.DEFINE_float("fkSmoothed", 1.00, "")

tf.app.flags.DEFINE_string("generator", 'generator', "generator, generator_vbn, generator_mnist")
tf.app.flags.DEFINE_string("discriminator", 'discriminator', "discriminator, discriminator_mnist")

tf.app.flags.DEFINE_string("sResultDir", "../result/", "where to save the checkpoint and sample")
tf.app.flags.DEFINE_string("sSourceDir", "../code/", "")

############################################################################################################################################

tf.app.flags.DEFINE_float("iInputNoisePow", 2, "")
tf.app.flags.DEFINE_float("fInputNoise",    0.10, "")
tf.app.flags.DEFINE_float("fInputNoiseMin", 0.10, "")
tf.app.flags.DEFINE_float("fLayerNoiseD",   0.00, "")
tf.app.flags.DEFINE_float("fLayerDropoutD", 0.30, "")

tf.app.flags.DEFINE_integer("iKsizeG", 3, "3, 4, 5")
tf.app.flags.DEFINE_integer("iKsizeD", 3, "3, 4, 5")
tf.app.flags.DEFINE_integer("iFilterDimG", 192, "")
tf.app.flags.DEFINE_integer("iFilterDimD", 128, "")
tf.app.flags.DEFINE_float("fDimIncreaseRate", 2.0, "")

tf.app.flags.DEFINE_string("oUp", 'deconv', "deconv, resizen, resizel, phaseshift, deconvpool, depth_space")
tf.app.flags.DEFINE_string("oDown", 'conv', "conv, resizen, resizel, avgpool, maxpool, convpool, space_depth")

tf.app.flags.DEFINE_integer("dfirst1", 1, "")
tf.app.flags.DEFINE_integer("gfirst1", 1, "")
tf.app.flags.DEFINE_string("dfinal", 'avgpool', "avgpool, sqrpool, none")

tf.app.flags.DEFINE_boolean("bAddHZ", False, "")
tf.app.flags.DEFINE_boolean("bUseUniformZ", False, "")
tf.app.flags.DEFINE_boolean("bNormalizeZ", False, "") # set len(Z) = 1

############################################################################################################################################
tf.app.flags.DEFINE_integer("iTrainG", 1, "")
tf.app.flags.DEFINE_integer("iTrainD", 1, "")
tf.app.flags.DEFINE_integer("iWarmD", 0, "")
tf.app.flags.DEFINE_integer("iWarmDIterPer", 10, "")

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
tf.app.flags.DEFINE_integer("iResD", 0, "")

tf.app.flags.DEFINE_integer("iMinSizeD", 4, "")

############################################################################################################################################

tf.app.flags.DEFINE_boolean("bUseLabel", True, "")
tf.app.flags.DEFINE_boolean("bUseUnlabel", False, "")
tf.app.flags.DEFINE_integer("iNumLabelData", 400, "")

############################################################################################################################################

tf.app.flags.DEFINE_string("sEvaluateCheckpoint", "", "")

tf.app.flags.DEFINE_boolean("bAugment", False, "")
tf.app.flags.DEFINE_string("sLogfileName", 'log.txt', "log file name")

tf.app.flags.DEFINE_integer("iImageSize", 32, "")
tf.app.flags.DEFINE_boolean("Use32_MNIST", True, "")

tf.app.flags.DEFINE_integer("iSaveCount", 100, "")

tf.app.flags.DEFINE_boolean("bCropImage", True, "")
tf.app.flags.DEFINE_integer("iCenterCropSize", 108, "")

tf.app.flags.DEFINE_boolean("test", False, "")
tf.app.flags.DEFINE_boolean("debug", False, "")

############################################################################################################################################

def main(_):

    ##############################################

    if FLAGS.sDataSet == 'mnist':
        FLAGS.Use32_MNIST = (FLAGS.generator != 'generator_mnist')
        FLAGS.iImageSize = 32 if FLAGS.Use32_MNIST else 28
        FLAGS.iDimsC = 1

    if not FLAGS.bUseClassLabel:
        FLAGS.bUseLabel = False
        FLAGS.bUseUnlabel = True

    if not FLAGS.bUseLabel:
        assert FLAGS.bUseUnlabel

    if not FLAGS.bUseLabel:
        FLAGS.iNumLabelData = 0

    if not FLAGS.bUseUnlabel:
        FLAGS.iNumLabelData = 10000000000000000

    if FLAGS.CGAN:
        FLAGS.bPredefined = True

    FLAGS.logModel = 'a' # if FLAGS.bLoadCheckpoint else 'w'

    ##############################################

    FLAGS.sTestName = (FLAGS.sResultTag + '_' if len(FLAGS.sResultTag) else "") + FLAGS.sDataSet

    FLAGS.sTestCaseDir = FLAGS.sResultDir + FLAGS.sTestName
    FLAGS.sSampleDir = FLAGS.sTestCaseDir + '/samples'
    FLAGS.checkpoint_dir = FLAGS.sTestCaseDir + '/checkpoint'
    FLAGS.sLogfileName = FLAGS.sTestCaseDir + '/log.txt'

    makedirs(FLAGS.checkpoint_dir)
    makedirs(FLAGS.sTestCaseDir + '/code')

    makedirs(FLAGS.sSampleDir)
    makedirs(FLAGS.sSampleDir + '/class_random')
    makedirs(FLAGS.sSampleDir + '/fixed_noise')

    print(FLAGS.sTestCaseDir)

    tf.logging.set_verbosity(tf.logging.ERROR)

    ##############################################

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    if gpu_memory_allow_growth:
        config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    logfile = open(FLAGS.sLogfileName, FLAGS.logModel)

    def MyHookOut(text):
        if '\r' not in text:
            logfile.write(text)
        return 1, 0, text

    phOut = PrintHook()
    phOut.Start(MyHookOut)

    for arg in ['CUDA_VISIBLE_DEVICES="x" python'] + sys.argv:
        sys.stdout.write(arg + ' ')
    print('\n')

    print(pp.pformat(FLAGS.__flags))

    def copycode(src, dst):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if s.endswith('.py'):
                copy2(s, d)

    copycode(FLAGS.sSourceDir, FLAGS.sTestCaseDir + '/code')

    from model import DCGAN
    dcgan = DCGAN(sess, cfg=FLAGS)
    dcgan.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
