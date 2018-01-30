import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def get_name(layer_name, cts):

    if not layer_name in cts:
        cts[layer_name] = 0

    name = layer_name + '_' + str(cts[layer_name])
    cts[layer_name] += 1

    return name


def batch_norm(inputs, cts, ldc, bOffset=True, bScale=True, epsilon=0.001, reuse=None, decay=0.999, is_training=True):

    name = get_name('bn', cts)
    with tf.variable_scope(name, reuse=reuse):

        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        axis = list(range(len(inputs_shape) - 1))

        offset, scale = None, None
        if bOffset:
            offset = tf.get_variable('offset', shape=params_shape, trainable=True, initializer=tf.zeros_initializer())
        if bScale:
            scale = tf.get_variable('scale', shape=params_shape, trainable=True, initializer=tf.ones_initializer())

        batch_mean, batch_variance = tf.nn.moments(inputs, axis)
        outputs = tf.nn.batch_normalization(inputs, batch_mean, batch_variance, offset, scale, epsilon)

        # Note: here for fast training we did not do the moving average (for testing). which we usually not use.

    ldc.append(name + ' offset:' + str(bOffset) + ' scale:' + str(bScale))
    return outputs


def deconv2d(input, output_dim, cts, ldc, ksize=4, stride=2, stddev=0.05, padding='SAME', bBias=True, init_scale=1.0):

    name = get_name('deconv2d_wn', cts)

    def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
        if isinstance(dim_size, ops.Tensor):
            dim_size = tf.math_ops.mul(dim_size, stride_size)
        elif dim_size is not None:
            dim_size *= stride_size
        if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
        return dim_size

    output_shape = input.get_shape().as_list()
    output_shape[3] = output_dim
    output_shape[2] = get_deconv_dim(output_shape[2], stride, ksize, padding)
    output_shape[1] = get_deconv_dim(output_shape[1], stride, ksize, padding)

    with tf.variable_scope(name):

        def deconv2d_wn_initializer_g(v):
                v_norm = tf.nn.l2_normalize(v, [0, 1, 3])
                deconv = tf.nn.conv2d_transpose(input, v_norm, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)
                m_init, v_init = tf.nn.moments(deconv, [0, 1, 2])
                scale_init = init_scale / tf.sqrt(v_init + 1e-12)
                return scale_init

        def deconv2d_wn_initializer_b(v):
                v_norm = tf.nn.l2_normalize(v, [0, 1, 3])
                deconv = tf.nn.conv2d_transpose(input, v_norm, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)
                m_init, v_init = tf.nn.moments(deconv, [0, 1, 2])
                scale_init = init_scale / tf.sqrt(v_init + 1e-12)
                return -m_init * scale_init

        v = tf.get_variable('v', [ksize, ksize, output_shape[-1], input.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=True)
        g = tf.get_variable('g', dtype=tf.float32, initializer=deconv2d_wn_initializer_g(v.initialized_value()), trainable=True)

        w = tf.reshape(g, [1, 1, output_shape[-1], 1]) * tf.nn.l2_normalize(v, [0, 1, 3])
        x = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

        if bBias:
            b = tf.get_variable('b', dtype=tf.float32, initializer=deconv2d_wn_initializer_b(v.initialized_value()), trainable=True)
            x = tf.nn.bias_add(x, b)  # tf.reshape(tf.nn.bias_add(x, b), x.get_shape())

    ldc.append(str(x.get_shape().as_list()) + ' ' + name + ' ksize:' + str(ksize) + ' stride:' + str(stride) + ' init_scale:' + str(init_scale))
    return x


def conv2d(input, output_dim, cts, ldc, ksize=4, stride=2, stddev=0.05, padding='SAME', bBias=True, init_scale=1.0):

    name = get_name('conv2d_wn', cts)

    with tf.variable_scope(name):

        def conv2d_wn_initializer_g(v):
                v_norm = tf.nn.l2_normalize(v, [0, 1, 2])
                conv = tf.nn.conv2d(input, v_norm, strides=[1, stride, stride, 1], padding=padding)
                m_init, v_init = tf.nn.moments(conv, [0, 1, 2])
                scale_init = init_scale / tf.sqrt(v_init + 1e-12)
                return scale_init

        def conv2d_wn_initializer_b(v):
                v_norm = tf.nn.l2_normalize(v, [0, 1, 2])
                conv = tf.nn.conv2d(input, v_norm, strides=[1, stride, stride, 1], padding=padding)
                m_init, v_init = tf.nn.moments(conv, [0, 1, 2])
                scale_init = init_scale / tf.sqrt(v_init + 1e-12)
                return -m_init * scale_init

        v = tf.get_variable('v', [ksize, ksize, input.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=True)
        g = tf.get_variable('g', dtype=tf.float32, initializer=conv2d_wn_initializer_g(v.initialized_value()), trainable=True)

        w = tf.reshape(g, [1, 1, 1, output_dim]) * tf.nn.l2_normalize(v, [0, 1, 2])
        x = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding=padding)

        if bBias:
            b = tf.get_variable('b', dtype=tf.float32, initializer=conv2d_wn_initializer_b(v.initialized_value()), trainable=True)
            x = tf.nn.bias_add(x, b)  # tf.reshape(tf.nn.bias_add(x, b), x.get_shape())

    ldc.append(str(x.get_shape().as_list()) + ' ' + name + ' ksize:' + str(ksize) + ' stride:' + str(stride) + ' init_scale:' + str(init_scale))
    return x


def linear(input, output_size, cts, ldc, stddev=0.05, bBias=True, init_scale=1.0):

    name = get_name('linear_wn', cts)

    with tf.variable_scope(name):

        def linear_wn_initializer_g(v):
                v_norm = tf.nn.l2_normalize(v, [0])
                x_init = tf.matmul(input, v_norm)
                m_init, v_init = tf.nn.moments(x_init, [0])
                scale_init = init_scale / tf.sqrt(v_init + 1e-12)
                return scale_init

        def linear_wn_initializer_b(v):
                v_norm = tf.nn.l2_normalize(v, [0])
                x_init = tf.matmul(input, v_norm)
                m_init, v_init = tf.nn.moments(x_init, [0])
                scale_init = init_scale / tf.sqrt(v_init + 1e-12)
                return -m_init * scale_init

        if len(input.get_shape()) > 2:
            input = tf.reshape(input, [input.get_shape().as_list()[0], -1])

        v = tf.get_variable('v', [input.get_shape().as_list()[1], output_size], initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=True)
        g = tf.get_variable('g', dtype=tf.float32, initializer=linear_wn_initializer_g(v.initialized_value()), trainable=True)

        x = tf.matmul(input, v)
        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(v), [0]))
        x = tf.reshape(scaler, [1, output_size]) * x

        if bBias:
            b = tf.get_variable('b', dtype=tf.float32, initializer=linear_wn_initializer_b(v.initialized_value()), trainable=True)
            x += tf.reshape(b, [1, output_size])

    ldc.append(str(x.get_shape().as_list()) + ' ' + name + ' init_scale:' + str(init_scale))
    return x


def minibatch_feature(input, cts, ldc, n_kernels=100, dim_per_kernel=5):

    name = get_name('minibatch_feature', cts)
    with tf.variable_scope(name):

        if len(input.get_shape()) > 2:
            input = tf.reshape(input, [input.get_shape().as_list()[0], -1])

        batchsize = input.get_shape().as_list()[0]

        ldc_mini = []
        x = linear(input, n_kernels * dim_per_kernel, cts=cts, ldc=ldc_mini)
        x = tf.reshape(x, [-1, n_kernels, dim_per_kernel])

        mask = np.zeros([batchsize, batchsize])
        mask += np.eye(batchsize)
        mask = np.expand_dims(mask, 1)
        mask = 1. - mask
        rscale = 1.0 / np.sum(mask)

        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(x, 3) - tf.expand_dims(tf.transpose(x, [1, 2, 0]), 0)), 2)
        masked = tf.exp(-abs_dif) * mask
        dist = tf.reduce_sum(masked, 2) * rscale

    ldc.append('minibatch_feature ' + ' dim:' + str(n_kernels) + ' per_dim:' + str(dim_per_kernel) + ' :' + ldc_mini[0])
    return dist


def lrelu(x, leak=0.2, name="lrelu"):

    with tf.variable_scope(name):
        return tf.maximum(x, leak * x)


def lrelu_v2(x, leak=0.2, name="lrelu"):

    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


def avgpool(input, ksize, stride, cts, ldc):

    name = get_name('avgpool', cts)
    with tf.variable_scope(name):
        input = tf.nn.avg_pool(input, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)

    ldc.append(str(input.get_shape().as_list()) + ' ' + name + ' ksize:' + str(ksize) + ' stride:' + str(stride))
    return input


def noise(input, stddev, cts, ldc, bAdd=False, bMulti=True, keep_prob=None):

    name = get_name('noise', cts)

    with tf.variable_scope(name):

        if bAdd:
            input = input + tf.truncated_normal(input.get_shape().as_list(), 0, stddev, name=name)

        if bMulti:
            if keep_prob is not None:
                stddev = tf.sqrt((1-keep_prob) / keep_prob) # get equivalent stddev to dropout of keep_prob
            input = input * tf.truncated_normal(input.get_shape().as_list(), 1, stddev, name=name)

    ldc.append(name + ' ' + str(stddev))
    return input


def dropout(input, drop_prob, cts, ldc):

    name = get_name('dropout', cts)
    with tf.variable_scope(name):
        input = tf.nn.dropout(input, 1.0 - drop_prob, name=name)

    ldc.append(name + ' ' + str(drop_prob))
    return input


def PhaseShiftResize(X, r):

    def _phase_shift(I, r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(axis=1, num_or_size_splits=a, value=X)  # a, [bsize, b, r, r]
        X = tf.concat(axis=2, values=[tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(axis=1, num_or_size_splits=b, value=X)  # b, [bsize, a*r, r]
        X = tf.concat(axis=2, values=[tf.squeeze(x) for x in X])  #
        return tf.reshape(X, (bsize, a * r, b * r, 1))

    assert X.get_shape()[3] % (r * r) == 0
    if r * r != X.get_shape()[3]:
        Xc = tf.split(axis=3, num_or_size_splits=X.get_shape()[3] // r // r, value=X)
        X = tf.concat(axis=3, values=[_phase_shift(x, r) for x in Xc])
    else:
        X = _phase_shift(X, r)

    return X
