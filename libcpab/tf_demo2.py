# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:45:45 2018

@author: nsde
"""

# %%

import tensorflow as tf
import numpy as np

np.random.seed(13)
from libcpab import Cpab

# %%
if __name__ == '__main__':
    # Load some data
    n = 100  # Number of samples
    k = 2  # Sample dimension
    pi = np.pi

    t1 = np.linspace(0, 2 * pi, n, dtype=np.float32)
    t2 = np.linspace(0, 2 * pi, 2 * n, dtype=np.float32)

    t2_sampled = np.random.choice(t2[1:-1], n - 2, replace=False)
    t2_sampled_sorted = np.sort(t2_sampled)
    t2 = np.concatenate(([t2[0]], t2_sampled_sorted, [t2[-1]]))

    sin_t1 = np.sin(t1)
    cos_t1 = np.cos(t1)
    sin_t2 = np.sin(t2)
    cos_t2 = np.cos(t2)

    # Circles
    X1 = np.vstack((cos_t1, sin_t1))
    X2 = np.vstack((cos_t2, sin_t2))

    data_1 = X1.T[None, ...]
    data_2 = X2.T[None, ...]
    dtype = np.float32
    with tf.device('GPU'):
        data_1 = tf.convert_to_tensor(data_1, dtype=dtype)
        data_2 = tf.convert_to_tensor(data_2, dtype=dtype)

    # Create transformer class
    backend = 'tensorflow'
    device = 'gpu'

    tesselation_size = 2 ** 5
    T1 = Cpab(tess_size=[tesselation_size,], backend=backend, device=device,
         zero_boundary=True, volume_perservation=False, override=False)

    # Now, create tensorflow graph that enables us to estimate the transformation
    # we have just used for transforming the data

    theta = T1.sample_transformation(1)
    # theta_est = tf.Variable(initial_value=1e-3 * tf.ones_like(theta))
    theta_est = tf.Variable(initial_value=1e-3 * tf.zeros_like(theta))
    trainable_variables = [theta_est]
    opt = tf.optimizers.SGD(learning_rate=1e-3)


    with tf.GradientTape(persistent=True) as t:
        for i in range(3000):
            trans_est = T1.transform_data(data_1, theta_est, [n])
            loss = tf.norm(data_2 - trans_est)

            gradients = t.gradient(loss, trainable_variables)
            opt.apply_gradients(zip(gradients, trainable_variables))
            if not i % 100:
                print('Iter: {}, Loss: {}'.format(i, loss.numpy()))