# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:06:05 2019

@author: nsde
"""

#%%
import tensorflow as tf


def _real_case2x2(a, b):
    """ Real solution for expm function for 2x2 special form matrices"""
    Ea = tf.expand_dims(tf.expand_dims(tf.exp(a), axis=1), axis=2)
    Eb = tf.expand_dims(tf.expand_dims(b * (tf.exp(a) - 1) / a, axis=1), axis=2)
    return tf.concat([Ea, Eb], axis=2)


# %%
def _limit_case2x2(a, b):
    """ Limit solution for expm function for 2x2 special form matrices"""
    Ea = tf.expand_dims(tf.expand_dims(tf.ones_like(a), axis=1), axis=2)
    Eb = tf.expand_dims(tf.expand_dims(b, axis=1), axis=2)
    return tf.concat([Ea, Eb], axis=2)


# %%
def expm(A):
    """ Tensorflow implementation for finding the matrix exponential of a batch
        of 2x2 matrices that have special form (last row is zero).

    Arguments:
        A: 3D-`Tensor` [N,2,2]. Batch of input matrices. It is assumed
            that the second row of each matrix is zero.

    Output:
        expA: 3D-`Tensor` [N,2,2]. Matrix exponential for each matrix in input tensor A.
    """
    n_batch = tf.shape(A)[0]
    a, b = A[:, 0, 0], A[:, 0, 1]

    real_res = _real_case2x2(a, b)
    limit_res = _limit_case2x2(a, b)
    E = tf.compat.v1.where(tf.equal(a, 0), limit_res, real_res)

    zero = tf.zeros((n_batch, 1, 1))
    ones = tf.ones((n_batch, 1, 1))
    expA = tf.concat([E, tf.concat([zero, ones], axis=2)], axis=1)
    return expA

#%%
# try:
#     expm = tf.linalg.expm
# except:
#     def expm(A):
#         """ """
#         n_A = A.shape[0]
#         A_fro = tf.sqrt(tf.reduce_sum(tf.pow(tf.abs(A), 2.0), axis=[1,2], keepdims=True))
#
#         # Scaling step
#         with tf.device(A.device):
#             maxnorm = tf.cast([5.371920351148152], dtype=A.dtype)
#             zero = tf.cast([0.0], dtype=A.dtype)
#         n_squarings = tf.maximum(zero, tf.math.ceil(log2(A_fro / maxnorm)))
#         Ascaled = A / 2.0**n_squarings
#         n_squarings = tf.cast(tf.reshape(n_squarings, (-1, )), tf.int64)
#
#         # Pade 13 approximation
#         U, V = pade13(Ascaled)
#         P = U + V
#         Q = -U + V
#         R = tf.linalg.solve(Q, P)
#
#         # Unsquaring step
#         n = tf.reduce_max(n_squarings)
#         res = [R]
#         for i in range(n):
#             res.append(tf.matmul(res[-1], res[-1]))
#         R = tf.stack(res)
#         expmA = tf.gather_nd(R, tf.transpose(tf.stack([n_squarings, tf.range(n_A, dtype=tf.int64)])))
#         return expmA

#%%
def log2(x):
    with tf.device(x.device):
        denum = tf.math.log(tf.cast([2.0], dtype=x.dtype))
    return tf.math.log(x) / denum

#%%
def pade13(A):
    with tf.device(A.device):
        b = tf.cast([64764752532480000., 32382376266240000., 7771770303897600.,
                     1187353796428800., 129060195264000., 10559470521600.,
                     670442572800., 33522128640., 1323241920., 40840800.,
                     960960., 16380., 182., 1.], dtype=A.dtype)
        ident = tf.eye(A.shape[1], dtype=A.dtype)
    A2 = tf.matmul(A,A)
    A4 = tf.matmul(A2,A2)
    A6 = tf.matmul(A4,A2)
    U = tf.matmul(A, tf.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = tf.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V

#%%
if __name__ == '__main__':
        A = tf.random.normal((5,3,3), mean=1)
        expmA = expm(A)
