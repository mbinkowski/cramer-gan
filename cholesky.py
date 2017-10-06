import tensorflow as tf
import numpy as np

# Author: Dougal J. Sutherland
def cholesky_unblocked(X, size = None):
    if size is None:
        # need to have an explicit shape for X
        shape = X.get_shape()
        shape.assert_has_rank(2)
        m = shape[0].value
        assert m is not None
        assert shape[1].value == m
    else:
        m = size

    A = tf.matrix_band_part(X, -1, 0)
    cond = lambda j, A: j < m
    def body(j, A):
        s = A[j, :j]
        s2 = tf.matmul(tf.expand_dims(s, 0), tf.expand_dims(s, 1))[0, 0]
        Ajj = tf.sqrt(A[j, j] - s2)

        col = A[j+1:m, j]
        col -= tf.matmul(A[j+1:m, :j], tf.expand_dims(A[j, :j], 1))[:, 0]
        col /= Ajj

        A = tf.concat([
            A[:j],
            tf.concat([
                A[j:, :j],
                tf.expand_dims(tf.concat([tf.expand_dims(Ajj, 0), col],0), 1),
                A[j:, j+1:]
            ],1)],0)
        A.set_shape([m, m])
        return j + 1, A
    return tf.while_loop( cond, body, (tf.constant(0, tf.int32), A))[1]


def _padding(X, n, m):
    """
        Adds n rows of padding to the top of X.
        m equals the number of columns of X.
    """
    padding = tf.zeros((n, m), dtype=X.dtype)
    return tf.concat([padding, X],0)

def _chol(A,j,n,nb):
    if j >= n:
        return A
    elif j + nb > n:
        nb = n % nb

    B11 = tf.slice(A, [j, j], [nb, nb])
    B21 = tf.slice(A, [j + nb, j], [n - j - nb, nb])
    B22 = tf.slice(A, [j + nb, j + nb], [n - j - nb, n - j - nb])

    B11 = cholesky_unblocked(B11, nb)
    B21 = tf.transpose(tf.matrix_triangular_solve(B11, tf.transpose(B21), lower=True))
    B22 = B22 - tf.matmul(B21, tf.transpose(B21))

    B_left = tf.concat([B11, B21], 0)
    B_right = _padding(B22, nb, n - j - nb)
    B = tf.concat([B_left, B_right], 1)

    A_left = tf.slice(A, [0, 0], [n, j])
    A_right = _padding(B, j, n - j)
    A = tf.concat([A_left, A_right],1)

    return _chol(A,j+nb,n,nb)


def cholesky_blocked(A, matrix_order = None, block_size = 200):
    nb = block_size
    if matrix_order is not None:
        n = matrix_order
    else:
        n = A.get_shape()[0].value

    if n is None:
        print("Error. Size of matrix A must be known")

    A = (A+tf.transpose(A))/2.

    return _chol(A, 0, n, nb)


def me_loss(X, Y, d, batch_size, with_inv=True):
    eps = .00001

    Z = tf.transpose(X - Y)
    muZ = tf.expand_dims(tf.reduce_mean(Z, axis=1), 1)
#    c = lambda x: tf.expand_dims(x, 1)
    if not with_inv:
        return (tf.matmul(tf.transpose(muZ), muZ) * batch_size)[0][0]
    Z_ = Z - muZ
    S = tf.matmul(Z_, tf.transpose(Z_))
    S = S / np.float(batch_size - 1) + eps * tf.diag(tf.ones(d))
    
#    L = cholesky_unblocked(S)
    L = tf.cholesky(S)
#    return Z, Z_, S, L
    I = tf.Variable(tf.diag(tf.ones(d)))
    Linv = tf.matrix_triangular_solve(L , I)
    Sinv = tf.matmul(tf.transpose(Linv), Linv)

    lambda_n = tf.matmul(tf.matmul(tf.transpose(muZ), Sinv), muZ) * batch_size
    
    return lambda_n[0][0]
#    
#N = 1000
#import time
#a = tf.get_variable('a', [N, N], tf.float32)
#b = tf.get_variable('b', [N, N], tf.float32)    
#
#c = cholesky_unblocked(a)
#d = tf.cholesky(b)
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    x = np.random.rand(N, N)
#    x = np.dot(x, x.transpose()) + np.eye(N) * .0001
#    t = time.time()
#    C = sess.run(c, feed_dict={a:x})
#    print(time.time() - t)
#    t = time.time()
#    D = sess.run(d, feed_dict={b:x})
#    print(time.time() - t)
    

#a = tf.get_variable('a', [3,50], tf.float32)
#b = tf.get_variable('b', [3,50], tf.float32)
#me = me_loss(a,b)
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    x = np.random.rand(3, 50)
#    y = x + .00001* np.random.rand(3, 50)
#    l = sess.run(me, feed_dict={a: x, b: np.random.permutation(y)})