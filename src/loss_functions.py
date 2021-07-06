import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors


def focal_loss(y_true, y_pred, gamma=2, alpha=0.95):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = tf.clip_by_value(pt_1, 1e-3, .999)
    pt_0 = tf.clip_by_value(pt_0, 1e-3, .999)

    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)) - tf.reduce_sum((1-alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))


# For evaluation only
def point2point(x_ori, x_rec):
    # Check if x_rec is empty
    if x_rec.size == 0:
        return np.inf

    # Set x_ori as the reference. Loop over x_rec and find nearest neighbor in x_ori
    nbrsA = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(x_ori)
    distBA, _ = nbrsA.kneighbors(x_rec)
    mseBA = np.square(distBA).mean()
    # Set x_rec as the reference. Loop over x_ori and find nearest neighbor in x_rec
    nbrsB = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(x_rec)
    distAB, _ = nbrsB.kneighbors(x_ori)
    mseAB = np.square(distAB).mean()
    # Symmetric total mse
    mse_sym = np.maximum(mseBA, mseAB)

    return mse_sym
