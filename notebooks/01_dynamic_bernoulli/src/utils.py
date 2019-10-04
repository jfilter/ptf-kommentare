import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

def plot_with_labels(low_dim_embs, labels, fname):
    plt.figure(figsize=(28, 28))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(fname)
    plt.close()
    pass

def make_dir(name):
    dir_name = os.path.join('fits', name, 'EF_EMB_'+time.strftime("%y_%m_%d_%H_%M_%S"))
    while os.path.isdir(dir_name):
        time.sleep(np.random.randint(10))
        dir_name = os.path.join('fits', name, 'EF_EMB_'+time.strftime("%y_%m_%d_%H_%M_%S"))
    os.makedirs(dir_name)
    return dir_name

def variable_summaries(summary_name, var):
    with tf.name_scope(summary_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

def cosine_distance(z1, z2):
    return 1-z1.dot(z2)/np.sqrt(z1.dot(z1) *((z2*z2).sum(0)))
