import tensorflow as tf
import glob
from densenet_layers import *

TFRecords_dir = ""
fpaths = glob.glob(TFRecords_dir + "*.tfrecords")
db_121 = [6, 12, 24, 16]
db_169 = [6, 12, 32, 32]
db_201 = [6, 12, 48, 32]

#%%

# read from TFRecords
ds = tf.data.TFRecordDataset(fpaths)

#%%
# model

def DenseNet(l, labels, mode, depth = 121, k = 12):
    
    if depth == 121:
        N = db_121
    else if depth == 169:
        N = db_169
    else:
        N = db_201
    
    #before entering the first dense block, a conv operation with 16 output channels
    #is performed on the input images
    l = conv(name = 'conv', l, filter_size = 7, stride = 2, out_chn = 16)
    l = pooling(name = 'pooling', l, filter_size = 3, p = 'max')
    
    # each block is defined as a dense block + transition layer
    with tf.variable_scope('block1'):
        for i in range(N[0]):
            l = conv('bottleneck_layer.{}'.format(i), l, 1, out_chn=4*k)
            l = add_layer('dense_layer.{}'.format(i), l)
        l = transition_layer('transition1', l)
    
    with tf.variable_scope('block2'):
        for i in range(N[1]):
            l = conv('bottleneck_layer.{}'.format(i), l, 1, out_chn=4*k)
            l = add_layer('dense_layer.{}'.format(i), l)
        l = transition_layer('transition2', l)
    
    with tf.variable_scope('block3'):
        for i in range(N[2]):
            l = conv('bottleneck_layer.{}'.format(i), l, 1, out_chn=4*k)
            l = add_layer('dense_layer.{}'.format(i), l)
        l = transition_layer('transition3', l)
    
    # the last block does not have a transition layer
    with tf.variable_scope('block4'):
        for i in range(N[3]):
            l = conv('bottleneck_layer.{}'.format(i), l, 1, out_chn=4*k)
            l = add_layer('dense_layer.{}'.format(i), l)
    
    # classification (global max pooling and softmax)
    with tf.name_scope('classification'):
        l = batch_norm(l)
        l = tf.nn.relu(l, name='relu')
        l = pooling('pool', l, filter_size = 7)
        l = tf.layers.dense(l, units = 1000, activation = tf.nn.relu, name='fc1')
        logits = tf.layers.dense(l, units = 14, activation = tf.nn.relu, name='fc2') # [batch_size, 10]
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits, name='cost_fn') # cost function
    
    # predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
                'prob': tf.nn.sigmoid(logits)
                'labels': tf.round(tf.nn.sigmoid(logits))
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(cost)
        return tf.estimator.EstimatorSpec(mode, loss=cost, train_op=train_op)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
                'accuracy': tf.metrics.accuracy(labels, predictions['labels']
        }
        return tf.estimator.EstimatorSpec(mode, loss=cost, eval_metric_ops=metrics)
    
    return prob
#%%