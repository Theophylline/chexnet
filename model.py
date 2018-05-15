import tensorflow as tf
import glob
from densenet_layers import *

tf.logging.set_verbosity(tf.logging.INFO)

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
    
    predictions = {
            'prob': tf.nn.sigmoid(logits)
            'labels': tf.round(tf.nn.sigmoid(logits), name='labels')
            'accuracy': tf.metrics.accuracy(labels, predictions['labels'], name='accuracy')
    }
    # predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(loss=cost, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=cost, train_op=train_op)
    
    # evaluate
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
                'labels': predictions['labels']
                'accuracy': predictions['accuracy']
        }
        return tf.estimator.EstimatorSpec(mode, loss=cost, eval_metric_ops=metrics)
    
def input_func(tfrecords=fpaths):
    
    def parser(example):
        features = tf.parse_single_example(
                example,
                features={
                        'image': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.int64),
                })
            
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)
        label = tf.cast(features['label'], tf.int32)
        
        # image augmentation
        
        return image, label
    
    ds = tf.data.TFRecordDataset(tfrecords)
    ds = ds.map(parser) # parsing TFrecords; performance improvements?
    ds = ds.shuffle(1024)
    ds = ds.repeat()    
    ds = ds.batch(64)
    iterator = ds.make_one_shot_iterator()
    batch_img, batch_labels = iterator.get_next()
    
    return batch_img, batch_labels

def eval_func():
    return None

def main():
    chexnet = tf.estimator.Estimator(model_fn=DenseNet, model_fir='tmp/chexnet')
    
    log = { 
        "Accuracy" : 'accuracy'
    }
    
    logging_hook = tf.train.LoggingTensorHook(tensors=log, every_n_iter=100)
    
    chexnet.train(input_fn=input_func, hooks=[logging_hook], steps=40000)
    results = chexnet.evaluate(input_fn=eval_func)
    print(results) # dict containing predicted labels and accuracy
    
#%%