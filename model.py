import tensorflow as tf
import densenet_layers as layers

tfrecords_train = "E:\project data\chexnet\chexnet_train.tfrecords"
tfrecords_test = "E:\project data\chexnet\chexnet_test.tfrecords"
tfrecords_eval = "E:\project data\chexnet\chexnet_eval.tfrecords"
densenet_test = "E:\project data\chexnet\densenet_test.tfrecords" # contains only 20 examples for testing purposes

db_121 = [6, 12, 24, 16]
db_169 = [6, 12, 32, 32]

# generate assignment map for loading pretrained weights
var_map = {}

# input layer weights are not loaded with pretrained weights due to shape mismatch
for i in range(4):
    for j in range(db_121[i]):
        var_map.update({'densenet121/dense_block{0}/conv_block{1}/x1/Conv/weights'.format(i+1, j+1): 'block{0}/bottleneck_layer.{1}/weights'.format(i+1, j+1)})
        var_map.update({'densenet121/dense_block{0}/conv_block{1}/x2/Conv/weights'.format(i+1, j+1): 'block{0}/dense_layer.{1}/weights'.format(i+1, j+1)})
#%%
# model

def DenseNet(features, labels, mode, params):
    
    depth = params["depth"]
    k = params["growth"]
    
    if depth == 121:
        N = db_121
    else:
        N = db_169
        
    bottleneck_output = 4 * k
    
    #before entering the first dense block, a conv operation with 16 output channels
    #is performed on the input images
    
    with tf.variable_scope('input_layer'):
        #l = tf.reshape(features, [-1, 224, 224, 1])
        feature_maps = 2 * k
        l = layers.conv(features, filter_size = 7, stride = 2, out_chn = feature_maps)
        l = tf.nn.max_pool(l,
                           padding='SAME',
                           ksize=[1,3,3,1],
                           strides=[1,2,2,1],
                           name='max_pool')
    
    # each block is defined as a dense block + transition layer
    with tf.variable_scope('block1'):
        for i in range(N[0]):
            with tf.variable_scope('bottleneck_layer.{}'.format(i+1)):
                bn_l = layers.batch_norm('BN', l)
                bn_l = tf.nn.relu(bn_l, name='relu')
                bn_l = layers.conv(bn_l, out_chn=bottleneck_output, filter_size=1)
            l = layers.add_layer('dense_layer.{}'.format(i+1), l, bn_l)
        l = layers.transition_layer('transition1', l)
    
    with tf.variable_scope('block2'):
        for i in range(N[1]):
            with tf.variable_scope('bottleneck_layer.{}'.format(i+1)):
                bn_l = layers.batch_norm('BN', l)
                bn_l = tf.nn.relu(bn_l, name='relu')
                bn_l = layers.conv(bn_l, out_chn=bottleneck_output, filter_size=1)
            l = layers.add_layer('dense_layer.{}'.format(i+1), l, bn_l)
        l = layers.transition_layer('transition2', l)
    
    with tf.variable_scope('block3'):
        for i in range(N[2]):
            with tf.variable_scope('bottleneck_layer.{}'.format(i+1)):
                bn_l = layers.batch_norm('BN', l)
                bn_l = tf.nn.relu(bn_l, name='relu')
                bn_l = layers.conv(bn_l, out_chn=bottleneck_output, filter_size=1)
            l = layers.add_layer('dense_layer.{}'.format(i+1), l, bn_l)
        l = layers.transition_layer('transition3', l)
    
    # the last block does not have a transition layer
    with tf.variable_scope('block4'):
        for i in range(N[3]):
            with tf.variable_scope('bottleneck_layer.{}'.format(i+1)):
                bn_l = layers.batch_norm('BN', l)
                bn_l = tf.nn.relu(bn_l, name='relu')
                bn_l = layers.conv(bn_l, out_chn=bottleneck_output, filter_size=1)
            l = layers.add_layer('dense_layer.{}'.format(i+1), l, bn_l)
    
    # classification (global max pooling and softmax)
    with tf.name_scope('classification'):
        l = layers.batch_norm('BN', l)
        l = tf.nn.relu(l, name='relu')
        l = layers.pooling(l, filter_size = 7)
        l_shape = l.get_shape().as_list()
        l = tf.reshape(l, [-1, l_shape[1] * l_shape[2] * l_shape[3]])
        l = tf.layers.dense(l, units = 1000, activation = tf.nn.relu, name='fc1', kernel_initializer=tf.contrib.layers.xavier_initializer())
        output = tf.layers.dense(l, units = 14, name='fc2', kernel_initializer=tf.contrib.layers.xavier_initializer()) # [batch_size, 14]
    
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output) # cost function
    cost = tf.reduce_mean(cross_entropy, name='cost_fn')

    # load pretrained weights
    tf.train.init_from_checkpoint('./pretrained_model/tf-densenet121.ckpt', 
                                  assignment_map=var_map)
    
    predictions = {
            'prob': tf.nn.sigmoid(output, name='sigmoid_tensor'),
            'labels': tf.round(tf.nn.sigmoid(output), name='labels')
    }

    # accuracy, _ = tf.metrics.accuracy(labels, predictions['labels'])
    correct_predictions = tf.equal(predictions['labels'], labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = tf.train.exponential_decay(learning_rate=0.001,
                                               global_step=tf.train.get_global_step(),
                                               decay_steps=200,
                                               decay_rate=0.1,
                                               staircase=True)
        
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=cost, global_step=tf.train.get_global_step())
        logging_hook = tf.train.LoggingTensorHook({"accuracy": accuracy,
                                                   "label": labels,
                                                   "predictions": predictions['prob'],
                                                   }, 
                                                  every_n_iter=50)
        return tf.estimator.EstimatorSpec(mode, loss=cost, train_op=train_op, training_hooks=[logging_hook])
    
    # evaluate
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
                'model_accuracy': accuracy
        }
        return tf.estimator.EstimatorSpec(mode, loss=cost, eval_metric_ops=metrics)
    
#%%

# parser for TFrecords
def parser(example, augmentation=True):
    features = tf.parse_single_example(
            example,
            features={
                    'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([14], tf.int64)
            })
        
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [1024, 1024, 1])
    image = tf.cast(image, tf.float32)
    # image = tf.image.central_crop(image, 0.9) # crop the central 80% of the image
    image = tf.image.resize_images(image, [224, 224]) # Bilinear interpolation
    image = tf.image.per_image_standardization(image) # normalize; ChexNet actually uses avg and std of the ImageNet training set
    #image = tf.image.random_flip_left_right(image)
    label = tf.cast(features['label'], tf.float32)
    
    return image, label

# input function for Estimator train()
def input_func(path=densenet_test):
    
    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(parser) # parsing TFrecords; performance improvements?
    ds = ds.shuffle(20)
    ds = ds.repeat()
    ds = ds.batch(2) 
    iterator = ds.make_one_shot_iterator()
    batch_img, batch_labels = iterator.get_next()
    
    return batch_img, batch_labels

#%%

# input function for Estimator eval()
def eval_func(path=tfrecords_eval):
    
    ds = tf.data.TFRecordDataset(path)
    def _parser(example):
        features = tf.parse_single_example(
        example,
        features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([14], tf.int64)
        })
    
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, [1024, 1024, 1])
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_images(image, [224, 224]) # Bilinear interpolation
        image = tf.image.per_image_standardization(image)
        label = tf.cast(features['label'], tf.float32)
        return image, label
    
    ds = ds.map(_parser) # parsing TFrecords; performance improvements?   
    ds = ds.repeat(1) # go through evaluaation set only once
    iterator = ds.make_one_shot_iterator()
    eval_img, eval_labels = iterator.get_next()
    
    return eval_img, eval_labels

#%%
# for handling inference requests after deployment; takes no arguments
# =============================================================================
# def receiver_func():
#     # specifies the input node; can be a Tensor or a dict of string to Tensor
#     receiver_tensor = tf.placeholder(dtype=tf.float32, name='input_node')
#     
#     # specifies the features to be passed to the model
#     features = tf.parse_example(serialized_example, 
#                                 features={'image': tf.FixedLenFeature([], tf.string)})
#     
#     
#     tmp = tf.decode_raw(features['image'], out_type=tf.float32) 
#     tmp = tf.image.resize_images(tmp, [224, 224]) # Bilinear interpolation
#     tmp = tf.image.per_image_standardization(tmp) # normalize
#     features['image'] = tmp
#     
#     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
# =============================================================================
#%%

#%%

def main(argv):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    chexnet = tf.estimator.Estimator(model_fn=DenseNet, 
                                     params={
                                             "depth": 121,
                                             "growth": 32
                                    },
                                    model_dir='./model',
                                    config=tf.estimator.RunConfig(session_config=config)) 

    chexnet.train(input_fn=input_func, steps=50000)
    # chexnet.export_savedmodel(receiver_func, export_dir_base='./model')
    results = chexnet.evaluate(input_fn=eval_func)
    print(results) # dict containing predicted labels and accuracy
    
#%%

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
    
#%%
