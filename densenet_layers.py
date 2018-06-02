import tensorflow as tf

#%%
def conv(batch_input, filter_size, out_chn, stride=1, is_trainable=True):
# =============================================================================
#     Arguments:
#         filter_size: assumes k x k square filter.
#         stride: stride length. default = 1
#         out_chn: number of filters
# =============================================================================
    
    #convolution
    shape = batch_input.get_shape().as_list()
    in_chn = shape[3]
    w = tf.get_variable(name='weights',
                        trainable=is_trainable,
                        shape=[filter_size,filter_size, in_chn, out_chn],
                        initializer=tf.contrib.layers.xavier_initializer())

    b = tf.get_variable(name='bias',
                        trainable=is_trainable,
                        shape=[out_chn],
                        initializer=tf.constant_initializer(0.0)) 
    batch_input = tf.nn.conv2d(batch_input, w, strides=[1, stride, stride, 1], padding="SAME", name='conv')
    
    #bias add
    batch_input = tf.nn.bias_add(batch_input, b, name='add_bias')
    
    return batch_input

#%%
def batch_norm(name, batch_input):
    
    # batch_input: 4D tensor [batch, length, width, depth]
    # no offset and scale
    # returns normalized input
    
    batch_mean, batch_variance = tf.nn.moments(batch_input, axes=[0])
    
    batch_input = tf.nn.batch_normalization(x = batch_input, 
                                           mean = batch_mean, 
                                           variance = batch_variance,
                                           offset=None,
                                           scale=None,
                                           variance_epsilon=1e-3,
                                           name=name)
    return batch_input

#%%
    

def pooling(batch_input, filter_size=2, p="avg"):
    
# =============================================================================
#     Arguments:
#         layer_name: name of layer
#         batch_input: 4D tensor [batch, l, w, depth]
#         filter_size: DenseNet uses 2x2 filter
#         p: type of pooling layer. DenseNet uses average pooling in the transition layers
#     
#     Returns:
#         pooled batch; 4D tensor
# =============================================================================
    
    if p == "max":
        return tf.nn.max_pool(value=batch_input,
                                 padding='SAME',
                                 ksize=[1,filter_size, filter_size, 1],
                                 strides=[1,filter_size, filter_size, 1],
                                 name='max_pool')
    if p == 'avg':
        return tf.nn.avg_pool(value=batch_input,
                                 padding='SAME',
                                 ksize=[1,filter_size, filter_size, 1],
                                 strides=[1,filter_size, filter_size, 1],
                                 name='avg_pool')
        
#%%                                 
def composite_func(l, growth=32):
    
        l = batch_norm('BN', l)
        l = tf.nn.relu(l, name='relu')
        l = conv(l, out_chn=growth, filter_size=3)
        
        return l
    
#%%
def add_layer(name, prev_l, l):
    
    with tf.variable_scope(name):
        l = composite_func(l)
        l = tf.concat([prev_l, l], 3)
        
    return l
#%%

def transition_layer(name, l, compression=0.5):
    # compression ratio: (0, 1)
    
    shape = l.get_shape().as_list()
    out_chn = shape[3] * compression
    with tf.variable_scope(name):
        l = batch_norm('BN', l)
        l = tf.nn.relu(l, name='relu')
        l = conv(l, out_chn=out_chn, filter_size=1)
        l = pooling(l)
    
    return l
        
        
