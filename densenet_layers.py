import tensorflow as tf

#%%
def conv(name, batch_input, filter_size, stride=1, out_chn, is_trainable=True):
# =============================================================================
#     Arguments:
#         filter_size: assumes k x k square filter.
#         stride: stride length. default = 1
#         out_chn: number of filters
# =============================================================================
    
    #convolution
    in_chn = tf.shape(batch_input)[-1]
    h, w = filter_size
    
    w = tf.get_variable(name='weights',
                        trainable=is_trainable,
                        shape=[h, w, in_chn, out_chn],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name='bias',
                        trainable=is_trainable,
                        shape=[out_chn],
                        initializer=tf.constant_initializer(0.0)) 
    batch_input = tf.nn.conv2d(batch_input, w, strides=[1, stride, stride, 1], padding="SAME", name=name)
    
    #bias add
    batch_input = tf.nn.bias_add(batch_input, b, name='add_bias')
    
    return batch_input

#%%
def batch_norm(batch_input):
    
    # batch_input: 4D tensor [batch, length, width, depth]
    # returns normalized input
    
    epsilon = 1e-3
    batch_mean, batch_variance = tf.nn.moments(batch_input, axis=[0])
    
    gamma = tf.get_variable(name="scale", 
                            shape=tf.shape(batch_input)[-1],
                            intializer=tf.ones_initializer())
    beta = tf.get_variable(name="offset",
                           shape=tf.shape(batch_input)[-1],
                           intializer=tf.zeros_initializer())
    
    batch_input = tf.nn.batch_normalization(x = batch_input, 
                                           mean = batch_mean, 
                                           variance = batch_variance,
                                           offset = beta,
                                           scale = gamma,
                                           variance_epsilon = epsilon,
                                           name='BN')
    return batch_input

#%%
    

def pooling(layer_name, batch_input, filter_size=2, p="avg"):
    
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
        return tf.nn.max_pooling(value=batch_input,
                                 padding='SAME',
                                 ksize=filter_size,
                                 strides=[1,filter_size, filter_size, 1]
                                 name=layer_name)
    else:
        return tf.nn.avg_pooling(value=batch_input,
                                 padding='SAME',
                                 ksize=filter_size,
                                 strides=[1,filter_size, filter_size, 1]
                                 name=layer_name)
        
#%%                                 
def composite_func(l, growth=k):
    
        l = batch_norm('BN', l)
        l = tf.nn.relu(l, name='relu')
        l = conv('conv', l, filter_size=3, growth)
        
        return l
    
#%%
def add_layer(name, l):
    
    with tf.variable_scope(name):
        c = composite_func(l)
        c = tf.concat([c, l], 3)
        
    return c
#%%

def transition_layer(name, l):
    out_chn = tf.shape(l).shape[3]
    with tf.variable_scope(name):
        l = batch_norm('BN', l)
        l = tf.nn.relu(l, name='relu')
        l = conv('conv', l, filter_size=3, out_chn)
        l = pooling('pooling', l, 2)
        
        
