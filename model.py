import tensorflow as tf
import glob

TFRecords_dir = ""
fpaths = glob.glob(TFRecords_dir + "*.tfrecords")

#%%

# read from TFRecords
ds = tf.data.TFRecordDataset(fpaths)
# model

def DenseNet(batch_img, depth = 121, k = 12):
    
    # dense block 1
    
    #transition layer
    
    # dense block 2
    
    # ...
    
    # classification (global max pooling and softmax)
    
    return None