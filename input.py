import tensorflow as tf
import skimage.io as io
import pandas as pd
import os
from random import shuffle

IMAGE_DIR = "E:\project data\chexnet\images" # raw data directory
TFRECORD_DIR = "E:\project data\chexnet" # tfrecords directory


#%%

def int64_feature(value):
  # Wrapper for inserting int64 features into Example protocol
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
      # Wrapper for inserting byte features into Example protocol
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#%%

def load_files(IMAGE_DIR):
# =============================================================================
#   
#     Arguments:
#       IMAGE_DIR: directory of raw image data
#     
#     Returns:
#       image_paths: list of image directories (list of strings)
#       labels: one-hot vector representing the corresponding lung disease (list of ints)
# =============================================================================
    
    image_paths = []
    labels = []
    diseases = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
               "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
               "Fibrosis", "Pleural_Thickening", "Hernia"]
    classes = len(diseases)
    
    df = pd.read_csv('Data_Entry_2017.csv')

    # append image paths and one hot encoding
    for _, row in df[["Image Index", "Finding Labels"]].iterrows():
        image_paths.append(os.path.join(IMAGE_DIR, row["Image Index"]))
        label = row["Finding Labels"]
        
        if label == "No Finding":
            labels.append([0] * classes)
        else:
            # multi hot encoding
            multi_hot_vec = [0] * classes
            diagnosis = label.split("|")
            indices = [diseases.index(s) for _, s in enumerate(diagnosis)]
            
            for _, v in enumerate(indices):
                multi_hot_vec[v] = 1
                
            labels.append(multi_hot_vec)
    
    df['label vector'] = labels
    df['file_paths'] = image_paths
    patient_IDs = [df for _, df in df.groupby('Patient ID')]
    shuffle(patient_IDs)
    df = pd.concat(patient_IDs).reset_index(drop=True)
    df.to_csv('shuffled_dataset.csv', index=False) # save
    
    return df # shuffle dataframe grouped by patients

#%%

def write_TFRecords(dataset, name, TFRECORD_DIR, shards=1):
    
# =============================================================================
#     Converts preprocessed images into sharded TFRecords
#     
#     Arguments:
#       dataset: list of tuples containing image paths and label
#       name: name of the TFrecord file
#       TFRECORD_DIR: output directory
#     
#     Returns:
#       none
# =============================================================================

    shard_size = int(len(dataset) / shards)
    print("Writing TFRecords...")
    
    count = 0 # track progress
    ds_shards = [dataset[i:i+shard_size] for i in range(0, len(dataset), shard_size)]
    
    for num, shard in enumerate(ds_shards):
        fname = os.path.join(TFRECORD_DIR, name + '_{}.tfrecords'.format(num))
        with tf.python_io.TFRecordWriter(fname) as writer:
            for image_path, label in shard:
    
                # some files in Data_Entry_2017.csv could not be extracted; skipped
                if os.path.isfile(image_path) == False: 
                    continue
                
                try:
                    image = io.imread(image_path)
                    if image.shape != (1024,1024):
                        image = image[:,:,0] # some images are (1024,1024,4)
                    image_raw = image.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                                'label': int64_feature(label),
                                'image': bytes_feature(image_raw)
                                }))
                    writer.write(example.SerializeToString())
                    
                    # print some stuff
                    count += 1
                    if count % 1000 == 0:
                        print("Still working on it... Wrote {} files".format(count))
                    
                except IOError as err:
                    print("Image could not be read. Error: %s" %err)
                    print("Image skipped\n")
                except ValueError as err:
                    print("broken data stream")
        
    print("Conversion complete. Total files:", count)
    tmp = len(dataset) - count
    print("There were {} corrupt files".format(tmp))

#%%
df = load_files(IMAGE_DIR)
image_paths = list(df['file_paths'])
labels = list(df['label vector'])
ds = list(zip(image_paths, labels))

#%%
train, val, test = ds[0:78631], ds[78631:90014], ds[90014:] # 70/10/20 train/val/test split

print("Total files:", len(ds))
print("Training set:", len(train))
print('---------', train[-1])
print("Validation set:", len(val))
print('---------', val[-1])
print("Test set:", len(test))

#%%

# # writes a tfrecord file for train, validation, and training set
write_TFRecords(train, 'chexnet_train', TFRECORD_DIR, shards=30)
write_TFRecords(val, 'chexnet_val', TFRECORD_DIR)
write_TFRecords(test, 'chexnet_test', TFRECORD_DIR)

#%%

# generate a small dataset for testing purposes

ds = load_files(IMAGE_DIR)
small_ds = ds[0:20]
write_TFRecords(small_ds, 'densenet_test', TFRECORD_DIR)


#%%

# Code below inspects the TFRecords files to make sure everything is ok
# prints out a sample of images and corresponding labels

import matplotlib.pyplot as plt

sample = 0

for example in tf.python_io.tf_record_iterator("E:\project data\chexnet\chexnet_train_0.tfrecords"):
    if sample == 30:
        break
    result = tf.parse_single_example(example, features={
                                                        'image': tf.FixedLenFeature([], tf.string),
                                                        'label': tf.FixedLenFeature([14], tf.int64)
                                                        })
    image = tf.decode_raw(result['image'], tf.uint8)
    image = tf.reshape(image, [1024,1024,1])
    
    label = tf.cast(result['label'], tf.int32)
    #label = result['label']
    with tf.Session() as sess:
        img, label = sess.run([image, label])
        img = img.reshape([1024,1024])
        plt.imshow(img)
        plt.show()
        print(label)
    #print(result.features.feature['label'].int64_list.value)
    sample += 1

#%%

# inspect checkpoint of pretrained DenseNet

from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file("./pretrained_model/tf-densenet121.ckpt", tensor_name='', all_tensors=False)
chkp.print_tensors_in_checkpoint_file("./model/model.ckpt-1", tensor_name='', all_tensors=False)


    

