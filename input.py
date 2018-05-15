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
    df = df.head(10)
    
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
    
    dataset = list(zip(image_paths, labels))
    shuffle(dataset)
    
    return dataset # list of tuples [(image_paths, label), ... ]

#%%

def write_TFRecords(dataset, name, TFRECORD_DIR):
    
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
    
    fname = os.path.join(TFRECORD_DIR, name + '.tfrecords')
    print("Writing TFRecords...")
    
# =============================================================================
#     for i in range(0, len(image_paths), num_files):
#         # Set TFrecords file name
#         if num_files == len(image_paths):
#             fname = os.path.join(TFRECORD_DIR, name + '.tfrecords')
#             image_slice = image_paths
#         else:
#             fname = os.path.join(TFRECORD_DIR, name + str(file_count) + '.tfrecords')
#             image_slice = image_paths[i:i+num_files]
# =============================================================================
            
    with tf.python_io.TFRecordWriter(fname) as writer:
        
        count = 0
        
        for image_path, label in zip(*dataset):        
            try:
                image = io.imread(image_path)
                assert image.shape == (1024, 1024)
                image_raw = image.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                            'label': int64_feature(label),
                            'image': bytes_feature(image_raw)
                            }))
                writer.write(example.SerializeToString())
                
            except IOError as err:
                count += 1
                print("Image could not be read. Error: %s" %err)
                print("Image skipped\n")
        
    print("Conversion complete")
    print("There were {} corrupt files".format(count))

#%%

ds = load_files(IMAGE_DIR)
t, cv = int(103568 * 0.935), int(103568 * 0.06)
train, val, test = ds[0:t], ds[t:t+cv], ds[cv:] # 93.5/6/0.5 train/val/test split

# writes a tfrecord file for train, validation, and training set
write_TFRecords(train, 'chexnet_train', TFRECORD_DIR)
write_TFRecords(val, 'chexnet_val', TFRECORD_DIR)
write_TFRecords(test, 'chexnet_test', TFRECORD_DIR)



