import tensorflow as tf
import numpy as np
import skimage.io as io
import skimage.transform as st
import pandas as pd
import os

IMAGE_DIR = "E:\project data\chexnet\images" # raw data directory
TFRECORD_DIR = "" # output dir


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
            #one hot encoding
            one_hot_vec = [0] * classes
            diagnosis = label.split("|")
            indices = [diseases.index(s) for _, s in enumerate(diagnosis)]
            
            for _, v in enumerate(indices):
                one_hot_vec[v] = 1
                
            labels.append(one_hot_vec)
    
    return image_paths, labels

#%%

def write_TFRecords(image_paths, labels, num_files=len(image_paths), name, TFRECORD_DIR):
    
# =============================================================================
#     Converts preprocessed images into sharded TFRecords
#     
#     Arguments:
#       image_paths: paths of image files
#       labels: one hot vector representing disease class
#       num_files: number of files per shard
#       name: name of the TFrecord file
#       TFRECORD_DIR: output directory
#     
#     Returns:
#       none
# =============================================================================
    if len(image_paths) != len(labels):
        raise ValueError("There are %d image files and %d labels." %(len(image_paths), len(labels)))
    
    if num_files > len(image_paths):
        raise ValueError("Cannot have shard size greater than the total number of images")
    
    file_count = 0
    print("Writing TFRecords...")
    
    for i in range(0, len(image_paths), num_files):
        # Set TFrecords file name
        if num_files == len(image_paths):
            fname = os.path.join(TFRECORD_DIR, name + '.tfrecords')
            image_slice = image_paths
        else:
            fname = os.path.join(TFRECORD_DIR, name + str(file_count) + '.tfrecords')
            image_slice = image_paths[i:i+num_files]
            
        with tf.python_io.TFRecordWriter(fname) as writer:
            for i, file_path in enumerate(image_slice):        
                try:
                    image = io.imread(file_path)
                    assert image.shape == (1024, 1024)
                    image_raw = image.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                                'label':int64_feature(labels[i]),
                                'image_raw': bytes_feature(image_raw)
                                }))
                    writer.write(example.SerializeToString())
                    
                except IOError as err:
                    print("Image could not be read. Error: %s" %err)
                    print("Image skipped\n")
        file_count += 1
        
    print("Conversion complete")

#%%

