import os
import fnmatch
import numpy as np
import tensorflow as tf
from lib import load_audio 


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def load(files, shuffle):
    dataset = tf.data.Dataset.from_generator(
        lambda: load_audio(files, shuffle=shuffle),
        output_types=tf.float32,
        #output_shapes=((None, 1))
    )
    return dataset


def pad_batch(batch, batch_size, seq_len, amount):
    num_samps = ( int(np.floor(len(batch[0]) / float(seq_len))) * seq_len )
    zeros = np.zeros([batch_size, amount], dtype='float32')
    return tf.concat([zeros, batch[:, :num_samps]], axis=1)

def pad(dataset, batch_size, seq_len, amount):
    dataset = dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, amount], Tout=tf.float32
    ))
    return dataset


def get_batch_subseq(dataset, batch_size, seq_len, overlap):
    for batch in dataset:
        num_samps = len(batch[0])
        for i in range(overlap, num_samps, seq_len):
            x = batch[:, i-overlap : i+seq_len]
            y = x[:, overlap : overlap+seq_len]
            yield (x, y)

def get_cross_batch_sequence(dataset, batch_size, seq_len, overlap):
    return tf.data.Dataset.from_generator(
        lambda: get_batch_subseq(dataset, batch_size, seq_len, overlap),
        output_types=(tf.int32, tf.int32),
        output_shapes=(
            (batch_size, seq_len + overlap),
            (batch_size, seq_len)))


def print_dataset(dataset):
    for elt in dataset.as_numpy_iterator():
        print(elt)