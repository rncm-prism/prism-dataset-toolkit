import numpy as np
import tensorflow as tf
from lib import load_audio 


def load(files, shuffle):
    dataset = tf.data.Dataset.from_generator(
        lambda: load_audio(files, shuffle=shuffle),
        output_types=tf.float32,
        output_shapes=((None, 1))
    )
    return dataset


def pad_batch(batch, batch_size, seq_len, overlap):
    num_samps = ( int(np.floor(len(batch[0]) / float(seq_len))) * seq_len )
    zeros = np.zeros([batch_size, overlap, 1], dtype='float32')
    return tf.concat([zeros, batch[:, :num_samps, :]], axis=1)

def pad(dataset, batch_size, seq_len, overlap):
    dataset = dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, overlap], Tout=tf.float32
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
            (batch_size, seq_len + overlap, 1),
            (batch_size, seq_len, 1)))