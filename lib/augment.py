from __future__ import print_function
import json
import numpy as np
import tensorflow as tf
import audiomentations
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TanhDistortion


DEFAULT_AUGMENTATIONS = '[\
    ["AddGaussianNoise", {"min_amplitude": 0.001, "max_amplitude": 0.015, "p": 0.5}],\
    ["TimeStretch", {"min_rate": 0.8, "max_rate": 1.25, "p": 0.5}],\
    ["PitchShift", {"min_semitones": -4, "max_semitones": 4, "p": 0.5}],\
    ["Shift", {"min_fraction": -0.5, "max_fraction": 0.5, "p": 0.5}],\
    ["Reverse", {"p": 0.5}]\
]'

def compose_augmentations(augmentations):
    augs = json.loads(augmentations)
    return Compose([
        getattr(audiomentations, method_name)(**kwargs)
        for (method_name, kwargs) in augs])


def augment(dataset, augmentations=DEFAULT_AUGMENTATIONS, sample_rate=16000):
    aug = compose_augmentations(augmentations)

    def map_fn(samples):
        return aug(np.array(samples), sample_rate)

    return dataset.map(lambda samples: tf.py_function(
        func=map_fn, inp=[samples], Tout=tf.float32
    ))