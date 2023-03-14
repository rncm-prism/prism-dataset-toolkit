# PRiSM Dataset Toolkit

A Python toolkit for building and manipulating audio data pipelines for TensorFlow.

Developed by PRiSM for use with neural synthesis models such as [PRiSM SampleRNN](https://github.com/rncm-prism/prism-samplernn).

## Installation

Install with `pip install -r ./requirements.txt`.

We recommend running the tools in an Anaconda environment.

## Usage

The repository provides a library of tools that can be integrated into a TensorFlow data pipeline, built with the [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) API. It also incorporates augmentations from the [audiomentations](https://github.com/iver56/audiomentations) library, useful for generating augmented datasets.

Standalone python scripts are also provided for generating input data.

**N.B**: Currently the toolkit is restricted to mono audio files, in WAV format.

## Scripts

### `chunk_audio.py`

Splits a WAV file into chunks, with optional overlap between consecutive chunks. The size of the chunks, and any overlap, are specified in milliseconds.

#### _Command Line Arguments_

| Name                       | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `input_file`              | Path to the input .wav file to be chunked.          | `None`           | Yes        |
| `output_dir`          | Path to the directory to contain the chunks. If the directory does not already exist it will be created.           | `None`           | Yes        |
| `chunk_length`              | Chunk length (defaults to 8000ms).          | 8000           | No        |
| `overlap`              | Overlap between consecutive chunks (defaults to 0ms, no overlap). | 0         | No        |

Example usage:

```shell
python chunk_audio.py \
  --input_file path/to/input.wav \
  --output_dir ./chunks \
  --chunk_length 8000 \
  --overlap 4000
```

### `concat_audio.py`

#### _Command Line Arguments_

| Name                       | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `input_file`              | Path to the input .wav file to be chunked.          | `None`           | Yes        |
| `output_dir`          | Path to the directory to contain the chunks. If the directory does not already exist it will be created.           | `None`           | Yes        |
| `chunk_length`              | Chunk length (defaults to 8000ms).          | 8000           | No        |
| `overlap`              | Overlap between consecutive chunks (defaults to 0ms, no overlap). | 0         | No        |

## API

### `load`

Generator for loading audio into a data pipeline.

#### _Parameters_

| Name                       | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `files`                    | Path to the generated .wav file.          | `None`           | Yes        |
| `shuffle`                  | Path to a saved checkpoint for the model.           | `None`           | Yes        |

#### _Returns_

A dataset.

### `pad`

Zero pads an audio buffer (tensor).

#### _Parameters_

| Name                       | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `dataset`                  | Input dataset.          | `None`           | Yes        |
| `batch_size`               | Input dataset batch size.            | `None`           | Yes        |
| `seq_len`                  | Length of the subsequence.          | 8000           | No        |
| `overlap`                  | Overlap between consecutive chunks. | 0         | No        |

#### _Returns_

A dataset.

### `augment`

Applies augmentations to an audio buffer, using the [audiomentations](https://github.com/iver56/audiomentations) library. Default augmentations are:

- `AddGaussianNoise`
- `TimeStretch`
- `PitchShift`
- `Shift`
- `Reverse`

Augmentations are specified in JSON format as follows:

```json
[
    [
        "AddGaussianNoise",
        {
            "min_amplitude": 0.001,
            "max_amplitude": 0.015,
            "p": 0.5
        }
    ],
    [
        "TimeStretch",
        {
            "min_rate": 0.8,
            "max_rate": 1.25,
            "p": 0.5
        }
    ],
    [
        "PitchShift",
        {
            "min_semitones": -4,
            "max_semitones": 4,
            "p": 0.5
        }
    ],
    [
        "Shift",
        {
            "min_fraction": -0.5,
            "max_fraction": 0.5,
            "p": 0.5
        }
    ],
    [
        "Reverse",
        {
            "p": 0.5
        }
    ]
]
```

For the full list of available augmentations see the [audiomentations documentation](https://iver56.github.io/audiomentations/).

#### _Parameters_

| Name                       | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `dataset`              | Input dataset.          | `None`           | Yes        |
| `augmentations`          | List of augmentations to apply.           | `None`           | Yes        |

#### _Returns_

A dataset.

### `get_cross_batch_sequence`

Generator for obtaining batch slices, useful for implementing the [cross batch statefulness](https://www.tensorflow.org/guide/keras/rnn#cross-batch_statefulness) pattern.

#### _Parameters_

| Name                       | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `dataset`                  | Input dataset.          | `None`           | Yes        |
| `batch_size`               | Input dataset batch size.           | `None`           | Yes        |
| `seq_len`                  | Length of the subsequence.          | 8000           | No        |
| `overlap`                  | Overlap between consecutive chunks. | 0         | No        |

#### _Returns_

A dataset.