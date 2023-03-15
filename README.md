# PRiSM Dataset Toolkit

A Python toolkit for building and manipulating audio data pipelines for TensorFlow.

Developed by PRiSM for use with neural audio synthesis models such as [PRiSM SampleRNN](https://github.com/rncm-prism/prism-samplernn).

## Installation

Install with `pip install -r ./requirements.txt`.

We highly recommend running the tools in a virtual environment, or with [Anaconda](https://www.anaconda.com/).

## Usage

The repository provides a library of tools that can be integrated into a TensorFlow data pipeline, built with the [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) API. It also incorporates augmentations from the [audiomentations](https://github.com/iver56/audiomentations) library, useful for generating augmented datasets.

Standalone python scripts are also provided for generating input data.

**N.B**: Currently the toolkit is restricted to _mono_ audio files, in WAV format.

### Basic Example

The following example shows how different elements from the toolkit can be integrated into a larger TensorFlow data pipeline.

```python
import lib as pdt

# Mono audio source to chunk.
input_wav = './Bloody-Ludwig.wav'

# Output directory for the chunks (will
# be created if it doesn't exist).
output_dir = './chunks'

# Create 8 second chunks (the default), with an
# overlap of 4 seconds between consecutive chunks.
pdt.create_chunks(input_wav, output_dir, chunk_length=8000, overlap=4000)

# If everything went to plan our source chunk directory will now be in the sepcified place...

# The following function builds a TensorFlow data pipeline incorporating functions
# from the toolkit. The first argument `data_dir` is the path to the directory
# of chunks we just created.
def get_dataset(data_dir, num_epochs=1, batch_size=32, seq_len=1024, shuffle=True):
    # Obtain the list of filenames from the
    # data dir and load them into the pipeline...
    files = pdt.find_files(data_dir)
    dataset = pdt.load(files, shuffle)
    # Apply some augmentations...
    dataset = pdt.augment(dataset)
    # The following step is standard for a data pieline, batching
    # the loaded audio and setting the number of epochs...
    drop_remainder = True
    dataset = dataset.repeat(num_epochs).batch(batch_size, drop_remainder)
    # Add zero padding to the start of each batch...
    target_offset = 64
    dataset = pdt.pad(dataset, batch_size, seq_len, target_offset)
    # Finally get the (X, y) batch subsequnces to be passed
    # as the direct input to the network...
    return pdt.get_cross_batch_sequence(dataset, batch_size, seq_len, target_offset)
```

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

Concatenates WAV files from a directory. Sample rate, bit depth and channel count of the output are inferred from the first source file.

Useful for building a larger WAV file from a collection of separate smaller ones, in order to be further processed.

#### _Command Line Arguments_

| Name                       | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `input_dir`              | Path to the directory of files to concatenate.          | `None`           | Yes        |
| `output_path`          | Path for the output file.           | `None`           | Yes        |
| `shuffle`              | Whether to shuffle files before concatenating.          | `True`           | No        |

## API

### `create_chunks`

Splits a WAV file into chunks, with optional overlap between consecutive chunks. The size of the chunks, and any overlap, are specified in milliseconds. Overlapping is simple but effective type of data augmentation.

Used internally by the `chunk_audio.py` script.

#### _Parameters_

| Name                       | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `input_file`              | Path to the input .wav file to be chunked.          | `None`           | Yes        |
| `output_dir`          | Path to the directory to contain the chunks. If the directory does not already exist it will be created.           | `None`           | Yes        |
| `chunk_length`              | Chunk length (defaults to 8000ms).          | 8000           | No        |
| `overlap`              | Overlap between consecutive chunks (defaults to 0ms, no overlap). | 0         | No        |

#### _Returns_

None.

### `load`

Generator for loading audio into a data pipeline.

#### _Parameters_

| Name                       | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `files`                    | List of path strings to the input WAV files.          | `None`           | Yes        |
| `shuffle`                  | PWhether to shuffle the list.           | `True`           | No        |

#### _Returns_

A Dataset.

### `pad`

Zero pads a batched dataset of audio buffers (tensors).

#### _Parameters_

| Name                       | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `dataset`                  | Input dataset (batched).          | `None`           | Yes        |
| `batch_size`               | Input dataset batch size.            | `None`           | Yes        |
| `seq_len`                  | Length of the subsequence (for cross-batch statefulness).          | `None`          | Yes        |
| `amount`                   | Number of zeroes to pad with. | `None`         | Yes        |

#### _Returns_

A Dataset.

### `augment`

Applies augmentations to an audio buffer, using the [audiomentations](https://github.com/iver56/audiomentations) library. Default augmentations are:

- `AddGaussianNoise`
- `TimeStretch`
- `PitchShift`
- `Shift`
- `Reverse`

Augmentations are specified as a JSON array, where each element is an array taking the form `[ augmentationName, parameters ]`:

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
| `augmentations`          | List of augmentations to apply.           | See above           | No        |

#### _Returns_

A Dataset.

### `get_cross_batch_sequence`

Generator for obtaining batch slices, useful for implementing the [cross batch statefulness](https://www.tensorflow.org/guide/keras/rnn#cross-batch_statefulness) pattern.

#### _Parameters_

| Name                       | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `dataset`                  | Input dataset (batched).          | `None`           | Yes        |
| `batch_size`               | Input dataset batch size.           | `None`           | Yes        |
| `seq_len`                  | Length of the subsequence.          | `None`           | Yes        |
| `target_offset`                  | Offset of the target. | `None`         | Yes        |

#### _Returns_

A Dataset. Each yielded element is a Numpy array of the form `array(X, y)`, where `X` is the input data for the network, and `y` is the offset target.