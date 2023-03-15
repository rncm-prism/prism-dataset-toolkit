'''
Concatenates .wav files from a directory. Sample rate, bit depth and channel count
of the output are inferred from the first source file.

Source files can be shuffled by passing the --shuffle argument.
'''

from __future__ import print_function
import argparse
import os
import fnmatch
import random
import wave


def check_boolean(value):
    val = str(value).upper()
    if 'TRUE'.startswith(val):
        return True
    elif 'FALSE'.startswith(val):
        return False
    else:
        raise ValueError('Argument is neither `True` nor `False`')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Path to the directory of files to concatenate.')
    parser.add_argument('--output_path', type=str, help='Path for the output file.')
    parser.add_argument('--shuffle', type=check_boolean, default=True, help='Whether to shuffle files before concatenating.')
    return parser.parse_args()

def find_wav_files(directory):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.wav'):
            files.append(os.path.join(root, filename))
    return files

def main():
    args = get_args()
    input_dir = args.input_dir
    output_path = args.output_path
    files = find_wav_files(input_dir)
    if args.shuffle==True: random.shuffle(files)
    data = []
    nchan = None
    sampwidth = None
    framerate = None
    for infile in files:
        wav = wave.open(infile, 'rb')
        if nchan == None: nchan = wav.getnchannels()
        if sampwidth == None: sampwidth = wav.getsampwidth()
        if framerate == None: framerate = wav.getframerate()
        nframes = wav.getnframes()
        data.append( wav.readframes(nframes) )
        wav.close()
    output = wave.open(output_path, 'wb')
    output.setnchannels(nchan)
    output.setsampwidth(sampwidth)
    output.setframerate(framerate)
    for i in range(len(data)):
        output.writeframes(data[i])
    template = 'Finished concatenating {} files from {} into {}'
    msg = template.format(len(files), input_dir, output_path)
    print(msg)

main()
