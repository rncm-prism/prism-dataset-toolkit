import os
import sys
import argparse
from pydub import AudioSegment
from pydub import silence
from lib import create_chunks

parser = argparse.ArgumentParser(description='Splits a WAV file into chunks, with optional overlap between consecutive chunks.')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input .wav file')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the chunks')
parser.add_argument('--chunk_length', type=int, default=8000, help='Output chunk size in milliseconds')
parser.add_argument('--overlap', type=int, default=0, help='Overlap between consecutive chunks in milliseconds')

args = parser.parse_args()

input_file = args.input_file
output_dir = args.output_dir
chunk_length = args.chunk_length
overlap = args.overlap

# Create output dir if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

create_chunks(input_file, output_dir, chunk_length=chunk_length, overlap=overlap)