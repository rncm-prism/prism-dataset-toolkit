import os
import sys
from pydub import AudioSegment
from pydub import silence


# Helper that checks for silent chunks
def is_silent(chunk, chunk_length, silence_thresh=-64):
    # if at least half of chunk is silence, mark as silent
    silences = silence.detect_silence(
        chunk,
        min_silence_len=int(chunk_length/2),
        silence_thresh=silence_thresh)
    if silences:
        return True
    else:
        return False

# Chunker
def create_chunks(input_file, output_dir, **kwargs):
    chunk_length = kwargs.get('chunk_length', 8000)
    overlap = kwargs.get('overlap', 2000)
    silence_thresh = kwargs.get('silence_thresh', -64)

    # Create output dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load file and set directories
    audio = AudioSegment.from_wav(input_file)
    input_filename = input_file.split('/')[-1].replace('.wav', '')

    # Get length of audio
    audio_len = len(audio)

    # Initialize start and end seconds to 0
    start = 0
    end = 0

    # Iterate from 0 to end of the file,
    # with increment = chunk_length
    cnt = 0
    num_silent = 0
    flag = 0 # Break loop once we reach the end of the audio

    # Main loop.
    for i in range(0, 8 * audio_len, chunk_length):

        # At first, start is 0, end is the chunk_length
        # Else, start=prvs end-overlap, end=start+chunk_length
        if i == 0:
            start = 0
            end = chunk_length
        else:
            start = end - overlap
            end = start + chunk_length

        # Set flag to 1 if endtime exceeds length of file
        if end >= audio_len:
            end = audio_len
            flag = 1

        # Storing audio file from the defined start to end
        chunk = audio[start:end]
        if flag == 0:
            cnt = cnt + 1
            chunk_is_silent = is_silent(chunk, chunk_length, silence_thresh)
            if chunk_is_silent:
                print('Chunk {} is silent, omitting it.'.format(cnt))
                num_silent = num_silent + 1
            else:
                filename = input_filename + f'_chunk_{cnt}.wav'
                chunk.export(os.path.join(output_dir, filename), format="wav")
                print("Processing chunk " + str(cnt) + ". Start = "
                        + str(start) + " end = " + str(end))

    print('\n')
    print("Finished chunking {}.".format(input_file))
    print("{} chunks processed, {} were silent.".format(cnt, num_silent))
    print("Saved {} chunks to {}.".format(cnt - num_silent, output_dir))
