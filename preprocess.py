import os
import json
import music21 as m21
from typing import Optional, Any
import numpy as np
import tensorflow.keras as keras

KERN_DATASET_PATH = "deutschl/test"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

# Durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]


def load_songs_in_kern(dataset_path: str):
    """
    go through all the files in the dataset and load them with music21
    :param dataset_path:
    :return:
    """
    songs = []

    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_duration(song: Optional[Any], acceptable_durations) -> bool:
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song: Optional[Any]) -> Optional[Any]:
    """
    Transpose the song to A minor or C major depending on the mode.
    :param song:
    :return:
    """
    # Get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # Estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # Get interval for transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # Transpose song by calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song


def encode_song(song, time_step=0.25) -> str:
    # p = 60, d = 1.0 -> [60, "_", "_", "_"]
    encoded_song = []

    for event in song.flat.notesAndRests:
        # Handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi  # 60
        # Handle rests
        if isinstance(event, m21.note.Rest):
            symbol = "r"

        # Convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # Cast encoded song to a str
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song


def preprocess(dataset_path):
    # Load the folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):
        # Filter out songs that have non-acceptable durations
        if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
            continue

        # Transpose songs to Cmaj/Amin
        song = transpose(song)

        # Encode songs with music time series represantation
        encoded_song = encode_song(song)

        # Save songs to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # Load encoded songs and add delimeters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs += song + " " + new_song_delimiter

    songs = songs[:-1]
    # Save string that contains all dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, mapping_path):
    mappings = {}
    # Identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # Create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # Save vocabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    # Load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # Cast songs string to a list
    songs = songs.split()

    # Map songs to int
    int_songs = [mappings[symbol] for symbol in songs]

    return int_songs


def generate_training_sequences(sequence_length):
    # [11, 12, 13, 14, ...] -> i: [11, 12], t: 13; i: [12, 13], t: 14;

    # Load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    # Generate the training sequences
    # 100 symbols, 64 sl, 36 sequences
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i + sequence_length])
        targets.append(int_songs[i + sequence_length])

    # One-hot encode the sequences
    # inputs: (# of sequences, sequence length, vocabulary size)
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
