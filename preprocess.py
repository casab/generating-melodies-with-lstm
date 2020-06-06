import os
import music21 as m21
from typing import Optional, Any

KERN_DATASET_PATH = "deutschl/test"
SAVE_DIR = "dataset"
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
    encoded_song = "".join(map(str, encoded_song))
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


if __name__ == "__main__":
    # TODO: WILL BE REMOVED. Some tests are done here.
    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs.")
    song = songs[0]

    preprocess(KERN_DATASET_PATH)
    # transpose song
    transposed_song = transpose(song)
    transposed_song.show()
