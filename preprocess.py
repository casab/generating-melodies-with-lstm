import os
import music21 as m21
from typing import Optional, Any

KERN_DATASET_PATH = "deutschl/test"
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


def preprocess(dataset_path):
    # Load the folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for song in songs:
        # Filter out songs that have non-acceptable durations
        if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to Cmaj/Amin
        song = transpose(song)

        # TODO: encode songs with music time series represantation

        # TODO: save songs to text file


if __name__ == "__main__":
    # TODO: WILL BE REMOVED. Some tests are done here.
    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs.")
    song = songs[0]
    print(song)
    print(f"Has acceptable duration? {has_acceptable_duration(song, ACCEPTABLE_DURATIONS)}")
    transpose(song).show()
