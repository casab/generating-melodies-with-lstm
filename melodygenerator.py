import json
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class MelodyGenerator:
    def __init__(self, model_path="model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        # seed = "64 _ 63 _ _"

        # Create seed with start symbol
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # Map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # Limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # One-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # Make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # Update seed
            seed.append(output_int)

            # Map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # Check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # Update the melody
            melody.append(output_symbol)

        return melody

    @staticmethod
    def _sample_with_temperature(probabilities, temperature):
        # Temperature -> infinity
        # Temperature -> 0
        # Temperature -> 1
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))  # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.midi"):
        pass
        # Create a music21 stream
        stream = m21.stream.Stream()

        # Parse all the symbols in the melody and create note/rest objects
        # 60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            # Handle case in which we have a note/rest
            if symbol != "_" or i == len(melody) - 1:
                # Ensure we're not dealing with first event
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter
                    # Handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quaterLength=quarter_length_duration)

                    # Handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quaterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # Reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # Handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # Write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "67 _ _ _ 72 _ _ _ 72 _ _ _ 76 _ 74 _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.4)
    print(melody)
    mg.save_melody(melody)
