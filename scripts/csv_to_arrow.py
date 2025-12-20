from pathlib import Path

import librosa
from datasets import load_dataset, Audio

from config.variables import FC_COMMON_VOICE_DATA_RAW

# The CSV must contain path: the name of the file and text, the text of the audio
audio_column = 'audio'

common_voice_fc = (load_dataset('data', data_files='common_voice_fc.csv', split='train', verification_mode=None)
                   .train_test_split(test_size=0.2))

clips_path = Path("../cv-corpus-23.0-2025-09-05/fr/clips/")


def retrieve_audio(file_name):
    audio_path = clips_path.joinpath(file_name)
    audio_data, sample_rate = librosa.load(audio_path)

    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    if audio_bytes is None:
        print("No audio file found")

    return {
        "path": str(audio_path),
        "sampling_rate": sample_rate,
        "bytes": audio_bytes
    }


def normalize_data(batch):
    audio_file_name_array = list(map(retrieve_audio, batch['path']))
    batch[audio_column] = audio_file_name_array
    return batch


common_voice_fc = common_voice_fc.map(normalize_data,
                                      remove_columns=['path'],
                                      batched=True,
                                      batch_size=100)
common_voice_fc = common_voice_fc.cast_column(audio_column, Audio())
common_voice_fc.shuffle()

common_voice_fc.save_to_disk(FC_COMMON_VOICE_DATA_RAW)

print(common_voice_fc)
