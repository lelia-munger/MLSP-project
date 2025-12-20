from datasets import load_from_disk
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

from config.variables import *

# file = Path(sys.argv[1])
# if not file.exists():
#     print(file)
#     print("File does not exist")
#     exit(0)

fc_data = load_from_disk(FC_DATA_RAW)

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_VERSION)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_VERSION,
                                             language="French",
                                             task="transcribe")
processor = WhisperProcessor.from_pretrained(MODEL_VERSION,
                                             language="French",
                                             task="transcribe")


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    audio_samples = audio.get_all_samples()
    audio_array = audio_samples.data.squeeze(0).numpy()

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio_array, sampling_rate=16000).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


fc_data_processed = fc_data.map(prepare_dataset,
                                remove_columns=fc_data.column_names["train"],
                                batched=True)
fc_data_processed.save_to_disk(FC_DATA_PROCESSED)
