from datasets import load_from_disk, load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from config.variables import *


def process_raw_data(dataset):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_VERSION)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_VERSION,
                                                 language="French",
                                                 task="transcribe")

    def prepare_dataset(batch):
        input_features = []
        labels = []

        for audio, text in zip(batch["audio"], batch["text"]):
            # audio["array"] contient le signal audio
            features = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
            input_features.append(features)

            # encode target text
            label_ids = tokenizer(text).input_ids
            labels.append(label_ids)

        batch["audio"] = input_features
        batch["text"] = labels
        return batch

    return dataset.map(prepare_dataset,
                       batched=True,
                       batch_size=100,
                       keep_in_memory=True)


def get_quebecois_raw():
    if FC_DATA_RAW.exists():
        fc_data = load_from_disk(FC_DATA_RAW)
        print(fc_data['train']['audio'])
    else:
        fc_data = DatasetDict()
        fc_data["train"] = load_dataset("rishabbahal/quebecois_canadian_french_dataset", "default", split="train")
        fc_data["test"] = load_dataset("rishabbahal/quebecois_canadian_french_dataset", "default", split="test")
        fc_data = fc_data.remove_columns(["audio_filepath", "__index_level_0__"])
    return fc_data


def load_quebecois_data(save=False):
    if FC_DATA_PROCESSED.exists():
        processed = load_from_disk(FC_DATA_PROCESSED)
        print(processed)
        return processed
    raw = get_quebecois_raw()
    print(raw)
    processed = process_raw_data(raw)
    if save:
        processed.save_to_disk(FC_DATA_PROCESSED)
    print(processed)
    return processed


def load_common_voice_data():
    if FC_COMMON_VOICE_PROCESSED.exists():
        processed = load_from_disk(FC_COMMON_VOICE_PROCESSED)
        print(processed)
        return processed
    elif FC_COMMON_VOICE_RAW.exists():
        raw = load_from_disk(FC_COMMON_VOICE_RAW)
        processed = process_raw_data(raw)
        print(processed)
        return processed
    raise FileNotFoundError()


def combine_datasets(dataset1, dataset2):
    raise NotImplementedError()
