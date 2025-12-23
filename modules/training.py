from dataclasses import dataclass
from typing import Any, List, Union, Dict

import evaluate
import torch
from transformers import WhisperTokenizer

from config.variables import MODEL_VERSION, TOKENIZER_LANGUAGE

tokenizer = WhisperTokenizer.from_pretrained(MODEL_VERSION, language=TOKENIZER_LANGUAGE, task="transcribe")
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        attention_mask = torch.ones(batch["input_features"].shape[:-1], dtype=torch.long)
        is_padding = (batch["input_features"] == 0).all(dim=-1)
        attention_mask[is_padding] = 0
        batch["attention_mask"] = attention_mask

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
