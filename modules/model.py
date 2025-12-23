from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor

from config.variables import MODEL_VERSION, TOKENIZER_LANGUAGE


def get_model():
    return WhisperForConditionalGeneration.from_pretrained(MODEL_VERSION)


def get_tokenizer():
    return WhisperTokenizer.from_pretrained(MODEL_VERSION,
                                            language=TOKENIZER_LANGUAGE,
                                            task='transcribe')


def get_processor():
    return WhisperProcessor.from_pretrained(MODEL_VERSION,
                                            language=TOKENIZER_LANGUAGE,
                                            task='transcribe')
