from transformers import WhisperForConditionalGeneration


def get_whisper():
    return WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')

# model.print_trainable_parameters()
