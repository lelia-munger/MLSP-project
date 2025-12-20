from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FC_DATA_RAW = PROJECT_DIR / 'data' / 'fc_data_raw'
FC_COMMON_VOICE_DATA_RAW = PROJECT_DIR / 'data' / 'fc_common_voice_raw'
FC_DATA_PROCESSED = PROJECT_DIR / 'data' / 'fc_data_processed'
MODEL_VERSION = "openai/whisper-small"
