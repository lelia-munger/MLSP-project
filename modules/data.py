from datasets import load_from_disk

from config.variables import FC_DATA_PROCESSED


def get_quebecois_data():
    if FC_DATA_PROCESSED.exists():
        return load_from_disk(FC_DATA_PROCESSED)
    else:
        raise FileNotFoundError(FC_DATA_PROCESSED)
