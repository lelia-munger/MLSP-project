from datasets import load_dataset, load_from_disk, DatasetDict

from config.variables import FC_DATA_RAW

fc_data = None
if FC_DATA_RAW.exists():
    fc_data = load_from_disk(FC_DATA_RAW)
    print(fc_data['train']['audio'])
else:
    fc_data = DatasetDict()

    fc_data["train"] = load_dataset("rishabbahal/quebecois_canadian_french_dataset", "default", split="train")
    fc_data["test"] = load_dataset("rishabbahal/quebecois_canadian_french_dataset", "default", split="test")
    fc_data = fc_data.remove_columns(["audio_filepath", "__index_level_0__"])
    fc_data.save_to_disk(FC_DATA_RAW)
