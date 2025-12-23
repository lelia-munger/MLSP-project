Authors: Manuel Podeur and Lélia Munger

# Description

This model is meant to train.

# How to use

## Data

To get the data, you need to make csv file with the columns text and audio where audio is the file paths of the
audio files.

When this is done, run the `csv_to_arrow.py` after changing the input csv file name.

The dataset folder should be placed in the `data` folder. If not, put it there.

The dataset used (Common Voice 24.0 FR) was manually filtered in Excel to remove the rows that were not Québécois or
French Canadian.

## Training

To train the model, open the `notebooks -> trainer.ipynb` and run the cells. Everything should be in place for the
trainer to be able to start.
Everything cell of this notebook should be well titled to say what it does.


