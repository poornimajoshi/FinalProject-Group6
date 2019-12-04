# FinalProject-Group6
# Detecting photoshopped photos from real ones

## Data
Download the data from `https://www.kaggle.com/ciplab/real-and-fake-face-detection/download`
Uncompress it into the data directory and save the path to the data.

## Usage
Run `create_train_test_set.py` to create the train test split of the data.
```
python3 create_train_test_set.py
```
Run `baseline_model.py` in drn_26_baseline to get the baseline model.
```
python3 baseline_model.py
```
Run `train.py` in Resnet18 folder which generates a csv file.
```
python3 train.py
```
Run `mobilenet.py` in MobileNetV2 folder which generates a csv file.
```
python3 mobilenet.py
```
Run `train_fastai.py` to run and see results of resnet, vvg16 and dense101 and output predictions in pickle files
```
python3 train_fastai.py
```
Move all the csv files to the pickle folder.

Finally run the `ensemble_example.py` file for the final result.

