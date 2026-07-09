## Data Downloading
The data can be found from [this url](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V).  Since the data for train/validate/test
is very large, we  split each data set into small chunks, which can be found in the directory ending with `_chunks`, such as `train_chunks`. After downloading, please run the following command to each set to merge those chunks together:

```
cat train.zip.part* > train.zip
unzip train.zip
```
If you have good internet, you can also directly download the whole zip file, e.g. train.zip

For the semantic label, you should follow [this project](https://github.com/rruisong/CoHFF) to download the simulated 4 semantic LiDARs dataset and generate the label with it. For a convenient inference, we provide a data sample set covering 3 different scenes; you can just download them in [this url](https://drive.google.com/drive/folders/1l6HMFD7eRpP9PJ4pGwpuKned1Het2otS?usp=sharing)

## Installation
```python
conda create -n attens python=3.8
conda activate attens
# pytorch >= 1.8.1, newest version can work well
conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch
# spconv 2.0 install, choose the correct cuda version for you
pip install spconv-cu113
# Install dependencies
pip install -r requirements.txt
python setup.py develop
```

## Quick Start
### Pre-trained Weights
The pre-trained model can be downloaded from [this url](https://drive.google.com/drive/folders/1l6HMFD7eRpP9PJ4pGwpuKned1Het2otS?usp=sharing)


### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder refers to the testing dataset path, e.g. `v2xset/test`.

```python
python opencood/tools/inference.py --model_dir opencood/logs/zero-r --cal_comm

python opencood/tools/inference.py --model_dir opencood/logs/cobevt --cal_comm

python opencood/tools/inference.py --model_dir opencood/logs/maxpool --cal_comm
```