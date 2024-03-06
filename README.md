# TCN-beat-tracker
PyTorch implementation of [Matthew E. P. Davies; Sebastian Böck. "Temporal convolutional networks for musical audio beat tracking". 2019 27th European Signal Processing Conference (EUSIPCO)](https://ieeexplore.ieee.org/document/8902578) for "Music Informatics" course at Queen Mary University of London.

### Installation
You need to create a python >3.9 environment (e.g. with conda) and install the requirements:
```
pip install -r requirements.txt
```

### Running
A trained beat and downbeat model is provided in this repo. You can either import the beat tracker method in your code and run it as such:
```python
from beat_tracker import beatTracker
beats, downbeats = beatTracker(<path_to_your_audio_file>)
```
or through command line arguments:
```shell
python beat_tracker -i <path_to_your_audio_file>
```

### Reproducing results

##### Training
You need to download the Ballroom dataset, uncompress it, and place it in the right directory.
```shell
wget http://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz  # download
untar -xvf data1.tar.gz  # uncompress
mkdir data/ballroom  # create data dir
mv BallroomData data/ballroom/audio  # move audio to data dor
git clone git@github.com:CPJKU/BallroomAnnotations.git  # clone repo with annotations
mv BallroomAnnotations data/ballroom/annotations  # move to annotations dir
```
You'll then be ready to train a beat or downbeat model:
```shell
python train.py -t <model_type>
```
where <model_type> is either `beats` or `downbeats`.

##### Evaluation
To evaluate your model, you can run:
```shell
python evaluate.py -n <model_name> -t <model_type>
```
where <model_type> is either `beats` or `downbeats`, and <model_name> is the name of your model. You can skip the model name argument if you just want to select the model with the highest validation accuracy in the `./models/` directory.