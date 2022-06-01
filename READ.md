- To download nltk tools run:
```angular2html
python3 nlp/download.py
```
See downloads on home directory with name **nltk_data**.

### Usage
- data_operations.py file:

The **data** directory contains the different categories of questions obtained in the intermediate steps, combined with test and training, and exported to different json files. All jsons are merged in a json file and a csv file is created from this json file. Finally, the test data is added to the csv file.

- train.py file:

Performs the train operation of the bert model and saves the resulting model.

### System Requirements
Download PyAudio for speech recognition **(on Ubuntu)**.

```angular2html
sudo apt-get install portaudio19-dev python3-pyaudio
```
Then install requirements with:

```angular2html
pip3 install -r requirements.txt
```
### Run

Use **flask run** command to start service **(in front directory)**.

```
flask run
```