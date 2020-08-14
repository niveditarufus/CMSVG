## Data
The images can be found [here](https://drive.google.com/open?id=1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek). Unzip the images and copy them to the ./data folder. Notice that the ./data folder contains two separate json files. In one case, we list 64 possibly overlapping region proposals extracted by CenterNet. In the latter case, we removed duplicate proposals from the list.

You can run the code as follows.

```
python3 train.py --root ./data --lr 0.01 --nesterov --evaluate 
```

The published code can be used to train a model that obtains +- 65.8% AP50 on the validation set. The training can be done on a system with 4 x 1080ti GPU in just a few hours.

## Submission
A submission file can be created by running the test.py script. The predictions.json file that is delivered with the repository contains the predictions of the pre-trained model. 

```
python3 test.py --root ./data
```

## Requirements

See requirements.txt for a list of packages. Notice that you need to install the English language model for spacy by running the following command from within your environment.

```
python -m spacy download en_core_web_sm
```