# Cosine meets Softmax: A tough-to-beat baseline for visual grounding  
This repository contains the source code of our solution which came in third in the [C4AV Challenge](https://www.aicrowd.com/challenges/eccv-2020-commands-4-autonomous-vehicles). A technical report detailing our solution can be found [here](https://arxiv.org/abs/2009.06066).  

## Data
The images can be found [here](https://drive.google.com/open?id=1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek). Unzip the images and copy them to the ./data folder. Notice that the ./data folder contains two separate json files. In one case, we list 64 possibly overlapping region proposals extracted by CenterNet. In the latter case, we removed duplicate proposals from the list.

You can run the code as follows.

```
python3 train.py --root ./data --lr 0.01 --nesterov --evaluate 
```

The published code can be used to train a model that obtains +- 65.8% AP50 on the validation set. The training can be done on a system with 4 x 1080ti GPU in just a few hours. You can specify different versions for EfficientNet for the `img_encoder`, you will have to change the image dimensions accordingly(stated below) in dataset.py. However, this will utilise more memory and longer training time for deeper models.(Best performance was obtained by EfficientNet b2)  
1. EfficientNet b0: 224
2. EfficientNet b1: 240
3. EfficientNet b2: 260
4. EfficientNet b3: 300
5. EfficientNet b4: 380
6. EfficientNet b5: 456
7. EfficientNet b6: 528
8. EfficientNet b7: 600  

The `text_encoder` also can be modified to one of the following. The dimensions of the Full-connected layer (`fc_model`) should be modified to satisfy dimesion requirements.(Best performance was obtained by Roberta large)  
1. bert-base-nli-stsb-mean-tokens	
2. bert-large-nli-stsb-mean-tokens	
3. roberta-base-nli-stsb-mean-tokens	
4. roberta-large-nli-stsb-mean-tokens	
5. distilbert-base-nli-stsb-mean-tokens  

## Submission
A submission file can be created by running the test.py script. The predictions.json file that is delivered with the repository contains the predictions of the pre-trained model. 

```
python3 test.py --root ./data
```

See requirements.txt for a list of packages. Notice that you need to install the English language model for spacy by running the following command from within your environment.

```
python -m spacy download en_core_web_sm
```
If you find this work useful in your research work please cite :

```
@InProceedings{10.1007/978-3-030-66096-3_4,
author="Rufus, Nivedita
and Nair, Unni Krishnan R.
and Krishna, K. Madhava
and Gandhi, Vineet",
editor="Bartoli, Adrien
and Fusiello, Andrea",
title="Cosine Meets Softmax: A Tough-to-beat Baseline for Visual Grounding",
booktitle="Computer Vision -- ECCV 2020 Workshops",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="39--50",
abstract="In this paper, we present a simple baseline for visual grounding for autonomous driving which outperforms the state of the art methods, while retaining minimal design choices. Our framework minimizes the cross-entropy loss over the cosine distance between multiple image ROI features with a text embedding (representing the given sentence/phrase). We use pre-trained networks for obtaining the initial embeddings and learn a transformation layer on top of the text embedding. We perform experiments on the Talk2Car dataset and achieve 68.7{\%} AP50 accuracy, improving upon the previous state of the art by 8.6{\%}. Our investigation suggests reconsideration towards more approaches employing sophisticated attention mechanisms or multi-stage reasoning or complex metric learning loss functions by showing promise in simpler alternatives.",
isbn="978-3-030-66096-3"
}
```
