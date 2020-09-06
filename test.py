import os
import argparse
import json
import shutil
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from dataset import Talk2Car
from utils.util import AverageMeter, ProgressMeter, save_checkpoint
import models.resnet as resnet
from sentence_transformers import SentenceTransformer
from efficientnet_pytorch import EfficientNet


parser = argparse.ArgumentParser(description='Talk2Car object referral')
parser.add_argument('--root', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 18)')

def main():
    args = parser.parse_args()

    # Create dataset
    print("=> creating dataset")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    dataset = Talk2Car(root=args.root, split='test',
                                transform=transforms.Compose([transforms.ToTensor(), normalize]))
    dataloader = data.DataLoader(dataset, batch_size = args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    print('Test set contains %d samples' %(len(dataset)))

    # Create model
    print("=> creating model")
    img_encoder = nn.DataParallel(EfficientNet.from_pretrained('efficientnet-b2'))
    text_encoder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    fc_model = nn.Sequential(nn.Linear(1024, 1000), nn.ReLU(), nn.Linear(1000,1000))

    img_encoder.cuda()
    text_encoder.cuda()
    fc_model.cuda()

    cudnn.benchmark = True

    # Evaluate model
    print("=> Evaluating best model")
    checkpoint = torch.load('best_model.pth.tar', map_location='cpu')
    img_encoder.load_state_dict(checkpoint['img_encoder'])
    fc_model.load_state_dict(checkpoint['fc_model'])
    evaluate(dataloader, img_encoder, text_encoder, fc_model, args)

@torch.no_grad()
def evaluate(val_dataloader, img_encoder, text_encoder, fc_model, args):
    img_encoder.eval()
    fc_model.eval()
    text_encoder.eval()
    
    ignore_index = val_dataloader.dataset.ignore_index
    prediction_dict = {}       
 
    for i, batch in enumerate(val_dataloader):
        
        # Data
        region_proposals = batch['rpn_image'].cuda(non_blocking=True)
        sentence = batch['sentence']
        command = batch['command'].cuda(non_blocking=True)
        command_length = batch['command_length'].cuda(non_blocking=True)

        b, r, c, h, w = region_proposals.size()

        # Image features
        img_features = img_encoder(region_proposals.view(b*r, c, h, w))
        norm = img_features.norm(p=2, dim=1, keepdim=True)
        img_features = img_features.div(norm).view(b, r, -1)
                
        #Sentence features
        sentence_features = torch.from_numpy(np.array(text_encoder.encode(sentence))).cuda(non_blocking=True)
        sentence_features = fc_model(sentence_features)
             
        # Product in latent space
        scores = torch.bmm(img_features, sentence_features.unsqueeze(2)).squeeze()


        pred = torch.argmax(scores, 1)
        
        # Add predictions to dict
        for i_, idx_ in enumerate(batch['index'].tolist()):
            token = val_dataloader.dataset.convert_index_to_command_token(idx_[0])
            bbox = batch['rpn_bbox_lbrt'][i_, pred[i_]].tolist()
            bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
            if token in prediction_dict.keys():
                print('Token already exists')
            prediction_dict[token] = bbox             

    print('Predictions for %d samples saved' %(len(prediction_dict)))
    with open('predictions.json', 'w') as f:
        json.dump(prediction_dict, f)
        
if __name__ == '__main__':
    main()