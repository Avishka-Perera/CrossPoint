import argparse
import glob
import random
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random

import yaml
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import Identity
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets.data import *
from datasets.plyfile import load_ply
from models.dgcnn import DGCNN, ResNet, DGCNN_partseg

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
parser.add_argument('--num_points', type=int, default=1024,
                    help='num of points to use')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=15, metavar='N',
                        help='Num of nearest neighbors to use')
parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
args = parser.parse_args()

device = torch.device("cuda")
net_self = torch.load(args.model_path, weights_only=True)
model_self = DGCNN(args).to(device)
model_self.load_state_dict(net_self)
model_self.inv_head = Identity()

modelnet_train_loader = DataLoader(ModelNet40SVM(partition='train', num_points=args.num_points),
                              batch_size=128, shuffle=True)
modelnet_test_loader = DataLoader(ModelNet40SVM(partition='test', num_points=args.num_points),
                              batch_size=128, shuffle=True)
scanobjectnn_train_loader = DataLoader(ScanObjectNNSVM(partition='train', num_points=args.num_points),
                              batch_size=64, shuffle=True)
scanobjectnn_test_loader = DataLoader(ScanObjectNNSVM(partition='test', num_points=args.num_points),
                              batch_size=64, shuffle=True)
# dataset = "ScanObjectNN" # Choose Dataset ["ModelNet40, ScanObjectNN"]
accuracies = {}
with tqdm(total=len(modelnet_train_loader)+len(modelnet_test_loader)+len(scanobjectnn_train_loader)+len(scanobjectnn_test_loader)) as pbar:
    for dataset in ["ModelNet40", "ScanObjectNN"]:
        pbar.set_description(f"Evaluating: {dataset}")

        feats_train = []
        labels_train = []
        model = model_self.to(device)
        model = model.eval()

        if dataset == "ModelNet40":
            train_loader = modelnet_train_loader
            test_loader = modelnet_test_loader
        elif dataset == "ScanObjectNN":
            train_loader = scanobjectnn_train_loader
            test_loader = scanobjectnn_test_loader

        for i, (data, label) in enumerate(train_loader):
            if dataset == "ModelNet40":
                labels = list(map(lambda x: x[0],label.numpy().tolist()))
            elif dataset == "ScanObjectNN":
                labels = label.numpy().tolist()
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = model(data)[2]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels
            pbar.update(1)

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []
        model = model_self.to(device)
        model = model.eval()

        for i, (data, label) in enumerate(test_loader):
            if dataset == "ModelNet40":
                labels = list(map(lambda x: x[0],label.numpy().tolist()))
            elif dataset == "ScanObjectNN":
                labels = label.numpy().tolist()
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = model(data)[2]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            # labels_train.append(label)
            labels_test += labels
            pbar.update(1)

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)

        c = 0.01 # Linear SVM parameter C, can be tuned
        model_tl = SVC(C = c, kernel ='linear')
        model_tl.fit(feats_train, labels_train)
        acc = model_tl.score(feats_test, labels_test)
        accuracies[dataset] = acc

print(yaml.dump(accuracies, indent=4))
save_path = f"results/{args.model_path.split('/')[-1].split('.')[0]}/linear.yaml"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w') as file:
    yaml.dump(accuracies, file, indent=4)
print(f"Results saved at {save_path}")