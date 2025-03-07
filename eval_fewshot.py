import torch
from torch import nn
import random
import numpy as np
from datasets.data import *
from models.dgcnn import DGCNN, ResNet
import argparse
import sklearn
from sklearn.svm import SVC
import glob
import torch.nn.functional as F
from tqdm import tqdm
import yaml

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--num_points', type=int, default=1024,
                    help='num of points to use')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=5, metavar='N',
                        help='Num of nearest neighbors to use')
parser.add_argument('--n_runs', type=int, default=10,
                        help='Num of few-shot runs')
parser.add_argument('--n_query', type=int, default=20,
                        help='Num of query samples in one class')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
args = parser.parse_args()

device = torch.device("cuda")

#Try to load models
model = DGCNN(args).to(device)

model.load_state_dict(torch.load(args.model_path, weights_only=True))
model.inv_head = nn.Identity()
print("Model Loaded !!")

n_q = args.n_query
k_ways = [5, 10]
m_shots = [10, 20]
datasets = ['modelnet40', 'scanobjectnn']
accuracies = {k:{} for k in datasets}
with tqdm(total=len(k_ways) * len(m_shots) * len(datasets) * args.n_runs) as pbar:
    for dataset in datasets:
        if dataset == 'modelnet40':
            # ModelNet40 - Few Shot Learning

            data_train, label_train = load_modelnet_data('train')
            data_test, label_test = load_modelnet_data('test')
            n_cls = 40
            
        elif dataset == 'scanobjectnn':
            # ScanObjectNN - Few Shot Learning

            data_train, label_train = load_ScanObjectNN('train')
            data_test, label_test = load_ScanObjectNN('test')
            n_cls = 15

        label_idx = {}
        for key in range(n_cls):
            label_idx[key] = []
            for i, label in enumerate(label_train):
                # if label[0] == key:
                if label == key:
                    label_idx[key].append(i)

        for k in k_ways:
            for m in m_shots:
                acc = []
                job_nm=f"Dataset: {dataset}, K: {k}, M: {m}"
                pbar.set_description(job_nm)
                for run in range(args.n_runs):
                    pbar.update(1)
                    k_way = random.sample(range(n_cls), k)

                    data_support = [] ; label_support = [] ; data_query = [] ; label_query = []
                    for i, class_id in enumerate(k_way):
                        support_id = random.sample(label_idx[class_id], m)
                        query_id = random.sample(list(set(label_idx[class_id]) - set(support_id)), n_q)

                        pc_support_id = data_train[support_id]
                        pc_query_id = data_train[query_id]
                        data_support.append(pc_support_id)
                        label_support.append(i * np.ones(m))
                        data_query.append(pc_query_id)
                        label_query.append(i * np.ones(n_q))

                    data_support = np.concatenate(data_support)
                    label_support = np.concatenate(label_support)
                    data_query = np.concatenate(data_query)
                    label_query = np.concatenate(label_query)

                    feats_train = []
                    labels_train = []
                    model = model.eval()

                    for i in range(k * m):
                        data = torch.from_numpy(np.expand_dims(data_support[i], axis = 0))
                        label = int(label_support[i])
                        data = data.permute(0, 2, 1).to(device)
                        data = torch.cat((data, data))
                        with torch.no_grad():
                            feat = model(data)[2][0, :]
                        feat = feat.detach().cpu().numpy().tolist()
                        feats_train.append(feat)
                        labels_train.append(label)

                    feats_train = np.array(feats_train)
                    labels_train = np.array(labels_train)

                    feats_test = []
                    labels_test = []

                    for i in range(k * n_q):
                        data = torch.from_numpy(np.expand_dims(data_query[i], axis = 0))
                        label = int(label_query[i])
                        data = data.permute(0, 2, 1).to(device)
                        data = torch.cat((data, data))
                        with torch.no_grad():
                            feat = model(data)[2][0, :]
                        feat = feat.detach().cpu().numpy().tolist()
                        feats_test.append(feat)
                        labels_test.append(label)

                    feats_test = np.array(feats_test)
                    labels_test = np.array(labels_test)
                    
                    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

                    # scaler = MinMaxScaler()
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(feats_train)
                    model_tl = SVC(kernel ='linear')
                    model_tl.fit(scaled, labels_train)
                    # model_tl.fit(feats_train, labels_train)
                    
                    test_scaled = scaler.transform(feats_test)
                    
                    # accuracy = model_tl.score(feats_test, labels_test) * 100
                    accuracy = model_tl.score(test_scaled, labels_test) * 100
                    acc.append(accuracy)

                    # print(f"C = {c} : {model_tl.score(test_scaled, labels_test)}")
                    # print(f"Run - {run + 1} : {accuracy}")

                accuracies[dataset][f"{k}-way {m}-shot"] = [np.mean(acc).item(), np.std(acc).item()]


print(yaml.dump(accuracies, indent=4))
save_path = f"results/{args.model_path.split('/')[-1].split('.')[0]}/fewshot.yaml"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w') as file:
    yaml.dump(accuracies, file, indent=4)
print(f"Results saved at {save_path}")