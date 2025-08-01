import numpy as np
from scipy.sparse import csr_matrix
import torch
import tqdm
import sys

ground_truth_file = 'wiki500k/test_labels.txt'
print('reading ground truth label file')
#Reading label file
test_labels = []
fp = open(ground_truth_file)
for line in fp:
    test_labels.append([int(x) for x in line.split()])
    
print('number of training points:',len(test_labels))


prediction_file = 'wiki500k/tst_score_mat.npz'
print('reading prediction file')
#reading the prediction file
prediction_clf = np.load(prediction_file)
print(prediction_clf['shape'],prediction_clf['format'])
indices = prediction_clf['indices']
indptr = prediction_clf['indptr']
data = prediction_clf['data']
prediction = csr_matrix((data, indices, indptr), shape=prediction_clf['shape'])

assert len(test_labels)== prediction_clf['shape'][0], " Number of datapoints mismatch in ground truth and prediction files"
topk=100
pred_scores = []
pred_labels = []
pred_ids = []
labels = []

print('creating values.....')
for i in tqdm.tqdm(range(len(indptr)-1)):
    scores,ind = torch.topk(torch.tensor(data[indptr[i]:indptr[i+1]]),k=topk)
    p_ids = torch.tensor(indices[indptr[i]:indptr[i+1]][ind]).type(torch.int64)
    #print(p_ids)
    gt_labels = torch.zeros(prediction_clf['shape'][1]).scatter(0,torch.tensor(test_labels[i]),torch.tensor([1.0 for i in test_labels[i]]))
    
    pred_scores.append(scores)
    pred_ids.append(p_ids)
    pred_labels.append(torch.gather(gt_labels,0,p_ids))
    labels += [(i,x) for x in test_labels[i]]
    
pred_scores = torch.stack(pred_scores)
pred_labels = torch.stack(pred_labels)
pred_ids = torch.stack(pred_ids)
labels = np.array(labels)

np.savez('elias_wiki500k_test.npz', scores=pred_scores.numpy(),true_positives=pred_labels.numpy(),prediction_ids=pred_ids.numpy(),labels=labels)

#np.savez(npz_fname, scores=pred_scores.numpy(),true_positives=pred_labels.numpy(),prediction_ids=pred_ids.numpy(),labels=labels)