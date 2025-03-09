import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pynvml import *
import os
import time

from dataset.dataset import dataset
from models.net import net
from loss import loss
os.environ['CUDA_VISIBLE_DEVICES']='4'

# def compute_accuracy(predicted, labels):
#     for i in range(3):
#         predicted[i][predicted[i] > 0.5] = 1
#         predicted[i][predicted[i] <= 0.5] = 0
#     predicted = predicted[0] + predicted[1] + predicted[2]
    
#     predicted[predicted < 2] = 0
#     predicted[predicted >= 2] = 1
#     predicted = predicted.view(-1)
#     accuracy = torch.sum(predicted == labels).item() / labels.size()[0]
#     return accuracy

def compute_accuracy(pre, labels,text_line_):
    # print('predicted',pre)
    # print('labels',labels)

    predicted = 0
    predicted = pre
    predicted[predicted > 0.5] = 1
    predicted[predicted <= 0.5] = 0

    predicted = predicted.view(-1)
    accuracy = torch.sum(predicted == labels).item() / labels.size()[0]

    #print(predicted)
    test_num = [0,0]
   # print(test_num[0])

    error_line = []

    for i in range(predicted.size()[0]):
        if predicted[i] != labels[i]:
            test_num[labels[i].int()] += 1
            print(text_line_[i])
            error_line.append(text_line_[i])
   # print(test_num[0],test_num[1])
    return accuracy,error_line


np.random.seed(0)
torch.manual_seed(1)

cuda = torch.cuda.is_available()
print(cuda)

test_set = dataset(classification="Bengali_test")
#test_set = dataset(classification="MSDS-ChS-test-online2offline")
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
model_path = "Bengali_model_0.925790313225058.pth"
print(model_path)

model = net()
model.load_state_dict(torch.load(model_path))
if cuda:
    model = model.cuda()
criterion = loss()

if cuda:
    criterion = criterion.cuda()
t = time.strftime("%m-%d-%H-%M", time.localtime())
print(len(test_loader))

accuracys = []
err_lines = []
for i_, (inputs_, labels_,text_line_) in enumerate(test_loader):
    labels_ = labels_.float()
    if cuda:
        inputs_, labels_ = inputs_.cuda(), labels_.cuda()
    predicted_ = model(inputs_)
    #print(predicted_)
   # print(labels_)
    acc,err_line = compute_accuracy(predicted_, labels_,text_line_)
    # if(int(acc)):
    #     print(i_)
    accuracys.append(acc)
    err_lines = err_lines + err_line

print(sum(accuracys),len(accuracys))
accuracy_ = sum(accuracys) / len(accuracys)
print(t+'/test_accuracy', accuracy_)

# with open(f'error_lines.txt', 'w') as f:
#     for line in err_lines:
#         f.write(line)