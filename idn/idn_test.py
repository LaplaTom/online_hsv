from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pynvml import *
import os
import time

#from dataset.dataset import dataset
from idn.models.net import net
from idn.loss import loss
os.environ['CUDA_VISIBLE_DEVICES']='3'

def compute_accuracy(predicted, labels):
    for i in range(3):
        predicted[i][predicted[i] > 0.5] = 1
        predicted[i][predicted[i] <= 0.5] = 0
    predicted = predicted[0] + predicted[1] + predicted[2]
    
    predicted[predicted < 2] = 0
    predicted[predicted >= 2] = 1
    predicted = predicted.view(-1)
    accuracy = torch.sum(predicted == labels).item() / labels.size()[0]
    return accuracy

def model_run(test_set):
    np.random.seed(0)
    torch.manual_seed(1)

    cuda = torch.cuda.is_available()
    print(cuda)

    #test_set = dataset(classification="MSDS-350-402-test-data")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    model_path = "./idn/MSDS_chn_model_online2offline_0.78125.pth"
    print(model_path)

    model = net()
    model.load_state_dict(torch.load(model_path))
    if cuda:
        model = model.cuda()
    criterion = loss()

    if cuda:
        criterion = criterion.cuda()
    t = time.strftime("%m-%d-%H-%M", time.localtime())
    #print(len(test_loader))
    accuracys = []
    for i_, (inputs_) in enumerate(test_loader):
        if cuda:
            inputs_ = inputs_.cuda()
        predicted_ = model(inputs_)
     #   print(predicted_)
        label_ = torch.FloatTensor(torch.ones(1)).cuda()
        acc = compute_accuracy(predicted_, label_)
        print(acc)
        accuracys.append(acc)
    accuracy_ = sum(accuracys) / len(accuracys)
    return accuracy_
