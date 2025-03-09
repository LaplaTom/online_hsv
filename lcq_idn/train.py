import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pynvml import *
import os
from tensorboardX import SummaryWriter
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

def compute_accuracy(pre, labels):
    # print('predicted',pre)
    # print('labels',labels)

    predicted = 0
    predicted = pre
    predicted[predicted > 0.5] = 1
    predicted[predicted <= 0.5] = 0

    predicted = predicted.view(-1)
    accuracy = torch.sum(predicted == labels).item() / labels.size()[0]
    return accuracy


BATCH_SIZE = 32
EPOCHS = 666
LEARNING_RATE = 0.0001

np.random.seed(2023)
torch.manual_seed(1)

cuda = torch.cuda.is_available()

train_set = dataset(classification="MSDS-ChS-train")
test_set = dataset(classification="MSDS-ChS-test")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2*BATCH_SIZE, shuffle=False, drop_last = True)

model = net()
if cuda:
    model = model.cuda()

print(model.parameters)
criterion = loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=0.01)
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

writer = SummaryWriter(log_dir='scalar')

if cuda:
    criterion = criterion.cuda()
iter_n = 0
best_test_acc = 0
t = time.strftime("%m-%d-%H-%M", time.localtime())
print(len(train_loader))
for epoch in range(1, EPOCHS + 1):
    for i, (inputs, labels,text_line) in enumerate(train_loader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()


        labels = labels.float()
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        predicted = model(inputs)
       # print('predicted1111',predicted)


        loss = criterion(predicted, labels)  
  #     print(loss)

        loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(predicted, labels)

        writer.add_scalar(t+'/train_loss', loss.item(), iter_n)
        writer.add_scalar(t+'/train_accuracy', accuracy, iter_n)
        
        if i % 100 == 0:
           # print(predicted)
            with torch.no_grad():

                accuracys = []
                test_loss = []
                for i_, (inputs_, labels_,text_line_) in enumerate(test_loader):
                    labels_ = labels_.float()
                    if cuda:
                        inputs_, labels_ = inputs_.cuda(), labels_.cuda()
                    predicted_ = model(inputs_)
                    test_loss.append(criterion(predicted_, labels_))
                    accuracys.append(compute_accuracy(predicted_, labels_))
                accuracy_ = sum(accuracys) / len(accuracys)
                loss_ = sum(test_loss) / len(test_loss)
                writer.add_scalar(t+'/test_accuracy', accuracy_, iter_n)
                writer.add_scalar(t+'/test_loss', loss_, iter_n)
            print('test loss:{:.6f},test accuracy:{:.6f}'.format(loss_,accuracy_))
            if accuracy_ > best_test_acc:
                best_test_acc = accuracy_
                torch.save(model.state_dict(), 'model_baidiheizi_'+t+'_'+str(best_test_acc)+'.pth')
                

        if i % 10 == 0:
            print('Epoch[{}/{}], iter {}, loss:{:.6f}, accuracy:{}'.format(epoch, EPOCHS, i, loss.item(), accuracy))
        iter_n += 1

writer.close()
