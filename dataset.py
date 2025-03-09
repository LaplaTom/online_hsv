# -*- coding: utf-8 -*-
#import cv2
import os
from random import sample

files_path = r'/home/linchaoqun/idn/lcq/cache/test/'
all_data = []

def namename(i):
    return files_path + f'{i}.jpg'
for i in range(1,6):
    for j in range(6,13):
        all_data.append(namename(i)+" "+namename(j)+" 1")



# num_data = len(all_data)
# train_num = int(num_data*0.7)

print(len(all_data))

# train_data = sample(all_data, train_num)
# test_data = set(all_data) - set(train_data)

train_data_path = r'test.txt'
file2 = open(train_data_path,'w+')
for aaa in all_data:
    file2.write(aaa+"\n")
file2.close()
