from torch.utils import data
import torch
import cv2
import numpy as np

class dataset(data.Dataset):
    def __init__(self, root='/home/linchaoqun/idn/IDN-master/CEDAR/', classification="train"):
        super(dataset, self).__init__()
        print(classification)
        if classification == "train":
            path = root + 'gray_train.txt'
        elif classification == "test":
            path = root + 'gray_test.txt'
        elif classification == "attacktest":
            path = root + 'attack_test.txt'
        elif classification == "MSDS-ChS-train":
            path = '/home/linchaoqun/dataset/MSDS_jpg/train_data.txt'
            root=""
        elif classification == "MSDS-ChS-test":
            path = '/home/linchaoqun/dataset/MSDS_jpg/test_data.txt'
            root=""
        elif classification == "MSDS-ChS-train-tiny":
            path = '/home/linchaoqun/dataset/MSDS_jpg/train_data_tiny.txt'
            root=""
        elif classification == "MSDS-ChS-test-tiny":
            path = '/home/linchaoqun/dataset/MSDS_jpg/test_data_tiny.txt'
            root=""
        elif classification == "MSDS-ChS-train-tiny-256-192":
            path = '/home/linchaoqun/dataset/MSDS_jpg/train_data_tiny_256_192.txt'
            root=""
        elif classification == "MSDS-ChS-test-tiny-256-192":
            path = '/home/linchaoqun/dataset/MSDS_jpg/test_data_tiny_256_192.txt'
            root=""
        elif classification == "MSDS-ChS-train-tiny-1080-960":
            path = '/home/linchaoqun/dataset/MSDS_jpg/train_data_tiny_1080_960.txt'
            root=""
        elif classification == "MSDS-ChS-test-tiny-1080-960":
            path = '/home/linchaoqun/dataset/MSDS_jpg/test_data_tiny_1080_960.txt'
            root=""
        elif classification == "MSDS-all-data":
            path = '/home/linchaoqun/dataset/MSDS_jpg/data_online2offline.txt'
            root=""
        elif classification == "MSDS-ChS-test-tiny-online2offline":
            path = '/home/linchaoqun/dataset/MSDS_jpg/test_data_tiny_online2offline.txt'
            root=""
        elif classification == "MSDS-ChS-train-tiny-online2offline":
            path = '/home/linchaoqun/dataset/MSDS_jpg/train_data_tiny_online2offline.txt'
            root=""
        elif classification == "MSDS-ChS-test-online2offline":
            path = '/home/linchaoqun/dataset/MSDS_jpg/test_data_online2offline.txt'
            root=""
        elif classification == "MSDS-ChS-train-online2offline":
            path = '/home/linchaoqun/dataset/MSDS_jpg/train_data_online2offline.txt'
            root=""
        elif classification == "MSDS-350-402-test-data":
            path = '/home/linchaoqun/dataset/MSDS_jpg/data_online2offline-381-402.txt'
            root=""
        elif classification == "MSDS-ChS-online2offline":
            path = '/home/linchaoqun/dataset/MSDS_jpg/data_online2offline.txt'
            root=""
        else:
            print("error.")
            exit()
        
        with open(path, 'r') as f:
            lines = f.readlines()

        self.labels = []
        self.datas = []
        for line in lines:
            refer, test, label = line.split()
         #   print(line)
            refer_img = cv2.imread(root + refer, 0)
            test_img = cv2.imread(root + test, 0)

        #    refer_img = cv2.resize(refer_img, (256, 140), interpolation=cv2.INTER_AREA)
        #    test_img = cv2.resize(test_img, (256, 140), interpolation=cv2.INTER_AREA) 
       #     print(refer_img.shape)
           # cv2.imwrite("testtest.jpg",test_img)

            refer_img = refer_img.reshape(-1, refer_img.shape[0], refer_img.shape[1])
            test_img = test_img.reshape(-1, test_img.shape[0], test_img.shape[1])

            refer_test = np.concatenate((refer_img, test_img), axis=0)
       #     print("refer_test:",refer_test)
            self.datas.append(refer_test)
            self.labels.append(int(label))

        # print(self.datas[0].shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.FloatTensor(self.datas[index]), float(self.labels[index])

# img = cv2.imread('dataset/original_2_9.png')
# print(img.shape)