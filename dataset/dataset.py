from torch.utils import data
import torch
import cv2
import numpy as np

class dataset(data.Dataset):
    def __init__(self, root='/home/linchaoqun/idn/IDN-master/CEDAR/', classification="train"):
        super(dataset, self).__init__()
        if classification == "train":
            path = root + 'gray_train.txt'
        elif classification == "test":
            path = root + 'gray_test.txt'
        elif classification == "attacktest":
            path = root + 'attack_test.txt'
        elif classification == "MSDS-ChS-train":
            path = '/home/linchaoqun/dataset/MSDS_jpg/data_online2offline.txt'
            root=""
        elif classification == "MSDS-ChS-test":
            path = '/home/linchaoqun/dataset/MSDS_jpg/data_online2offline-381-402.txt'
            root=""
        # elif classification == "Bengali_train":
        #     # path = '/home/linchaoqun/dataset/BHSig260/Bengali_resize/xxy_276_train.txt'
        #     # root = ""
        #     path = '/home/linchaoqun/project/fuji37450_IDN/dataset/BHSig260/Bengali_resize/train_pairs.txt'
        #     root="/home/linchaoqun/project/fuji37450_IDN/dataset/BHSig260/Bengali_resize/"
        # elif classification == "Bengali_test":
        #     # path = '/home/linchaoqun/dataset/BHSig260/Bengali_resize/xxy_276_test.txt'
        #     # root = ""
        #     path = '/home/linchaoqun/project/fuji37450_IDN/dataset/BHSig260/Bengali_resize/test_pairs.txt'
        #     root="/home/linchaoqun/project/fuji37450_IDN/dataset/BHSig260/Bengali_resize/"

        else:
            print("error.")
            exit()
        
        with open(path, 'r') as f:
            lines = f.readlines()

        self.labels = []
        self.datas = []
        self.line = []
        for line in lines:
            refer, test, label = line.split()
            #print(root + refer)
            refer_img = cv2.imread(root + refer, 0)
            test_img = cv2.imread(root + test, 0)

            # refer_img = cv2.resize(refer_img, (220, 155))
            # test_img = cv2.resize(test_img, (220, 155)) 
          #  print(refer_img.shape)
            
            refer_img = 255 - refer_img
            test_img = 255 - test_img

      #      cv2.imwrite("test.jpg",test_img)

            refer_img = refer_img.reshape(-1, refer_img.shape[0], refer_img.shape[1])
            test_img = test_img.reshape(-1, test_img.shape[0], test_img.shape[1])

            refer_test = np.concatenate((refer_img, test_img), axis=0)
       #     print("refer_test:",refer_test)
            self.datas.append(refer_test)
            self.labels.append(int(label))
            self.line.append(line)

        # print(self.datas[0].shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.FloatTensor(self.datas[index]), float(self.labels[index]),self.line[index]

# img = cv2.imread('dataset/original_2_9.png')
# print(img.shape)