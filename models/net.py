import torch
import torch.nn as nn
from lcq_idn.models.stream import stream
from lcq_idn.models.stream import single
#from models.stream import ResNet
import torchvision

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.stream = stream()
        self.single = single()
#        self.resnet = ResNet()
        # self.conv_ref_inv = single()
        # self.conv_test = single()
        # self.conv_test_inv = single()
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            # nn.Conv2d(512, 128, 3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, 256),
            # nn.ReLU(inplace=True),
     #       nn.Dropout2d(0.5),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
           # nn.Dropout2d(0.5),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()
      
        

    def forward(self, inputs):
        half = inputs.size()[1] // 2
        reference = inputs[:, :half, :, :]
        reference_inverse = 255 - reference
        test = inputs[:, half:, :, :]
        del inputs
        test_inverse = 255 - test

        reference, reference_inverse = self.stream(reference, reference_inverse)
        test, test_inverse = self.stream(test, test_inverse)
      #  print("reference",reference.size())  # reference torch.Size([32, 128, 7, 31])

       # print(reference.size())

        # reference = self.conv_ref(reference)
        # reference_inverse = self.conv_ref_inv(reference_inverse)
        # test = self.conv_test(test)
        # test_inverse = self.conv_test_inv(test_inverse)


        # cat_ref = torch.cat((reference, reference_inverse), dim=1)
        # cat_test = torch.cat((test, test_inverse), dim=1)
        cat_image = torch.cat((reference, reference_inverse,test_inverse,test), dim=1)
        #cat_image = torch.cat((reference, test), dim=1)
       # print(cat_ref.size(),cat_test.size(),cat_image.size())

        del reference, reference_inverse, test, test_inverse

        #print(cat_image.size())
       # cat_image = self.resnet(cat_image) 

    #    print("cat_image",cat_image.size())  #cat_image torch.Size([32, 128, 5, 29])
        cat_image = self.single(cat_image) 
        cat_image = self.sub_forward(cat_image)
        # cat_2 = self.sub_forward(cat_2)
        # cat_3 = self.sub_forward(cat_3)

        return cat_image
    
    def sub_forward(self, inputs):
        out = self.GAP(inputs)
        out = out.view(-1, inputs.size()[1])
      #  print(out.size())
        out = self.classifier(out)
      #   out = out.transpose(0,1)
      #   out = torch.sub(out[0],out[1])
        
      #   out = self.sig(out)
      #  # print(out)
        return out

if __name__ == '__main__':
    net = net()
    x = torch.ones(1, 3, 32, 32)
    y = torch.ones(1, 3, 32, 32)
    x_ = torch.ones(1, 3, 32, 32)
    y_ = torch.ones(1, 3, 32, 32)
    out_1, out_2, out_3 = net(x, y, x_, y_)
    # vgg = torchvision.models.vgg13()
    # print(vgg)