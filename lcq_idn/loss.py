import torch
import torch.nn as nn

class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.mse_less = nn.MSELoss()

    # def forward(self, x, y, z, label):
    #     alpha_1, alpha_2, alpha_3 = 0.3, 0.4, 0.3
    #     label = label.view(-1, 1)
    #     # print(max(x), max(label))
    #     loss_1 = self.bce_loss(x, label)
    #     loss_2 = self.bce_loss(y, label)
    #     loss_3 = self.bce_loss(z, label)
    #     return torch.mean(alpha_1*loss_1 + alpha_2*loss_2 + alpha_3*loss_3)
    def forward(self,predicted,label):
        # label = torch.zeros_like(la)
        # label = la.clone()
        label = label.view(-1, 1)
        # label[label == 0] = -1


        # losss = torch.max(torch.zeros_like(predicted), 1 - label * predicted)

        # del label 

      #  predicted = predicted.view(-1, 1)
        # print('predicted2222',predicted)
        # print('label111',label)
        # print('losss',losss)
        return self.bce_loss(predicted,label)
        #return self.mse_less(predicted,label)
        # return torch.mean(losss, dtype=torch.float)

# def loss(x, y, z, label):
#     bce_loss = nn.BCELoss()
#     alpha_1, alpha_2, alpha_3 = 1, 1, 1
#     loss_1 = self.bce_loss(x, label)
#     loss_2 = self.bce_loss(y, label)
#     loss_3 = self.bce_loss(z, label)
#     return torch.mean(torch.add(torch.add(torch.mul(alpha_1, loss_1), torch.mul(alpha_2, loss_2)), torch.mul(alpha_3, loss_3)))
