import torch
import torch.nn as nn
from lcq_idn.models.acmix import ACmix
import cv2

class stream(nn.Module):
	def __init__(self):
		super(stream, self).__init__()

		self.stream = nn.Sequential(
			nn.Conv2d(1, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(64, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			# nn.Conv2d(64, 96, 3, stride=1, padding=1),
			# nn.ReLU(inplace=True),
			# nn.Conv2d(96, 96, 3, stride=1, padding=1),
			# nn.ReLU(inplace=True),
			# nn.MaxPool2d(2, stride=2),

			# nn.Conv2d(96, 128, 3, stride=1, padding=1),
			# nn.ReLU(inplace=True),
			# nn.Conv2d(128, 128, 3, stride=1, padding=1),
			# nn.ReLU(inplace=True),
			# nn.MaxPool2d(2, stride=2)
			)

		self.Conv_32 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
		self.Conv_64 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.Conv_96 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
		self.Conv_128 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

		self.fc_32 = nn.Linear(32, 32)
		self.fc_64 = nn.Linear(64, 64)
		self.fc_96 = nn.Linear(96, 96)
		self.fc_128 = nn.Linear(128, 128)
		
		self.max_pool = nn.MaxPool2d(2, stride=2)

	def forward(self, reference, inverse):
		# print(self.stream[0]) Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		# print(self.stream[1]) ReLU(inplace=True)
		for i in range(2):
			reference = self.stream[0 + i * 5](reference)
			reference = self.stream[1 + i * 5](reference)
			inverse = self.stream[0 + i * 5](inverse)
			inverse = self.stream[1 + i * 5](inverse)
			inverse = self.stream[2 + i * 5](inverse)
			inverse = self.stream[3 + i * 5](inverse)
			inverse = self.stream[4 + i * 5](inverse)
			reference = self.attention(inverse, reference)
			reference = self.stream[2 + i * 5](reference)
			reference = self.stream[3 + i * 5](reference)
			reference = self.stream[4 + i * 5](reference)

		return reference, inverse


	def attention(self, inverse, discrimnative):
	
		GAP = nn.AdaptiveAvgPool2d((1, 1))
		sigmoid = nn.Sigmoid()

		up_sample = nn.functional.interpolate(inverse, (discrimnative.size()[2], discrimnative.size()[3]), mode='nearest')
		conv = getattr(self, 'Conv_' + str(up_sample.size()[1]), 'None')
		g = conv(up_sample)
		tmp = g * discrimnative + discrimnative
		f = GAP(tmp)
		f = f.view(f.size()[0], 1, f.size()[1])

		fc = getattr(self, 'fc_' + str(f.size(2)), 'None')
		f = fc(f)
		f = sigmoid(f)
		f = f.view(-1, f.size()[2], 1, 1)
		out = tmp * f

		return out



class single(nn.Module):
	def __init__(self):
		super(single, self).__init__()
		#(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
		self.acmix = ACmix(512, 128, 3, 2, 3, stride=1, dilation=1)
		# self.stream1 = nn.Sequential(
		# 	nn.Conv2d(512, 256, 3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# 	nn.MaxPool2d(2, stride=1),
		# )
		self.stream2 = nn.Sequential(
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=1),
			)

	def forward(self, image):
	#	out = self.stream1(image)
		out = self.acmix(image)
		out = self.stream2(out)
		return out
