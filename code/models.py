import torch
from torch import nn
import torch.nn.functional as F


# model for One-hot encoding feature representation
class Onehot(nn.Module):
	def __init__(self):
		super(Onehot, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size = (4, 8), stride = 1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size = (1, 8), stride = 1)

		self.dropout = nn.Dropout(0.5)

		self.fc1 = nn.Linear(1408, 256)
		self.fc2 = nn.Linear(256, 64)
		self.fc3 = nn.Linear(64, 1)


	def forward(self, onehot):
		
		onehot = F.max_pool2d(F.relu(self.conv1(onehot)), (1, 2))
		
		onehot = F.max_pool2d(F.relu(self.conv2(onehot)), (1, 2))
		
		onehot = onehot.view(-1, 1408)
	
		onehot = F.relu(self.dropout(self.fc1(onehot)))

		onehot = F.relu(self.dropout(self.fc2(onehot)))

		onehot = F.sigmoid(self.fc3(onehot))

		return onehot

# model for the basic set of DNA shape features
class Shape4(nn.Module):
	def __init__(self):
		super(Shape4, self).__init__()

		self.fc1 = nn.Linear(786, 128)
		self.fc2 = nn.Linear(128, 32)
		self.fc3 = nn.Linear(32, 1)
		
		self.dropout = nn.Dropout(0.5)

	def forward(self, shape):
		
		shape = F.relu(self.dropout(self.fc1(shape)))

		shape = F.relu(self.dropout(self.fc2(shape)))

		shape = F.sigmoid(self.fc3(shape))

		return shape

# model for the full set of DNA shape features
class Shape13(nn.Module):
	def __init__(self):
		super(Shape13, self).__init__()

		self.fc1 = nn.Linear(2554, 256)
		self.fc2 = nn.Linear(256, 64)
		self.fc3 = nn.Linear(64, 1)
		
		self.dropout = nn.Dropout(0.5)


	def forward(self, shape):
		
		shape = F.relu(self.dropout(self.fc1(shape)))

		shape = F.relu(self.dropout(self.fc2(shape)))

		shape = F.sigmoid(self.fc3(shape))

		return shape

# model for the full set of DNA shape features
class Shape(nn.Module):
	def __init__(self):
		super(Shape, self).__init__()

		self.fc1 = nn.Linear(2554, 256)
		self.fc2 = nn.Linear(256, 64)
		self.fc3 = nn.Linear(64, 1)
		
		self.dropout = nn.Dropout(0.5)


	def forward(self, shape):
		
		shape = F.relu(self.dropout(self.fc1(shape)))

		shape = F.relu(self.dropout(self.fc2(shape)))

		shape = F.sigmoid(self.fc3(shape))

		return shape

# model for kmer feature representation
class Kmer(nn.Module):
	def __init__(self):
		super(Kmer, self).__init__()

		self.fc1 = nn.Linear(1364, 256)
		self.fc2 = nn.Linear(256, 64)
		self.fc3 = nn.Linear(64, 1)
		
		self.dropout = nn.Dropout(0.5)

	def forward(self, kmer):

		kmer = F.relu(self.dropout(self.fc1(kmer)))

		kmer = F.relu(self.dropout(self.fc2(kmer)))

		kmer = F.sigmoid(self.fc3(kmer))

		return kmer
		
# model for one-hot and DNA shape feature representations
class Onehot_Shape(nn.Module):
	def __init__(self):
		super(Onehot_Shape, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size = (4, 8), stride = 1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size = (1, 8), stride = 1)

		self.dropout = nn.Dropout(0.5)

		self.fc1 = nn.Linear(3962, 512)
		self.fc2 = nn.Linear(512, 64)
		self.fc3 = nn.Linear(64, 1)

	def forward(self, onehot, shape):
		
		onehot = F.max_pool2d(F.relu(self.conv1(onehot)), (1, 2))
		
		onehot = F.max_pool2d(F.relu(self.conv2(onehot)), (1, 2))

		onehot = onehot.view(-1, 1408)

		combine = torch.cat((onehot, shape), dim = 1)
		combine = F.relu(self.dropout(self.fc1(combine)))
		combine = F.relu(self.dropout(self.fc2(combine)))
		combine = F.sigmoid(self.fc3(combine))

		return combine

# model for one-hot and kmer feature representations
class Onehot_Kmer(nn.Module):
	def __init__(self):
		super(Onehot_Kmer, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size = (4, 8), stride = 1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size = (1, 8), stride = 1)

		self.dropout = nn.Dropout(0.5)

		self.fc1 = nn.Linear(2772, 512)
		self.fc2 = nn.Linear(512, 64)
		self.fc3 = nn.Linear(64, 1)

	def forward(self, onehot, kmer):
		
		onehot = F.max_pool2d(F.relu(self.conv1(onehot)), (1, 2))
		
		onehot = F.max_pool2d(F.relu(self.conv2(onehot)), (1, 2))

		onehot = onehot.view(-1, 1408)

		combine = torch.cat((onehot, kmer), dim = 1)
		combine = F.relu(self.dropout(self.fc1(combine)))
		combine = F.relu(self.dropout(self.fc2(combine)))
		combine = F.sigmoid(self.fc3(combine))

		return combine

# model for DNA shape and kmer feature representations
class Shape_Kmer(nn.Module):
	def __init__(self):
		super(Shape_Kmer, self).__init__()
		
		self.dropout = nn.Dropout(0.5)

		self.fc1 = nn.Linear(3918, 512)
		self.fc2 = nn.Linear(512, 64)
		self.fc3 = nn.Linear(64, 1)

	def forward(self, shape, kmer):

		combine = torch.cat((shape, kmer), dim = 1)
		combine = F.relu(self.dropout(self.fc1(combine)))
		combine = F.relu(self.dropout(self.fc2(combine)))
		combine = F.sigmoid(self.fc3(combine))

		return combine

# model for all the 3 feature representations
class Onehot_Shape_Kmer(nn.Module):
	def __init__(self):
		super(Onehot_Shape_Kmer, self).__init__()

		self.conv1 = nn.Conv2d(1, 16, kernel_size = (4, 8), stride = 1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size = (1, 8), stride = 1)

		self.dropout = nn.Dropout(0.5)

		self.fc1 = nn.Linear(5326, 512)
		self.fc2 = nn.Linear(512, 64)
		self.fc3 = nn.Linear(64, 1)
		

	def forward(self, onehot, shape, kmer):

		onehot = F.max_pool2d(F.relu(self.conv1(onehot)), (1, 2))

		onehot = F.max_pool2d(F.relu(self.conv2(onehot)), (1, 2))

		onehot = onehot.view(-1, 1408)


		combine = torch.cat((onehot, shape, kmer), dim = 1)

		combine = F.relu(self.dropout(self.fc1(combine)))
		combine = F.relu(self.dropout(self.fc2(combine)))
		combine = F.sigmoid(self.fc3(combine))

		return combine