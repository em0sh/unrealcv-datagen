# Timing for benchmarking
import time
tNow = time.time()

# Project libraries
import client
import plot

# Standard libraries

# Libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

import numpy as np


# Device for training Pytorch network on
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

######################## Variables
xRes = client.xRes
yRes = client.yRes

######################## Hyperparameters

learning_rate = 5e-4
batch_size = 5
epochs = 15


######################## Network Definition
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(xRes*yRes, 60),
			nn.ReLU(),
			nn.Linear(60, 60),
			nn.ReLU(),
			nn.Linear(60, xRes*yRes),
		)

	def forward(self, x):
		logits = self.linear_relu_stack(x)
		return(logits)

######################## Datasets
# datasets with load module while converting to numpy array
trainDataX = np.array(client.trainDataX)
trainDataY = np.array(client.trainDataY)
testDataX = np.array(client.testDataX)
testDataY = np.array(client.testDataY)

# Convert arrayed datasets from numpy arrays to tensors while initializing against target device
train_x = torch.tensor(trainDataX, dtype=torch.float, device=device)
train_y = torch.tensor(trainDataY, dtype=torch.float, device=device)

test_x = torch.tensor(testDataX, dtype=torch.float, device=device)
test_y = torch.tensor(testDataY, dtype=torch.float, device=device)


class UCV(Dataset):
	def __init__(self):
		self.len = len(trainDataX)
		self.x = train_x
		self.y = train_y
	
	def __getitem__(self, index):
		return self.x[index], self.y[index]
	
	def __len__(self):
		return self.len

class UCV_test(Dataset):
	def __init__(self):
		self.len = len(testDataX)
		self.x = test_x
		self.y = test_y
	
	def __getitem__(self, index):
		return self.x[index], self.y[index]
	
	def __len__(self):
		return self.len

mnds = UCV()
mnds_test = UCV_test()

train_loader = DataLoader(dataset=mnds, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset=mnds_test, batch_size = batch_size, shuffle = True)


######################## Training
# Create an instance of NeuralNetwork and move it to device
model = NeuralNetwork().to(device)


def train_loop(model, loss_fn, optimizer):
	
	size = len(mnds.x)

	for item, data in enumerate(train_loader):

		inputs, labels = data


		# Compute the prediction, and the loss
		pred		=	model(inputs)
		loss		=	loss_fn(pred, labels)


		# Backpropogation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


		if item % 1000 == 0:
			# DIAG: Don't totally understand what's going on here. Investigate further
			loss, current = loss.item(), item * len(inputs)
			print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test_loop(x, y, model, loss_fn):

	size = len(x)
	# DIAG: Think this is wrong
	num_batches = len(y)

	test_loss, correct = 0, 0

	with torch.no_grad():
		for X in x:

			Y = trainTensorY[batch]

			pred = model(X)
			test_loss += loss_fn(pred, Y). item()
			correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

			test_loss /= num_batches
			correct /= size

			print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')
			
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
	print(f'Epoch {t+1}\n-----------------------------------')
	train_loop(model, loss_fn, optimizer)

with torch.no_grad():
	
	for inp, label in test_loader:
		# Compute the prediction, and the loss
		pred		=	model(inp)

		if client.plotenable == True:
			plot.showFlat(pred, xRes)
		errArray = torch.subtract(pred, label)
		errArray = torch.abs(errArray)
		
		errSum = torch.sum(errArray)
		errSum = float(errSum)
		print(errSum)




# Timing for benchmarking
tThen = time.time()
print(f'time delta was: {tThen - tNow}')
