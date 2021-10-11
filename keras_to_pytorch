
from keras.models import load_model
model1 = load_model(os.path.join(cur_path,'keras_model.h5'))

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		
		self.conv1 = nn.Conv2d(3, 16, 5)
		self.conv2 = nn.Conv2d(16, 32, 3)
		self.fc1 = nn.Linear(800, 256)
		self.fc2 = nn.Linear(256, 43)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x,2)
		x = F.dropout(x)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x,2)
		x = F.dropout(x)
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		x = F.relu(self.fc1(x))
		x = F.dropout(x)
		x = F.Softmax(self.fc2(x))
		return x

net = Net()
print(net)

# Get the pre-trained weights
weights = model1.get_weights()
net.conv1.weight.data=torch.from_numpy(np.transpose(weights[0]))
net.conv1.bias.data=torch.from_numpy(np.transpose(weights[1]))
net.conv2.weight.data=torch.from_numpy(np.transpose(weights[2]))
net.conv2.bias.data=torch.from_numpy(np.transpose(weights[3]))

net.fc1.weight.data=torch.from_numpy(np.transpose(weights[4]))
net.fc1.bias.data=torch.from_numpy(np.transpose(weights[4]))
net.fc2.weight.data=torch.from_numpy(np.transpose(weights[6]))
net.fc2.bias.data=torch.from_numpy(np.transpose(weights[7]))
