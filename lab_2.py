import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load the dataset
data = np.load('lab2_dataset.npz')
train_feats = torch.tensor(data['train_feats'])
test_feats = torch.tensor(data['test_feats'])
train_labels = torch.tensor(data['train_labels'])
test_labels = torch.tensor(data['test_labels'])
phone_labels = data['phone_labels']

print(train_feats.shape)
print(test_feats.shape)
print(train_labels.shape)
print(test_labels.shape)
print(phone_labels.shape)

# Set up the dataloaders
train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(test_feats, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)


# Define the model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.linear1 = nn.Linear(440, 800)
        self.linear2 = nn.Linear(800, 400)
        self.linear3 = nn.Linear(400, 200)
        self.linear4 = nn.Linear(200, 100)
        self.linear5 = nn.Linear(100, 50)
        self.linear6 = nn.Linear(50, 48)

        self.dropout = nn.Dropout(0.5)
        # self.linear2 = nn.Linear(400, 300) 
        # self.linear3 = nn.Linear(300, 200) 
        # self.linear4 = nn.Linear(200, 100) 
        # self.linear5 = nn.Linear(100, 50)
        # self.linear6 = nn.Linear(50, 48)

        self.relu = nn.ReLU() # activation function
        
    def forward(self, x):
        x = torch.reshape(x, (-1, 11 * 40))
        #print(x.shape)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.linear6(x)
        
        return x
    
# Instantiate the model, loss function, and optimizer
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_network(model, train_loader, criterion, optimizer):
    # TODO: fill in
    for epoch in range(10):
        # running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # running_loss += loss.item()
            # if i % 1000 == 999:
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            #     running_loss = 0.0
            # 
        test_network(model, test_loader)

def test_network(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Test accuracy: %d %%' % (100 * correct / total))

train_network(model, train_loader, criterion, optimizer)

print('Finished Training')

test_network(model, test_loader)