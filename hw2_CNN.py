import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random,time
import csv

batch_size = 64
test_batch_size = 64
epochs = 20

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using Device ",device)

print("Loading Data")

def cifar_loaders(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./', train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

#             0       1       2      3       4      5       6       7        8       9
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


print("Beginning training")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,64,3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128,256,3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

file = open('temp.csv', 'w')
writer = csv.writer(file)

csv_header=['no_data', 'epoch', 'epochs', 'avg_loss_epoch']

writer.writerow(csv_header)

no_data = 0
for epoch in range(epochs):  # loop over the dataset multiple times

    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        no_data += 1
        if no_data % 64 == 63:    # print every 64 mini-batches
            print(f'[{epoch + 1}, {no_data + 1:5d}] loss: {running_loss / 64:.3f}')
            # epoch_data = [epoch+1, no_data, running_loss/64]
            # writer.writerow(epoch_data)
            running_loss = 0.0
    
        total_batches += 1
        batch_loss += loss.item()
        avg_loss_epoch = batch_loss/total_batches
    
    print ('Total step {}, Epoch {}/{}, Average Loss: {:.4f}'.format(no_data, epoch+1, epochs, avg_loss_epoch))
    avg_loss_data = [no_data, epoch+1, epochs, avg_loss_epoch]
    writer.writerow(avg_loss_data)

print('Finished Training')

PATH = './cifar_net_CNN.pth'
torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Total accuracy of the network on the test images: {100 * correct // total} %')
accuracy_data = [ '', 'Total accuracy: ', 100 * correct // total]
writer.writerow(accuracy_data)

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    accuracy_data = [ classname, 'class accuracy: ', accuracy]
    writer.writerow(accuracy_data)
file.close()