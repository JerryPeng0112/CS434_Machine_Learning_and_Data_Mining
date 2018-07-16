from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import cPickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCHS = 20
BATCH_SIZE = 10
RATE = [0.1, 0.01, 0.001, 0.0001]
DATA_DIR = "./data/cifar-10-batches-py/"

# Variable for optimizing network for problem 3
OPTI_LEARN_RATE = 0.01
DROP_RATE = 0
MOMENTUM = 0.3
WEIGHT_DECAY = 0

# Variable for optimizing network for problem 4
OPTI_LEARN_RATE2 = 0.01
DROP_RATE2 = 0
MOMENTUM2 = 0.2
WEIGHT_DECAY2 = 0


def main():

    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

    data, testData = load_cifar10_data()
    trainData = data[:1000]
    validateData = data[1000:]

    # Training functions
    #sigmoidTrain(cuda, trainData, validateData, testData)
    #reluTrain(cuda, trainData, validateData, testData)
    optimizedReluTrain(cuda, trainData, validateData, testData)
    threeLayerReluTrain(cuda, trainData, validateData)


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x), dim=1)


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x), dim=1)


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 100)
        self.fc1_drop = nn.Dropout(0)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x), dim=1)


class Net4(nn.Module):

    def __init__(self):
        super(Net4, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 50)
        self.fc1_drop = nn.Dropout(0)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)


# Problem 1
def sigmoidTrain(cuda, trainData, validateData, testData):

    # Test all learning rates to decide which is better
    allLossv, allAccv, testLossv, testAccv = [], [], [], []

    for lr in RATE:
        model = Net1()
        if cuda:
            model.cuda()

        print("\n===> Model: Sigmoid activation function for hidden layer")
        print("Learning rate: " + str(lr))
        print(model)

        lossv, accv = [], []
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

        for epoch in range(1, EPOCHS + 1):
            train(model, cuda, optimizer, trainData, epoch)
            validate(model, cuda, validateData, lossv, accv)

        allLossv.append(lossv)
        allAccv.append(accv)

        # Validate on testing data
        validate(model, cuda, testData, testLossv, testAccv)

    # Plot result from all learning rates
    title = "2-layer network with sigmoid activation function for hidden layer"
    lossFilename = "sigmoidLoss.png"
    accFilename = "sigmoidAcc.png"
    label = "learning rate: "
    plotResult(title, lossFilename, accFilename, allLossv, allAccv, label, RATE)

    print testLossv
    print testAccv


# Problem 2
def reluTrain(cuda, trainData, validateData, testData):

    # Test all learning rates to decide which is better
    allLossv, allAccv, testLossv, testAccv = [], [], [], []

    for lr in RATE:
        model = Net2()
        if cuda:
            model.cuda()

        print("\n===> Model: Relu activation function for hidden layer")
        print("Learning rate: " + str(0.01))
        print(model)

        lossv, accv = [], []
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

        for epoch in range(1, EPOCHS + 1):
            train(model, cuda, optimizer, trainData, epoch)
            validate(model, cuda, validateData, lossv, accv)

        allLossv.append(lossv)
        allAccv.append(accv)

        # Validate on testing data
        validate(model, cuda, testData, testLossv, testAccv)

    # Plot result from all learning rates
    title = "2-layer network with relu activation function for hidden layer"
    lossFilename = "reluLoss.png"
    accFilename = "reluAcc.png"
    label = "learning rate: "
    plotResult(title, lossFilename, accFilename, allLossv, allAccv, label, RATE)

    print testLossv
    print testAccv


# Problem 3
def optimizedReluTrain(cuda, trainData, validateData, testData):

    allLossv, allAccv, testLoss, testAccu= [], [], [], []

    model = Net3()
    if cuda:
        model.cuda()

    print("\n===> Model: Optimized Relu activation function for hidden layer")
    print("Learning rate: " + str(OPTI_LEARN_RATE))
    print("Drop rate: " + str(DROP_RATE))
    print("Momentum: " + str(MOMENTUM))
    print("Weight decay: " + str(WEIGHT_DECAY))
    print(model)

    lossv, accv = [], []
    optimizer = optim.SGD(model.parameters(), lr=OPTI_LEARN_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        train(model, cuda, optimizer, trainData, epoch)
        validate(model, cuda, validateData, lossv, accv)

    allLossv.append(lossv)
    allAccv.append(accv)

    validate(model, cuda, testData, testLoss, testAccu)

    print testAccu


# Problem 4
def threeLayerReluTrain(cuda, trainData, testData):

    # Test all learning rates to decide which is better
    allLossv, allAccv, testLoss, testAccu = [], [], [], []

    model = Net4()
    if cuda:
        model.cuda()

    print("\n===> Model: Relu activation function for hidden layer")
    print("Learning rate: " + str(OPTI_LEARN_RATE2))
    print("Drop rate: " + str(DROP_RATE2))
    print("Momentum: " + str(MOMENTUM2))
    print("Weight decay: " + str(WEIGHT_DECAY2))
    print(model)

    lossv, accv = [], []
    optimizer = optim.SGD(model.parameters(), lr=OPTI_LEARN_RATE2, momentum=MOMENTUM2)

    for epoch in range(1, EPOCHS + 1):
        train(model, cuda, optimizer, trainData, epoch)
        validate(model, cuda, testData, lossv, accv)

    allLossv.append(lossv)
    allAccv.append(accv)

    # Validate on testing data
    validate(model, cuda, testData, testLoss, testAccu)

    print(testAccu)


def train(model, cuda, optimizer, trainData, epoch, log_interval=100):

    model.train()

    for idx, (data, target) in enumerate(trainData):

        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        # Training
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Print training status
        if idx %log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx, len(trainData), 100 * idx / len(trainData), loss.data.item()
            ))


def validate(model, cuda, testData, loss_vector, accuracy_vector):

    model.eval()
    val_loss, correct = 0, 0

    for data, target in testData:

        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        # Validate
        output = model(data)
        val_loss += F.nll_loss(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(testData)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct / (len(testData) * BATCH_SIZE)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(testData) * BATCH_SIZE, accuracy
    ))


def plotResult(title, lossFilename, accFilename, allLossv, allAccv, label, labelRange):

    epochRange = range(1, EPOCHS + 1)

    # Plot Average loss per epoch
    axes = plt.gca()
    axes.set_ylim([1, 3])
    for idx, val in enumerate(labelRange):
        plt.plot(epochRange, allLossv[idx], marker='o', label=(label + str(val)))
    plt.xlabel("Epoch #")
    plt.ylabel("Average loss")
    plt.title("Average loss: " + title)
    plt.legend()
    plt.savefig(lossFilename)
    plt.clf()

    # Plot accuracy per epoch
    axes = plt.gca()
    axes.set_ylim([0, 120])
    for idx, val in enumerate(labelRange):
        plt.plot(epochRange, allAccv[idx], marker='o', label=(label + str(val)))
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.title("Accuracy: " + title)
    plt.legend()
    plt.savefig(accFilename)
    plt.clf()

def plotTestResult(title, accFilename, allAccv, labelRange):
    plt.plot(labelRange, allAccv, marker='o')
    plt.xlabel("momentum")
    plt.ylabel("Accuracy")
    plt.title("Accuracy: " + title)
    plt.savefig(accFilename)
    plt.clf()

def load_cifar10_data():

    dict, data, testData, batchX, batchY = {}, [], [], [], []

    # Get training data with normalization
    for i in range(1, 6):
        dict = unpickle(DATA_DIR + "data_batch_{}".format(i))

        for idx in range(len(dict['data'])):
            # Create temp batch
            batchX.append(dict['data'][idx] / 255)
            batchY.append(dict['labels'][idx])

            if (idx + 1) % BATCH_SIZE == 0:

                # Create X and Y tuple and add to data
                x = np.array(batchX).reshape((BATCH_SIZE, 3, 32, 32))
                y = np.array(batchY).reshape((BATCH_SIZE,))
                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).long()

                tuple = (x, y)
                data.append(tuple)

                # Reset batch
                batchX, batchY = [], []

    # Get testing data with normalization
    dict = unpickle(DATA_DIR + "test_batch")

    for idx in range(len(dict['data'])):
        # Create temp batch
        batchX.append(dict['data'][idx] / 255)
        batchY.append(dict['labels'][idx])

        if (idx + 1) % BATCH_SIZE == 0:

            # Create X and Y tuple and add to data
            x = np.array(batchX).reshape((BATCH_SIZE, 3, 32, 32))
            y = np.array(batchY).reshape((BATCH_SIZE,))
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).long()

            tuple = (x, y)
            testData.append(tuple)

            # Reset batch
            batchX, batchY = [], []

    return data, testData


def unpickle(path):
    with open(path, 'rb') as f:
        dict = cPickle.load(f)
    return dict


if __name__ == "__main__":
	main()
