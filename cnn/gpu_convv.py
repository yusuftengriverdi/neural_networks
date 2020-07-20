import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imghdr
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on " + str(device) + ".")
else:
    device = torch.device("cpu")
    print("running on" + str(device) + ".")
# torch.cuda.device_count()

# pre_processing image

REBUILD_DATA = False  # rebuild data once.


class DogsVSCats():
    IMG_SIZE = 50
    # uniform sizes and shapes.
    # augmentation of data
    CATS = "D:\\Neural_Networks\\coursera_v2\\PetImages\\Cat"
    DOGS = "D:\\Neural_Networks\\coursera_v2\\PetImages\\Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    # !!! to check balance:
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    s = (label, f)
                    path = os.path.join(*s)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # is color relevant feature?
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    "np.eye() gives one-hot vector matrices"
                    # one-hot vector method
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                    # we can them to be nearly same values.
                except Exception as e:
                    print(str(e))
                    pass
        np.random.shuffle(self.training_data)
        np.save("training_Data.npy", self.training_data)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)


if REBUILD_DATA:
    dogvcats = DogsVSCats()
    dogvcats.make_training_data()

training_data = np.load("training_Data.npy", allow_pickle=True)


# print(len(training_data))

# 3 dimensional convolutional layers


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # input =1 , output= 32, kernel size=5, 5*5 kernel
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        # distribution of predictions.
        self.fc1 = nn.Linear(self._to_linear, 512)  # to flatten
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # other way
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        # print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


net = Net().to(device)

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X / 255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X) * VAL_PCT)
# print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[:-val_size]
test_y = y[:-val_size]

# print(len(train_X))
# print(len(test_y))


def train(net_):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    BATCH_SIZE = 100  # first thing to modify
    EPOCHS = 3
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            # print(i, i+BATCH_SIZE)
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i + BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            net.zero_grad()

            # optimizer.zero_grad() # zero the gradient buffers
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()  # Does the update

        print(f"Epoch: {epoch}. Loss: {loss}")


test_X.to(device)
test_y.to(device)


def test(net_):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy", round(correct / total, 3))


train(net)
test(net)
