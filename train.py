from model import model
from dataloader import data_loader
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

trainset, trainloader, testset, testloader = data_loader()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 50
train = []
test = []
start_time = time.time()


for epoch in range(epochs):
    #set running_loss variables
    running_loss = 0
    running_loss_test = 0
    incorrect = []
    print(f'Epoch: {epoch+1}/{epochs}')

    model.train()
    for i, (images, labels) in enumerate(iter(trainloader)):
        # images, labels = images.to(device), labels.to(device)
        images.resize_(images.size()[0], 784)

        optimizer.zero_grad()

        logits = model.forward(images)
        pred = F.softmax(logits, dim=1)
        train_loss = criterion(pred, labels)
        train_loss.backward()
        optimizer.step()

        running_loss += train_loss.item()

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(iter(testloader)):

            images.resize_(images.size()[0], 784)
            test_probs = model(images)
            test_loss = criterion(test_probs, labels)
            _, pred = torch.max(test_probs, 1)
            incorrect_pred = ((pred == labels) == False)#.nonzero()
            running_loss_test += test_loss.item()
            incorrect.append(images[incorrect_pred])


    train.append((running_loss/64))
    test.append((running_loss_test/64))

train_time = time.time() - start_time
print(f'Training time: {train_time}')


torch.save(model.state_dict(), 'checkpoint_50')

plt.plot(train, label='train loss')
plt.plot(test, label='test loss')
plt.legend()
plt.show()