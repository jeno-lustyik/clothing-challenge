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

epochs = 100
train = []
test = []
start_time = time.time()

for epoch in range(epochs):

    #save checkpoint models in every 10th epoch
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f'checkpoint_{epoch+1}.pth')

    #set running variables
    running_loss = 0
    running_loss_test = 0
    incorrect = []
    correct = 0
    print(f'Epoch: {epoch+1}/{epochs}')

    model.train()
    for i, (images, labels) in enumerate(iter(trainloader)):
        # images, labels = images.to(device), labels.to(device)
        images.resize_(images.size()[0], 784)

        optimizer.zero_grad()

        pred = model.forward(images)
        # pred = F.softmax(logits, dim=1)
        train_loss = criterion(pred, labels)
        train_loss.backward()
        optimizer.step()

        running_loss += train_loss.item()

    model.eval()
    with torch.no_grad():
        for i, (images_test, labels_test) in enumerate(iter(testloader)):

            images_test.resize_(images_test.size()[0], 784)
            test_probs = model.forward(images_test)
            test_loss = criterion(test_probs, labels_test)

            #Check incorrect:
            pred_test = torch.argmax(test_probs, dim=1)
            incorrect_pred = ((pred_test == labels_test) == False).nonzero()
            running_loss_test += test_loss.item()
            incorrect.append(images_test[incorrect_pred].numpy())

            #Check correct for accuracy:
            correct += (pred_test == labels_test).float().sum()

    #Accuracy
    accuracy = 100 * correct / len(testset)
    accuracy = torch.round(accuracy, decimals=2)
    print(f'Accuracy = {accuracy}')

    #Running loss
    train.append((running_loss/len(trainloader)))
    test.append((running_loss_test/len(testloader)))

train_time = time.time() - start_time
print(f'Training time: {train_time}')

plt.plot(train, label='train loss')
plt.plot(test, label='test loss')
plt.legend()
plt.show()