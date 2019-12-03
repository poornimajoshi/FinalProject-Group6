import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import random
import numpy as np


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_transforms = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomRotation(10),
                           transforms.RandomCrop((256, 256), pad_if_needed=True),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                       ])

test_transforms = transforms.Compose([
                           transforms.CenterCrop((256, 256)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                       ])

BATCH_SIZE = 64
'''
train_data = datasets.ImageFolder('/home/ubuntu/Deep-Learning/FinalProject/data/cropped2/train/', train_transforms)
valid_data = datasets.ImageFolder('/home/ubuntu/Deep-Learning/FinalProject/data/cropped2/test/', test_transforms)
test_data = datasets.ImageFolder('/home/ubuntu/Deep-Learning/FinalProject/data/cropped2/validate/', test_transforms)



print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')



train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torchvision.models as models

model = models.resnet18(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

print(model.fc)

model.fc = nn.Linear(in_features=512, out_features=2).to(device)

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc


def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        fx = model(x)

        loss = criterion(fx, y)

        acc = calculate_accuracy(fx, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            fx = model(x)

            loss = criterion(fx, y)

            acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), fx


EPOCHS = 10
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'resnet18-fake-vs-real.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(EPOCHS):

    train_loss, train_acc = train(model, device, train_iterator, optimizer, criterion)
    valid_loss, valid_acc, pred = evaluate(model, device, valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:05.2f}% |')





test_loss, test_acc, y_pred = evaluate(model, device, valid_iterator, criterion)
print(len(y_pred))
print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:05.2f}% |')

model.load_state_dict(torch.load(MODEL_SAVE_PATH))


'''
Generate csv for ensembling
'''
from os import listdir
from os.path import isfile, join
mypath = "/home/ubuntu/data/project_data/cropped2/train/real/"
onlyfiles1 = []
onlyfiles1.extend([mypath + f for f in listdir(mypath) if isfile(join(mypath, f))])
mypath = "/home/ubuntu/data/project_data/cropped2/train/fake/"
onlyfiles1.extend([mypath + f for f in listdir(mypath) if isfile(join(mypath, f))])
onlyfiles1 = sorted(onlyfiles1)
print(len(onlyfiles1))



import torch
from PIL import Image
from pathlib import Path
pred = []
pred_logs = []
acc=[]
categorical = {"real":1,"fake":0}
for file in onlyfiles1:
    image = Image.open(Path(file))
    input = test_transforms(image)
    input = input.view(1, 3, 256,256)
    output = model(input)
    prediction = np.argmax(torch.nn.functional.softmax(output.data, dim=1).numpy(), axis=-1)
    acc.append(categorical[file.split("/")[-2]])
    acc.append(prediction[0])
    acc.append((torch.nn.functional.softmax(output.data, dim=1)[0]).numpy())
    acc.append(file)

import pandas as pd
df = pd.DataFrame((np.array(acc).reshape(-1,4)))
df.columns = ["TrueLabels", "Prediction", "Logits", "FileName"]
df.to_csv("resnet18_train_poornimajoshi.csv", index = False)

print("---Done---")