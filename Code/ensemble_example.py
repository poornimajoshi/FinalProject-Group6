import torchvision.models as models
import torch.nn as nn
import torch

#alexnet = models.alexnet(pretrained=True)
#alexnet.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=False),nn.Linear(in_features=9216, out_features=4096, bias=True), nn.ReLU(inplace=True),nn.Dropout(p=0.5, inplace=False),nn.Linear(in_features=4096, out_features=4096, bias=True),nn.ReLU(inplace=True),nn.Linear(in_features=4096, out_features=1000, bias=True),nn.ReLU(inplace=True),nn.Dropout(p=0.5, inplace=False),nn.Linear(in_features=1000, out_features=7, bias=True))
#googlenet = models.googlenet(pretrained = False)
#googlenet.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=1000, bias=True),nn.CELU(),nn.Dropout(.3),nn.Linear(1000,7))
from torchvision import models
#pdf
#res101 = models.resnet101(pretrained = True)
#res101.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=1000, bias=True),nn.ReLU(),nn.Dropout(.5),nn.Linear(1000,7))
#decaprio = models.inception_v3(pretrained=True)
#decaprio.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=1000, bias=True),nn.ReLU(),nn.Dropout(.35),nn.Linear(1000,200),nn.ReLU(),nn.Dropout(.2),nn.Linear(200,7))

#res5032 = models.resnext50_32x4d(pretrained=True)
#res5032.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=300, bias=True),nn.ReLU(),nn.Dropout(.3),nn.Linear(300,7))
#vg16 = models.vgg16(pretrained=True)
#vg16.classifier[6] = nn.Sequential(
#                       nn.Linear(4096, 256),
#                       nn.ReLU(),
#                       nn.Dropout(0.4),
#                       nn.Linear(256, 7))
# BEST RESULT SO FAR WAS RUNING THAT BOI WITH THAT OPTIMIZER FOR LIKE 17 EPOCHS...
#densenet = models.densenet161(pretrained=True)
#densenet.classifier = nn.Sequential(nn.Linear(in_features=2208, out_features=1000, bias=True),nn.ReLU(),nn.Dropout(.3),nn.Linear(1000,7))
#optimizer = optim.Adam(model.parameters(),lr = .000005)
sFN = models.shufflenet_v2_x1_5(pretrained=False)
sFN.fc = nn.Sequential(nn.Linear(1024,1000,bias=True),nn.PReLU(),nn.Dropout(.4),nn.Linear(1000,7))
mna = models.mnasnet1_0(pretrained=True)
mna.classifier =  nn.Sequential(nn.Dropout(p=0.2, inplace=True),nn.Linear(in_features=1280, out_features=1000, bias=True),nn.PReLU(),nn.Dropout(.35),nn.Linear(1000,7))

class MyEnsemble(nn.Module):
    def __init__(self,mna,sFN):
        super(MyEnsemble,self).__init__()
        self.mna = mna
        self.sFN = sFN
        #self.vg16 = vg16
        self.classifier = nn.Linear(14,7)
    def forward(self,x1,x2):
        x1 = self.mna(x1)
        x2 = self.sFN(x2)
        #x3 = self.vg16(x3)
        x = torch.cat((x1,x2),dim =1)
        x = self.classifier(torch.relu(x))
        return x



        #self.fc1 = nn.Linear(14,7)

#vg9 =models.vgg19_bn(pretrained=True)
#vg9.classifier[6] = nn.Linear(in_features=4096,out_features = 7)

import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import zipfile
os.listdir()
myD = os.getcwd()
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset
#with zipfile.ZipFile(myD + '/train-Exam2.zip','r') as zip_ref:
#    zip_ref.extractall(myD+'/'+'final_exam_train')
from sklearn.preprocessing import LabelEncoder
import cv2
from cv2 import imread
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from torch import utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#from torch.utils.data import Dataset,DataLoader

import random
img_dir = myD  +'/final_exam_train/train'

my_imagez = sorted([i for i in os.listdir(img_dir) if 'png' in i])

my_labelz = sorted([i for i in os.listdir(img_dir) if 'txt' in i])

imageZ = np.array([cv2.resize(imread(img_dir + '/' + imag),(224,224)) for imag in my_imagez])
#imz = np.stack(imageZ)
textz =[ open(img_dir+'/'+i,'r').read().split('\n') for i in my_labelz]

y = np.array(textz)

xy = MultiLabelBinarizer()

xy.classes =  ['red blood cell', "difficult" ,"gametocyte","trophozoite","ring",'schizont','leukocyte']


TIMMAY = xy.fit_transform(y)


#### custom class to transform taken from stackflow

#https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

##################3 pretty sure I dont' use anything from when I create dist until I start talking about
# formatting the data :)
# dist = np.unique(TIMMAY,axis=0,return_counts=True)
#
# # create a dictionary that will keep track of classes that have less than 10 samples.
# # I will start by adding 5-10 transforms on each of these images
# cd = {}
#
# for i in range(dist[0].shape[0]):
#     cd[str(dist[0][i])] = dist[1][i]
#
#
# lows = [i for i in list(cd.keys()) if cd[i]< 11]
#
# # now, I will create a list of indices that I need to create transforms for?!
#
# # I then create a numpy array that is just those inidices and COPY IT
# # after I copy it.... I go through some kinda loop or manually create more copies which transformations will be applied to.
#
#
# TA = []
# indz = []
# for i in range(TIMMAY.shape[0]):
#     if str(TIMMAY[i]) in lows:
#         TA.append(True)
#         indz.append(i)
#         #print(cd[str(TIMMAY[i])],str(TIMMAY[i]))
#     else:
#         TA.append(False)
#
#
# tempIm = imageZ[TA]
# tempLab = TIMMAY[TA]
#
# baseIm = tempIm.copy()
# baseLab = tempLab.copy()

#tensor_ = torch.sta



##### FORMATTING THE DATA
imCop = imageZ.copy()
labCop = TIMMAY.copy()
texCop = textz.copy()
ds4 = np.array([i for i in range(len(textz)) if 'red blood cell' in textz[i] and 'leukocyte' in textz[i]])
leuk_test = ds4[0:10]
leuk_train = ds4[10:]
# do schizont
# then gameocyte
# difficult is 2nd to last
# trophozoite is last
# then we can just subset for red blood cell only...

#ds5 = np.array([i for i in range(len(textz)) if 'red blood cell' in textz[i] and 'leukocyte' in textz[i]])
ds5 = np.array([i for i in range(len(textz)) if 'red blood cell' in textz[i] and 'schizont' in textz[i] and i not in ds4])

sch_test = ds5[0:15]
sch_train = ds5[15:]

d45 = np.append(ds4,ds5)

ds6 = np.array([i for i in range(len(textz)) if 'red blood cell' in textz[i] and 'gametocyte' in textz[i] and i not in d45])

gam_test = ds6[0:15]
gam_train = ds6[15:]

d456 = np.append(d45,ds6)

ds7 = np.array([i for i in range(len(textz)) if 'red blood cell' in textz[i] and 'ring' in textz[i] and i not in d456])

ring_test = ds7[0:15]
ring_train = ds7[15:]


d4567  = np.append(d456,ds7)

ds8 = np.array([i for i in range(len(textz)) if 'red blood cell' in textz[i] and 'difficult' in textz[i] and i not in d4567])

dif_test = ds8[0:15]
dif_train = ds8[15:]

d45678 = np.append(d4567,ds8)

ds9 = np.array([i for i in range(len(textz)) if 'red blood cell' in textz[i] and 'trophozoite' in textz[i] and i not in d45678])

tro_test = ds9[0:15]
tro_train = ds9[15:]

ds10 = np.array([i for i in range(len(textz)) if textz[i] == ['red blood cell'] ])

red_test = ds10[0:20]
red_train = ds10[20:]

rtc = red_test.copy()

rtc = list(rtc)

rtc.extend(list(tro_test))
rtc.extend(list(dif_test) )
rtc.extend(list(ring_test))
rtc.extend(list(gam_test))
rtc.extend(list(sch_test))
rtc.extend(list(leuk_test))


train_inds = [i for i in range(len(labCop)) if i not in rtc]


test_labs = labCop[rtc]
test_ims = imCop[rtc]

train_transforms = transforms.Compose  ([transforms.ToPILImage(),transforms.ColorJitter(brightness =.5,contrast = .5),transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

train_labs = labCop[train_inds]
train_ims = imCop[train_inds]
train_ims  = train_ims.reshape(824,3,224,224)

#H_flip_train = train_ims.copy()
#H_flip_train = np.array([cv2.flip(i,0) for i in H_flip_train])
#Gauss_train = train_ims.copy()
#Gauss_train - np.array([cv2.GaussianBlur(i,(5,5),0) for i in Gauss_train])

#big_train_ims = np.concatenate((H_flip_train,train_ims))
#big_train_labs = np.concatenate((train_labs,train_labs))


##### PUT THAT ISH ON THE GPU / GET IT READY FOR PYTROCH

tensor_x_train = torch.stack([torch.Tensor(i) for i in train_ims])
tensor_y_train = torch.stack([torch.Tensor(i) for i in train_labs])

tensor_x_test = torch.stack([torch.Tensor(i) for i in test_ims])
tensor_y_test = torch.stack([torch.Tensor(i) for i in test_labs])


my_dataset = CustomTensorDataset(tensors= (tensor_x_train,tensor_y_train),transform = train_transforms)
#my_dataset = utils.data.TensorDataset(tensor_x_train,tensor_y_train,transform = train_transforms )
my_dataloader = utils.data.DataLoader(my_dataset, shuffle = True,batch_size = 10,num_workers = 0)


test_set = utils.data.TensorDataset(tensor_x_test,tensor_y_test)
test_loader = utils.data.DataLoader(test_set, shuffle = True,batch_size = 105,num_workers = 0)
# technically, we really only need to load the test data one time since we wont' iterate through it.... so let's do it here!
test_init = iter(test_loader)
x_test, y_test = test_init.next()

x_test = x_test.view(x_test.shape[0],3,224,224).float()
x_test = Variable(x_test).cuda()
y_test = Variable(y_test).cuda()
# for batch_idx, (data,target) in (enumerate(my_dataloader)):
#     print('Batch idx {}, data shape {}, target shape{}'.format(batch_idx,data.shape,target.shape))


# class CNN(torch.nn.Module):
#
#     def __init__ (self):
#         super(CNN, self).__init__()
#
#         self.conv1 = torch.nn.Conv2d(3,12,(5,5),padding = 1) # so in this case, 6,128,128
#         self.convonorm1 = nn.BatchNorm2d(12)
#         self.conv1a = torch.nn.Conv2d(12,36,(3,3),padding =1 )
#         self.convonorm1a = nn.BatchNorm2d(36)
#         self.pool1 = nn.MaxPool2d(2,2) # now I got 12, 64,64
#         self.conv2 = nn.Conv2d(36,72,kernel_size = (5,5), padding =1) # now I got 24, 62,62
#         self.convonorm2 = nn.BatchNorm2d(72)
#         self.conv2a = nn.Conv2d(72,144,kernel_size = (3,3),padding = 1)
#         self.convonorm2a = nn.BatchNorm2d(144)
#         self.pool2  = nn.MaxPool2d((2,2)) # now 96, 31,31
#         self.conv3 = torch.nn.Conv2d(144,180,(5,5),padding =2)
#         self.convonorm3 = nn.BatchNorm2d(180)
#         self.pool3 = nn.MaxPool2d((2,2),ceil_mode = True)
#         self.conv4 = torch.nn.Conv2d(180,201,(3,3),padding=1) # so....
#         self.convonorm4 = nn.BatchNorm2d(201)
#         self.pool4 = nn.MaxPool2d(2,2) # now 8x8, was 16x16 before allegedly
#         self.conv5 = torch.nn.Conv2d(201,288,(3,3),padding =1 )
#         self.convonorm5 = nn.BatchNorm2d(288)
#         self.linear1 = nn.Linear(288*8*8,200)
#         #self.linear1_bn = nn.BatchNorm1d(300)
#         self.drop = nn.Dropout(.2)
#         self.linear2 = nn.Linear(200,100)
#         self.linear3 = nn.Linear(100,50)
#         self.linear4 = nn.Linear(50,7)
#
#         self.act = torch.relu
#
#     def forward(self, x):
#         #x = self.pool1(self.convonorm1(self.act(self.conv1(x))))
#         #x = self.pool2(self.convonorm2(self.act(self.conv2(x))))
#         x = self.pool1(self.convonorm1a(self.act(self.conv1a(self.convonorm1(self.act(self.conv1(x)))))))
#         x = self.pool2(self.convonorm2a(self.act(self.conv2a(self.convonorm2 (self.act(self.conv2(x)))))))
#         x = self.pool3(  self.convonorm3(self.act(self.conv3(x)))   )
#         x = self.pool4(self.convonorm4(self.act(self.conv4(x))))
#         x = self.convonorm5(self.act(self.conv5(x)))
#         x = self.drop(self.act(self.linear1(x.view(len(x), -1))))
#         x = self.drop(self.act(self.linear2(x)))
#         x = self.drop(self.act(self.linear3(x)))
#         return self.linear4(x)
#

model = MyEnsemble(mna,sFN)
model.cuda()

criterion = nn.BCEWithLogitsLoss()
#optimizer = optim.SGD(model.parameters(),lr=.0001,momentum=.3)
optimizer = optim.Adam(model.parameters(),lr = .0000001)

epochs = 30
#print_every = 10
train_losses, test_losses = [], []
########
my_list = []
print("Starting training loop...")
for epoch in range(epochs):

    loss_train = 0
    model.train()
    for inputs, labels in my_dataloader :
        #model.train()
        #print(inputs)
        inputs = inputs.view(inputs.shape[0],3,224,224).float()
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        m = nn.Sigmoid()
        optimizer.zero_grad()
        logits = m(model(inputs,torch.tensor(inputs,requires_grad = True) ).float())
        #print(torch.eq((m(logits)>.5).float(),labels))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    model.eval()
    with torch.no_grad():
        m = nn.Sigmoid()
        y_test_pred = m(model(x_test,torch.tensor(x_test,requires_grad = False)))
        crit2 = nn.BCELoss()
        loss = crit2(y_test_pred, y_test)
        loss_test = loss.item()
    print("Epoch {} | Train loss {:.5f}, Test Loss {}".format(epoch,loss_train/10,loss_test))







#
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0003)
# epochs = 15
# print_every = 10
# train_losses, test_losses = [], []
# ########
# my_list = []
# print("Starting training loop...")
# for epoch in range(epochs):
#
#     loss_train = 0
#     model.train()
#     for inputs, labels in my_dataloader :
#         #model.train()
#         #print(inputs)
#         inputs = inputs.view(inputs.shape[0],3,130,130).float()
#         inputs = Variable(inputs).cuda()
#         labels = Variable(labels).cuda()
#         m = nn.Sigmoid()
#         optimizer.zero_grad()
#         logits = m(model(inputs))
#         #print(torch.eq((m(logits)>.5).float(),labels))
#         loss = criterion(logits, labels)
#         loss.backward()
#         optimizer.step()
#         loss_train += loss.item()
#     print("Epoch {} | Train loss {:.5f}".format(epoch,loss_train/10))
#     #model.eval()
#     #with torch.no_grad():
#     #    y_test_pred = model(x_test)
#     #    loss = criterion(y_test_pred, y_test)
#     #    loss_test = loss.item()

######
#3####
####
# for epoch in range(epochs):
#     print('ho HOOOOO, were TRAINing a NEURAL netWORK')
#     running_loss = 0
#     steps = 0
#     for inputs, labels in my_dataloader :
#         model.train()
#         steps += 1
#         #print(inputs)
#         inputs = inputs.view(1,3,130,130).float()
#         inputs = Variable(inputs).cuda()
#         labels = Variable(labels).cuda()
#         #inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         logps = model(inputs)
#         loss = criterion(logps, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() / steps
#
#         #if steps % print_every == 0:
#         #    test_loss = 0
#         #    accuracy = 0
#         #    model.eval()
#             # with torch.no_grad():
#             #     for inputs, labels in testloader:
#             #         inputs, labels = Variable(inputs).cuda(),  Variable(labels).cuda()#inputs, labels = inputs.to(device),
#             #      #   labels.to(device)
#             #         logps = model(inputs)
#             #         batch_loss = criterion(logps, labels)
#             #         test_loss += batch_loss.item()
#             #
#             #         ps = torch.exp(logps)
#             #         top_p, top_class = ps.topk(1, dim=1)
#             #         equals = top_class == labels.view(*top_class.shape)
#             #         accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
#                 #model.train()
#     train_losses.append(running_loss / len(my_dataloader ))
#     #test_losses.append(test_loss / len(testloader))
#     print(f"Epoch {epoch + 1}/{epochs}.. "
#           f"Train loss: {running_loss / print_every:.3f}.. ")
#           #f"Test loss: {test_loss / len(testloader):.3f}.. "
#           #f"Test accuracy: {accuracy / len(testloader):.3f}")
#     running_loss = 0

torch.save(model.state_dict(), 'shufflenet.pt')