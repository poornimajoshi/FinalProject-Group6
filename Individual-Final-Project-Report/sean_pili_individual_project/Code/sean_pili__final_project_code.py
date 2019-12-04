import pdb

import torchvision.models as models
import torch.nn as nn
import torch

alexnet = models.alexnet(pretrained=True)
alexnet.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=False),nn.Linear(in_features=9216, out_features=4096, bias=True), nn.ReLU(inplace=True),nn.Dropout(p=0.5, inplace=False),nn.Linear(in_features=4096, out_features=4096, bias=True),nn.ReLU(inplace=True),nn.Linear(in_features=4096, out_features=1000, bias=True),nn.ReLU(inplace=True),nn.Dropout(p=0.5, inplace=False),nn.Linear(in_features=1000, out_features=2, bias=True))
googlenet = models.googlenet(pretrained = True)
googlenet.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=1000, bias=True),nn.CELU(),nn.Dropout(.3),nn.Linear(1000,2))
from torchvision import models
#pdf
#res101 = models.resnet101(pretrained = True)
#res101.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=1000, bias=True),nn.ReLU(),nn.Dropout(.5),nn.Linear(1000,7))
#decaprio = models.inception_v3(pretrained=True)
#decaprio.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=1000, bias=True),nn.ReLU(),nn.Dropout(.35),nn.Linear(1000,200),nn.ReLU(),nn.Dropout(.2),nn.Linear(200,7))

#res5032 = models.resnext50_32x4d(pretrained=True)
#res5032.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=300, bias=True),nn.ReLU(),nn.Dropout(.3),nn.Linear(300,7))
vg16 = models.vgg16(pretrained=True)
vg16.classifier[6] = nn.Sequential(
                       nn.Linear(4096, 256),
                       nn.ReLU(),
                       nn.Dropout(0.4),
                       nn.Linear(256, 7))
#densenet = models.densenet161(pretrained=True)
#densenet.classifier = nn.Sequential(nn.Linear(in_features=2208, out_features=1000, bias=True),nn.ReLU(),nn.Dropout(.3),nn.Linear(1000,7))
#optimizer = optim.Adam(model.parameters(),lr = .000005)
sFN = models.shufflenet_v2_x1_5(pretrained=False)
sFN.fc = nn.Sequential(nn.Linear(1024,1000,bias=True),nn.ReLU(),nn.Dropout(.4),nn.Linear(1000,2))
mna = models.mnasnet1_0(pretrained=True)
mna.classifier =  nn.Sequential(nn.Dropout(p=0.2, inplace=True),nn.Linear(in_features=1280, out_features=1000, bias=True),nn.ReLU(),nn.Dropout(.35),nn.Linear(1000,2))
class MyEnsemble(nn.Module):
    def __init__(self,mna,sFN):
        super(MyEnsemble,self).__init__()
        self.mna = mna
        self.sFN = sFN
        #self.vg16 = vg16
        self.classifier = nn.Linear(4,2)
        self.ls = nn.LogSoftmax(dim=1)
    def forward(self,x1,x2):
        x1 = self.mna(x1)
        x2 = self.sFN(x2)
        #x3 = self.vg16(x3)
        x = torch.cat((x1,x2),dim =1)
        x = self.classifier(torch.relu(x))
        x = self.ls(x)
        return x
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable


torch.backends.cudnn.benchmark = True
BatchNorm = nn.BatchNorm2d


# __all__ = ['DRN', 'drn26', 'drn42', 'drn58']


webroot = 'https://tigress-web.princeton.edu/~fy/drn/models/'

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, pool_size=28, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                                   padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(
                BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(
                BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3,
                          bias=False),
                BatchNorm(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(
                channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(
                channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
                                       dilation=2, new_level=False)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
                                 new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
                                 new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)

        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)

        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x


class DRN_A(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(DRN_A, self).__init__()
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=4)
        self.avgpool = nn.AvgPool2d(28, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def drn_a_50(pretrained=False, **kwargs):
    model = DRN_A(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def drn_c_26(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-26']))
    return model








import math
import torch
import torch.nn as nn
#from networks.drn import drn_c_26


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, classes, pretrained_drn=False,
            pretrained_model=None, use_torch_up=False):
        super(DRNSeg, self).__init__()

        model = drn_c_26(pretrained=pretrained_drn)
        self.base = nn.Sequential(*list(model.children())[:-2])
        if pretrained_model:
            self.load_pretrained(pretrained_model)

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)

        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

    def load_pretrained(self, pretrained_model):
        print("loading the pretrained drn model from %s" % pretrained_model)
        state_dict = torch.load(pretrained_model, map_location='cpu')
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # filter out unnecessary keys
        pretrained_dict = state_dict['model']
        pretrained_dict = {k[5:]: v for k, v in pretrained_dict.items() if k.split('.')[0] == 'base'}

        # load the pretrained state dict
        self.base.load_state_dict(pretrained_dict)


class DRNSub(nn.Module):
    def __init__(self, num_classes, pretrained_model=None, fix_base=False):
        super(DRNSub, self).__init__()

        drnseg = DRNSeg(num_classes)
        if pretrained_model:
            print("loading the pretrained drn model from %s" % pretrained_model)
            state_dict = torch.load(pretrained_model, map_location='cpu')
            drnseg.load_state_dict(state_dict['model'])

        self.base = drnseg.base
        if fix_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




class DRNSean(nn.Module):
    def __init__(self, num_classes, pretrained_mod=None, fix_base=False):
        super(DRNSean, self).__init__()

        drnseg = DRNSeg(num_classes,pretrained_drn = False, pretrained_model = pretrained_mod)

        self.base = drnseg.base
        if fix_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512, 100),nn.ReLU(), nn.Dropout(.5), nn.Linear(100,num_classes),nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#############################
# training code
import matplotlib.pyplot as plt
print('y')
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

#data_dir = '/home/ubuntu/Deep-Learning/Final_Project/train'
data_dir = '/home/ubuntu/Deep-Learning/data/project_data/cropped2/train'


#Linear(in_features=512, out_features=2, bias=True)
#model = DRNSub(1,pretrained_model='/home/ubuntu/Deep-Learning/Final_Project/FALdetector/weights/global.pth')
model = DRNSean(2,pretrained_mod='/home/ubuntu/Deep-Learning/Final_Project/FALdetector/weights/global.pth')
#model = MyEnsemble(mna,sFN)
model.cuda()
print(model)
########
def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([#transforms.RandomRotation(30),  # data augmentations are great
                                       #transforms.RandomResizedCrop(224),  # but not in this case of map tiles
                                       #transforms.RandomHorizontalFlip(),
                                       transforms.Resize((400,400)),
                                       transforms.ToTensor(),
                                       #transforms.Normalize([0.485, 0.456, 0.406], # PyTorch recommends these but in this
                                       #                     [0.229, 0.224, 0.225]) # case I didn't get good results
                                       ])

    test_transforms = transforms.Compose([transforms.Resize((400,400)),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                      ])

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=17)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=17)
    return trainloader, testloader
print('y')
trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)
#TODO, we gotta find the correct samples to train and test on
LR = 0.00007
criterion = nn.NLLLoss()
#n = torch.sigmoid
#optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = optim.Adadelta(model.parameters())
# I got the best with 50000 resolution on the baseline with adadelta after the 5th epoch...
epochs = 15
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    print('ho HOOOOO, were TRAINing a NEURAL netWORK')
    LR = LR*.75
    running_loss = 0
    steps = 0
    #if epoch > 5:
    #    optimizer = optim.SGD(model.parameters(), lr=LR,weight_decay = .1)

    #optimizer = optim.Adam(model.parameters(),lr = LR)
    for inputs, labels in trainloader:
        steps += 1
        inputs = Variable(inputs).cuda()
        #labels = labels.LongTensor()
        #labels = torch.nn.functional.one_hot(labels,num_classes =2)
        labels = Variable(labels).cuda()
        #inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() / steps

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = Variable(inputs).cuda(),  Variable(labels).cuda()#inputs, labels = inputs.to(device),
                    #labels = torch.nn.functional.one_hot(labels, num_classes=2)

                    #   labels.to(device)
                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = logps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                #model.train()
    train_losses.append(running_loss / len(trainloader))
    test_losses.append(test_loss / len(testloader))
    print(f"Epoch {epoch + 1}/{epochs}.. "
          f"Train loss: {running_loss / print_every:.3f}.. "
          f"Test loss: {test_loss / len(testloader):.3f}.. "
          f"Test accuracy: {accuracy / len(testloader):.3f}")
    running_loss = 0
    torch.save(model, str(epoch) +'cropped_baseline_allAD.pth')

    model.train()
torch.save(model, 'cropped_baseline_base_allAD.pth')


########################################## now evaluation script 
import torch
import numpy as np
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import pickle
#columbia plaza
BatchNorm = nn.BatchNorm2d
device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, pool_size=28, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                                   padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(
                BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(
                BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3,
                          bias=False),
                BatchNorm(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(
                channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(
                channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
                                       dilation=2, new_level=False)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
                                 new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
                                 new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)

        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)

        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x


#############
model_urls = "we don't both with this."
def drn_c_26(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-26']))
    return model
#####



class DRNSeg(nn.Module):
    def __init__(self, classes, pretrained_drn=False,
            pretrained_model=None, use_torch_up=False):
        super(DRNSeg, self).__init__()

        model = drn_c_26(pretrained=pretrained_drn)
        self.base = nn.Sequential(*list(model.children())[:-2])
        if pretrained_model:
            self.load_pretrained(pretrained_model)

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)

        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

    def load_pretrained(self, pretrained_model):
        print("loading the pretrained drn model from %s" % pretrained_model)
        state_dict = torch.load(pretrained_model, map_location='cpu')
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # filter out unnecessary keys
        pretrained_dict = state_dict['model']
        pretrained_dict = {k[5:]: v for k, v in pretrained_dict.items() if k.split('.')[0] == 'base'}

        # load the pretrained state dict
        self.base.load_state_dict(pretrained_dict)

###########################################################################################################################
###########################################################################################################################



class DRNSean(nn.Module):
    def __init__(self, num_classes, pretrained_mod=None, fix_base=False):
        super(DRNSean, self).__init__()

        drnseg = DRNSeg(num_classes,pretrained_drn = False, pretrained_model = pretrained_mod)

        self.base = drnseg.base
        if fix_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512, 100),nn.ReLU(), nn.Dropout(.5), nn.Linear(100,num_classes), nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = torch.load('5cropped_baseline.pth')




data_dir = "/home/ubuntu/Deep-Learning/data/project_data/cropped2/test"

test_transforms = transforms.Compose([transforms.Resize((500,500)),
                                      transforms.ToTensor()
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225]),
                                      ])


def predict_image(image):
    #image_tensor = test_transforms(image).float()
    image_tensor = image.view(1, 3, 500, 500).float()
    #print(image_tensor.shape)
    #mage_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor).cuda()
    input = input.to(device)
    output = model(input)
    #return output
    index = output.data.cpu().numpy()
    index = np.exp(index)
    return index
#my_dataloader = utils.data.DataLoader(my_dataset, shuffle = True,batch_size = 10,num_workers = 0)

def get_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    #indices = list(range(len(data)))
    #np.random.shuffle(indices)
    #idx = indices[:num]
    #from torch.utils.data.sampler import SubsetRandomSampler
    #sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, shuffle =False, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels

for param in model.parameters():
    param.requires_grad = False

images, labels = get_images(381)

timwho = [predict_image(images[i]) for i in range(len(images))]
tim = [i.argmax() for i in timwho]
#xt = labels.numpy()
#sum(xt==tim)/len(xt)


p_dir = '/home/ubuntu/Deep-Learning/Final_Project/pickles/drnC26_seanpili_p'
L_dir = '/home/ubuntu/Deep-Learning/Final_Project/pickles/drnC26_seanpili_L'
pickle.dump(timwho,open(L_dir,'wb'))
pickle.dump(tim,open(p_dir,'wb'))


import pickle
import numpy as np
import torch
import os
import pandas as pd

# so the order is  resnet18, mobilenet, resnet50, vgg16
my_paths3 =['mobilenetv2_train_poornimajoshi.csv', 'resnet18_train_poornimjoshi.csv', 'densenet121_train_chirag.csv','resnet50_train_chirag.csv', 'vgg16_train_chirag.csv']
# you want to replace the list here with the list of paths directed to your predictions (pickled python lists)
source_dir = '/home/ubuntu/Deep-Learning/Final_Project/pickles'
pot_paths3 = os.listdir('/home/ubuntu/Deep-Learning/Final_Project/pickles')
#my_paths3 = [i for i in pot_paths3 if '.csv' in i ]

my_frames = [pd.read_csv(source_dir + '/' + i) for i in my_paths3]
import re
pat1 = re.compile('\.[0-9]+')
pat2 = re.compile('\.[0-9]+|0\.|1\.')

logitz = []
for x in my_frames:
    blanks = []
    for j in range(x.shape[0]):
        temp = np.array([ float(i)  for i in pat1.findall(x.Logits[j])])
        if len(temp) ==0:
            print('flamin cheetos')
            blanks.append(np.array([ float(i)  for i in pat2.findall(x.Logits[j])]))
        else:
            blanks.append(np.array([ float(i)  for i in pat1.findall(x.Logits[j])]))


    logitz.append(np.array(blanks))
# read in the model's predictions
st = logitz[0][0]
# we currently exclude #4 , or well 5, python indexing because chirag didn't have the correct model there.

chirag_logs = np.hstack((logitz[0],logitz[1],logitz[2],logitz[3],logitz[4]))
y_train = my_frames[1].TrueLabels



# read in the gound truth from list form
y_test_path = '/home/ubuntu/Deep-Learning/Final_Project/pickles/gt'
y_test = np.array(pickle.load(open(y_test_path,'rb')))

# now turn those indicators into actual predicted values..

####### THIS IS WHERE I LOAD IN THE TEST DATA WE NEED
def read_soft(paths):
    """
    :param paths:  should be a list of paths to the lists of predictions
    """
    if 'resnet18' in paths[0]:
        first_arr = np.array( list(reversed(pickle.load(open(paths[0], 'rb')))) ).reshape(1,381,2)
    else:
        first_arr = np.array(pickle.load(open(paths[0], 'rb'))).reshape(1, 381, 2)
    #print(first_arr.shape)
    # this loads all of the predictions into a list of lists.
    for i in paths[1:]:
        #print(i)
       # print(first_arr.shape)
        #print(np.array(pickle.load(open(i, 'rb'))).shape)

        first_arr = np.vstack((first_arr, np.array(pickle.load(open(i, 'rb'))).reshape(1,381,2)))
    # it = iter(blank)
    # it won't work if the lists aren't all the same the length, so I raise this error.
    # the_len = len(next(it))
    # if not all(len(l) == the_len for l in it):
    #    raise ValueError('not all lists have same length!')
    return first_arr
# you want to replace the list here with the list of paths directed to your predictions (pickled python lists)
source_dir = '/home/ubuntu/Deep-Learning/Final_Project/pickles'
pot_paths2 = os.listdir('/home/ubuntu/Deep-Learning/Final_Project/pickles')
my_paths2 = []

for i in pot_paths2:
    if '_p' not in i and '_L' in i and 'drn' not in i:
        my_paths2.append(source_dir + '/'+i)
poors = []
for i in pot_paths2:
    if 'poor' in i and '_L' in i:
        poors.append(i)
no_poors = [i for i in my_paths2 if 'poornimajoshi' not in i]
no_poors = sorted(no_poors)
predz2 = read_soft(no_poors)
print(no_poors)
poors = sorted(poors)
print(poors)
# dang it
if len(poors) ==0:
    pass
elif len(poors) ==1:
    predz2 = np.vstack((np.array(pickle.load(open(source_dir + '/' + poors[0], 'rb')).reshape(1, 381, 2), predz2)))
    #np.vstack(predz2,pickle.load(poors[0],'rb'))
elif len(poors)>1:
    for i in poors:
        print(predz2.shape)
        if 'resnet18' in i:
            print('res')
            predz2 = np.vstack((np.array(list(reversed(pickle.load(open(source_dir + '/' + i, 'rb'))))).reshape(1, 381, 2), predz2))

        else:
            predz2 = np.vstack((np.array(pickle.load(open(source_dir + '/' + i, 'rb'))).reshape(1, 381, 2), predz2))
##############################
skL = [np.concatenate((predz2[0][i],predz2[1][i],predz2[2][i],predz2[3][i],predz2[4][i])) for i in range(predz2.shape[1])]
start = skL[0]
for i in skL[1:]:
    start = np.vstack((start,i))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
penalties = ['l1','l2']
Cs = [.01,.1,.5,1,10,50,100]
fit_intercept = ['True','False']
rs = [0]

hyperparameters = {'C':Cs, 'penalty':penalties,'random_state':rs,'fit_intercept':fit_intercept}

#skL = [np.concatenate((predz2[0][i],predz2[1][i],predz2[2][i],predz2[3][i])) for i in range(predz2.shape[1])]
#
# start = skL[0]
#
# for i in skL[1:]:
#     start = np.vstack((start,i))

GS = GridSearchCV(LogisticRegression(),hyperparameters,cv = 3)

gwid = GS.fit(chirag_logs, y_train)

print(gwid.best_estimator_)

best_Log = gwid.best_estimator_.fit(chirag_logs,y_train)

blPreds = best_Log.predict(chirag_logs)

legit_preds = best_Log.predict(start)



ne = [30,50,100,250,1000,10000]
from sklearn.ensemble import AdaBoostClassifier
hyperparameters2 = {'n_estimators':ne,'random_state':rs}

GS2 = GridSearchCV(AdaBoostClassifier(),hyperparameters2,cv = 3)

ada = GS2.fit(chirag_logs, y_train)

ada_best = ada.best_estimator_.fit(chirag_logs,y_train)

adP = ada_best.predict(start)


hyperparameters3={
'criterion': ['gini','entropy'],
'n_jobs': [-1],
'random_state':[0],
'n_estimators':[51,101,501,1001]}


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
GS3 = GridSearchCV(RandomForestClassifier(),hyperparameters3,cv = 3)

rF = GS3.fit(chirag_logs, y_train)

best_rF = rF.best_estimator_.fit(chirag_logs,y_train)

rfP = best_rF.predict(start)



eclf1 = VotingClassifier(estimators=[('lr', best_Log), ('rf', best_rF)], voting='soft')

eclf1.fit(chirag_logs,y_train)

big_predz = eclf1.predict(start)
sum(big_predz==y_test)



import pickle
import numpy as np
import torch
import os


def read_fun(paths):
    """
    :param paths:  should be a list of paths to the lists of predictions
    """
    print(paths[0])
    first_arr = np.array([pickle.load(open(paths[0], 'rb'))])

    # this loads all of the predictions into a list of lists.
    for i in paths[1:]:
        print(i)
        if  'resnet18' in i:
            toomi = pickle.load(open(i, 'rb'))

            print(first_arr.shape)
            first_arr = np.vstack((first_arr, np.array(  list(reversed(pickle.load(open(i, 'rb')))) )))
        else:
            first_arr = np.vstack((first_arr, np.array(pickle.load(open(i, 'rb')))))


    # it = iter(blank)
    # it won't work if the lists aren't all the same the length, so I raise this error.
    # the_len = len(next(it))
    # if not all(len(l) == the_len for l in it):
    #    raise ValueError('not all lists have same length!')
    return first_arr
# you want to replace the list here with the list of paths directed to your predictions (pickled python lists)
source_dir = '/home/ubuntu/Deep-Learning/Final_Project/pickles'
pot_paths = os.listdir('/home/ubuntu/Deep-Learning/Final_Project/pickles')
my_paths = []
for i in pot_paths:
    if '_L' not in i and '_p' in i and 'drn' not in i and '.csv' not in i:
        my_paths.append(source_dir + '/'+i)
# read in the model's predictions
predz = read_fun(my_paths)
# this is what actually gets us the indicators for the majority decision

predz = predz[[0,2,3]]
maj = np.mean(predz,axis=0)

# read in the gound truth from list form
y_test_path = '/home/ubuntu/Deep-Learning/Final_Project/pickles/gt'
y_test = np.array(pickle.load(open(y_test_path,'rb')))

# now turn those indicators into actual predicted values..
blank = []

for i in maj:
    if i > .5:
        blank.append(1)
    elif i < .5:
        blank.append(0)
    elif i == .5:
        blank.append(int(np.random.randint(low=0, high=2, size=1)))

maj_vote = np.array(blank)
maj_acc = np.sum(y_test==maj_vote)/len(y_test)
print('the accuracy of all classifiers by hard voting is: {}'.format(round(maj_acc*100,2)))

for i in range(predz.shape[0]):
    print("The accuracy for your {}th classifier was: {}%".format(i+1,np.sum(y_test==predz[i])/len(y_test)))


#myT = pickle.load(open('/home/ubuntu/Deep-Learning/Final_Project/pickles/mobilenet_poornimajoshi_L.pickle','rb'))

# so mobilenet was only 58% accurate on the fakes, and 62% accurate on the reals, overal 60% accurate

#resnet18 was 72% accurate on the fakes on the 52% accurate on the reals,
# resnet 18 might do better if we trained a little more on

# vgg16 was 61% accurate on the fakes and 65% accurate on the reals

# res50   58% on fakes 65% on real


