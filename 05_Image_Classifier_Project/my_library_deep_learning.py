# Imports here
from torchvision import datasets, transforms, models 
import torch 
from torch import nn 
from torch import optim 
import torch.nn.functional as F 


#----------------------------------------------------------------#
def createMyVgg(dropout=0.2, lr=0.001):
    # create model
    model = models.vgg16(pretrained=True);

    # Freeze Parameter update
    for param in model.parameters():
        param.requires_grad = False    
    
    # modify last layers
    my_classifier = nn.Sequential(nn.Linear(25088, 4096),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(4096, 1000),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(1000, 102),
                                   nn.LogSoftmax(dim=1));
    model.classifier = my_classifier;
    
    # create an instance of the optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr) 
    
    return model, optimizer

#----------------------------------------------------------------#
def createMyResNet(dropout=0.2, lr=0.001):
    # create model
    model = models.resnet50(pretrained=True);

    # Freeze Parameter update
    for param in model.parameters():
        param.requires_grad = False    
    
    # modify last layers
    my_classifier = nn.Sequential(nn.Linear(2048, 1000),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(1000, 500),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(500, 102),
                                   nn.LogSoftmax(dim=1))
    model.fc = my_classifier;
    
    # create an instance of the optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr) 
    
    return model, optimizer






























    