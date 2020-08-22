# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import time
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

import pandas as pd
from pandas import Series
import json

#----------------------------------------------------------------# OK
def createTrainLoader( folder_path_data,batch_size=64):
    my_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    my_datasets = datasets.ImageFolder(folder_path_data, transform=my_transforms)
    my_loader   = torch.utils.data.DataLoader(my_datasets, batch_size=batch_size, shuffle=True)
    return my_loader

#----------------------------------------------------------------# OK
def createTestLoader( folder_path_data,batch_size=64):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    my_datasets = datasets.ImageFolder(folder_path_data, transform=my_transforms)
    my_loader   = torch.utils.data.DataLoader(my_datasets, batch_size=batch_size, shuffle=True)
    return my_loader
    
#----------------------------------------------------------------# OK
# getModelsInfo is user in createMyModel
def getModelsInfo( model ):
    model_vars         = vars(model)
    model_modules      = model_vars["_modules"]
    model_modules_list = list(model_modules.items())
    module_last_name   = model_modules_list[-1][0]
    module_last_seq    = model_modules_list[-1][1]
    #
    try:
        # if classifier has several layers
        module_last_seq_in = module_last_seq[0].in_features
    except:
        # if classifier has only one layer
        module_last_seq_in = module_last_seq.in_features
    #
    return module_last_name, module_last_seq_in
    
    
#----------------------------------------------------------------# OK
def createMyModel(model_type="vgg16", dropout=0.2, lr=0.001, h_1=2000, h_2=1000, out=102):
    # create model
    exec_str = "model = models." + model_type + "(pretrained=True)"
    
    # initialize
    model      = []
    optimizer  = []
    exec_local = {}
    
    # update local variable
    exec_local['model'] = model
    exec(exec_str, globals(), exec_local)
    model = exec_local['model']    

    # get model infomation
    module_last_name, module_last_seq_in = getModelsInfo( model )

    # Freeze Parameter update
    for param in model.parameters():
        param.requires_grad = False  
        
    # modify last layers
    my_classifier = nn.Sequential(nn.Linear(module_last_seq_in, h_1),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(h_1, h_2),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(h_2, out),
                                   nn.LogSoftmax(dim=1));
    
    # update local variable
    exec_local["my_classifier"] = my_classifier
    exec_str = "model." + module_last_name + " = my_classifier;"
    exec(exec_str, globals(), exec_local)
    model = exec_local['model']
    
    # create an instance of the optimizer
    exec_local["optimizer"] = optimizer
    exec_local["lr"]        = lr
    exec_str = "optimizer = optim.Adam(model." + module_last_name + ".parameters(), lr=lr)"
    exec(exec_str, globals(), exec_local)
    optimizer = exec_local['optimizer']
    
    return model, optimizer

#-------------------------------------------------------------------------------------#
def checkDeviceType(device_str):
    if (device_str == "gpu"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    #
    return device

#-------------------------------------------------------------------------------------#
def myTrain(model, optimizer, device, train_loader, valid_loader='dummy', epochs=1):
    train_losses = []
    valid_losses = []
    accuracies   = []
    time_start   = time.time()
    model.to(device)
    
    # create an instance of the loss
    criterion = nn.NLLLoss()
    
    #
    total_num = epochs * len(train_loader)
    print_cnt = int(len(train_loader)/20)
    
    

    for e in range(epochs):
        time_start = time.time()
        train_loss = 0
        counter    = 0
        print_coe  = 1 
        for images, labels in train_loader:
            # send date to the gpu
            images, labels = images.to(device), labels.to(device)
            
            # preparation -2
            optimizer.zero_grad() # initialize the gradients
            log_ps = model(images) # Forward
            loss   = criterion(log_ps, labels) # calculate the loss
            loss.backward()  # calculate the gradients
            optimizer.step() # update the weights and bias
            train_loss += loss.item()
            
            # print
            counter += 1
            if counter == ( print_cnt * print_coe ):
                print_coe += 1
                print('epoch<{0}/{1}> : {2:.0f} %'.format((e+1), epochs,(counter/len(train_loader)*100)))
        else:
            train_losses.append( train_loss / len(train_loader))
            #
            if valid_loader != 'dummy':
                model.eval() # Dropout = OFF
                test_loss = 0
                accuracy  = 0
                #
                with torch.no_grad():
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model(images) # Forward
                        loss   = criterion(log_ps, labels) # calculate the loss
                        test_loss += loss.item()
                        #
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals =( top_class == labels.view(*top_class.shape))
                        accuracy += torch.mean( equals.type(torch.FloatTensor))

                       #from IPython.core.debugger import Pdb; Pdb().set_trace()
                #
                model.train() # Dropout = ON
                valid_losses.append(   test_loss / len(valid_loader))
                accuracies.append( accuracy     / len(valid_loader))
            else:
                valid_losses.append(-999)
                accuracies.append( -999)
            #
            time_end  = time.time()
            time_diff = time_end - time_start
            #
            print("-----------------------------")
            print("Epoch      : {0}/{1}".format(e+1, epochs))
            print("Time       : {:.0f}".format(time_diff))
            print("Train_Loss : {:.7f}".format( train_losses[-1]))
            print("Valid_Loss : {:.7f}".format( valid_losses[-1]))
            print("Accuracy   : {:.7f}".format( accuracies[-1]))
            print("-----------------------------")
        #    
    return model, optimizer, train_losses, valid_losses, accuracies

#-------------------------------------------------------------------------------------#
# TODO: Save the checkpoint 
def saveChackPointMyModel(model, optimizer, epochs, model_type, dropout, lr, \
                          h_1, h_2, out, device, file_path_save = "./check_point.pth"):
    torch.save({
                "model_state_dict"     : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "epochs"               : epochs,
                "model_type"           : model_type,
                "dropout"              : dropout,
                "lr"                   : lr,
                "h_1"                  : h_1,
                "h_2"                  : h_2,
                "device"               : device,
                "out"                  : out,
               }, file_path_save)


#-------------------------------------------------------------------------------------#
# TODO: Write a function that loads a checkpoint and rebuilds the model
def loadCheckPointMyModel(file_path_check_point):
    #load data
    check_point = torch.load(file_path_check_point) # load check point
    
    # create model
    model, optimizer = createMyModel(model_type = check_point["model_type"],\
                                     dropout    = check_point["dropout"],\
                                     lr         = check_point["lr"],\
                                     h_1        = check_point["h_1"],\
                                     h_2        = check_point["h_2"],\
                                     out        = check_point["out"]) # create model
    
    # necessary keys
    model.load_state_dict(check_point["model_state_dict"])
    optimizer.load_state_dict(check_point["optimizer_state_dict"]) # torch bug -> this is located in gpu
    
    # option keys
    if "epochs" in check_point:
        epochs = check_point['epochs']
    else:
        epochs = -999

    return model, optimizer, epochs
    
#-------------------------------------------------------------------------------------#
def processImage(pil_image, value_resize=256, value_crop=224):
    # TODO: Process a PIL image for use in a PyTorch model
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    width, height = pil_image.size

    #------------------------------------------------------#
    # resize
    #value_resize = 256
    #
    if width > height:
        width_resized  = int( width / height * value_resize )
        height_resized = value_resize
    else:
        width_resized  = value_resize
        height_resized = int( height / width * value_resize )
    #    
    pil_image_resized = pil_image.resize((width_resized, height_resized))

    #------------------------------------------------------#
    # crop
    #value_crop = 224
    #
    left  = int( (width_resized/2) - (value_crop/2))
    right = left + value_crop
    upper = int( (height_resized/2) - (value_crop/2))
    lower = upper + value_crop
    #
    pil_image_cropped = pil_image_resized.crop((left, upper, right, lower))

    #------------------------------------------------------#
    # ImageNet Infomation
    mean_list = [0.485, 0.456, 0.406]
    std_list  = [0.229, 0.224, 0.225]

    #------------------------------------------------------#
    # [Input File Information]
    np_image1  = np.array(pil_image_cropped) # image to numpy array
    np_image2  = np_image1.transpose(2,0,1).reshape(3,-1) # Channel, W, H
    ave_images = np.mean(np_image2, axis=1) # mean of each Channel
    std_images = np.std( np_image2, axis=1) # std  of each Channel

    #------------------------------------------------------#
    # [Normalized File Information : np_image2]
    np_image2  = np_image2 / 255
    ave_images = np.mean(np_image2, axis=1)
    std_images = np.std( np_image2, axis=1)

    for i in range(3):
        np_image2[i,:] = (np_image2[i,:] - ave_images[i]) / std_images[i]
    #
    ave_images = np.mean(np_image2, axis=1)
    std_images = np.std( np_image2, axis=1)    
 

    #------------------------------------------------------#
    # [Normalized File Information for ImageNet : np_image3]
    # I think this is necessary to use pretrained model of ImageNet.
    if 1:
        for i in range(3):
            np_image2[i,:] =  np_image2[i,:] * std_list[i] + mean_list[i]
        #
        ave_images = np.mean(np_image2, axis=1)
        std_images = np.std(np_image2,  axis=1)
        #
        if 0:
            print("Results of process_image")
            print("   mean = ", ave_images)
            print("   std  = ", std_images)

    #------------------------------------------------------#
    # ReShape
    np_image3 = np_image2.reshape(3,value_crop,value_crop)

    return np_image3


#-------------------------------------------------------------------------------------#
def myImshow(torch_image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    np_image = torch_image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std   = np.array([0.229, 0.224, 0.225])
    np_image = std * np_image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    np_image = np.clip(np_image, 0, 1)
    
    ax.imshow(np_image)
    
    return ax


    
#-------------------------------------------------------------------------------------#
def myPredict(image_file, model, device, cat_to_name, topk=5):
    # load image 
    pil_image = Image.open(image_file)
    
    # pre_process image
    np_image = processImage(pil_image, value_resize=256, value_crop=224)
    image    = torch.from_numpy(np_image)  
    image    = image.type(torch.FloatTensor)
    image    = image.reshape(1,3,224,224)
    image    = image.to(device)
    
    # predict
    model.eval() # Dropout = OFF
    log_ps = model(image)
    model.train() # Dropout = ON
    ps     = torch.exp(log_ps)
    
    # check top k class
    top_p, top_class = ps.topk(topk, dim=1)
    top_p     = top_p.cpu()
    top_p     = top_p.detach().numpy().flatten()
    top_class = top_class.cpu().numpy().flatten()
    category  = top_class + 1
    
    # get names
    top_name = []
    #
    for i in range(topk):
        if cat_to_name != "dummy":
            top_name.append( cat_to_name[ str( category[i] )] )
        else:
            top_name.append("no_name")
    #
    return top_p, top_class, top_name     
    

