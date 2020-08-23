import argparse
from   my_library_deep_learning_for_app import *

#--------------------------------------------------------------------------#
# Create Parser
parser = argparse.ArgumentParser(description='UdaCity Project : Image Training')

#--------------------------------------------------------------------------#
# Analysis of arguments
parser.add_argument('arg1',        action='store', \
                     help = 'folder path of training data')
parser.add_argument('--valid_dir', action='store', default='dummy', \
                     help = 'folder path of validation data')
parser.add_argument('--save_dir',  action='store', default='./check_point.pth', \
                     help = 'file path of check point')
parser.add_argument('--restart',  action='store', default='dummy', \
                     help = 'input file path of check point which you created before')
parser.add_argument('--model_type',  action='store', default='vgg16', \
                     help = 'model type : explanation = vgg16, resnet50, etc....')
parser.add_argument('--epochs',  action='store', default='1', \
                     help = 'epochs of training')                                          
parser.add_argument('--dropout',  action='store', default='0.2', \
                     help = 'drop out ratio of optimizer')
parser.add_argument('--lr',  action='store', default='0.001', \
                     help = 'learning rate of optimizer')                     
parser.add_argument('--h_1',  action='store', default='1500', \
                     help = '1st hidden units of classifier')
parser.add_argument('--h_2',  action='store', default='1500', \
                     help = '2nd hidden units of classifier')
parser.add_argument('--out',  action='store', default='102', \
                     help = '    output units of classifier')
parser.add_argument('--gpu', action='store_const', const='gpu', default='cpu', \
                     help = 'device type')                     
#--------------------------------------------------------------------------#
# Analysis of arguments
args = parser.parse_args()

# get data
train_dir  = args.arg1
valid_dir  = args.valid_dir
save_dir   = args.save_dir
load_file  = args.restart
model_type = args.model_type
epochs     = int( float( args.epochs ) )
dropout    = float( args.dropout )
lr         = float( args.lr )
h_1        = int( float( args.h_1 ) )
h_2        = int( float( args.h_2 ) )
out        = int( float( args.out ) )
device_str = args.gpu

#--------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------#
# DEVICE
device = checkDeviceType(device_str)
print("---- Device : ", device )

#-------------------------------------------------------------------------------------#
# Create Data Loader
print("---- Creating Data Loader ----")
train_loader = createTrainLoader( train_dir, batch_size=64 )

if valid_dir == 'dummy':
    valid_loader = 'dummy'
else:
    valid_loader = createTrainLoader( valid_dir, batch_size=64 )

#-------------------------------------------------------------------------------------#
# Create my model
print("---- Creating Model ----")
model, optimizer = createMyModel(model_type=model_type, dropout=dropout, lr=lr, \
                                 h_1=h_1, h_2=h_2, out=out)

if load_file != "dummy":
    print("---- Loading Model to ReStart----")
    # to avoid torch loader bug, I do not use this optimizer_gpu
    # optimizer should be located in cpu
    model, optimizer_gpu, epochs_before = loadCheckPointMyModel(load_file)
    model.to(device)

#-------------------------------------------------------------------------------------#
# Training
print("---- Training ----")
model, optimizer, train_losses, valid_losses, accuracies = \
    myTrain(model, optimizer, device, train_loader, valid_loader=valid_loader, epochs=epochs )
    
#-------------------------------------------------------------------------------------#
# Save
print("---- Saving ----")
saveChackPointMyModel(model, optimizer, epochs, model_type, dropout, lr, \
                      h_1, h_2, out, device, file_path_save = save_dir)
                      
#-------------------------------------------------------------------------------------#
# Print
print("----------------------")
print(" train_step : finish !")
print(" check point : " + save_dir)
print("----------------------")