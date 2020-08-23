import argparse
from   my_library_deep_learning_for_app import *

#--------------------------------------------------------------------------#
# Create Parser
parser = argparse.ArgumentParser(description='UdaCity Project : Image Training')

#--------------------------------------------------------------------------#
# Analysis of arguments
parser.add_argument('arg1',        action='store', \
                     help = 'file path of image data')
parser.add_argument('arg2',        action='store', \
                     help = 'file path of check point')                     
parser.add_argument('--top_k', action='store', default='1', \
                     help = 'number to display best k results')
parser.add_argument('--category_names',  action='store', default='dummy', \
                     help = 'file path of category list .json')
parser.add_argument('--gpu', action='store_const', const='gpu', default='cpu', \
                     help = 'device type')                     
#--------------------------------------------------------------------------#
# Analysis of arguments
args = parser.parse_args()

# get data
image_file = args.arg1
load_file  = args.arg2
cate_file  = args.category_names
top_k      = int( float( args.top_k ) )
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
# load json
if cate_file!="dummy":
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
else:
    cat_to_name = "dummy"

#-------------------------------------------------------------------------------------#
# load model
print("---- Loading Model to ReStart----")
model, optimizer_gpu, epochs_before = loadCheckPointMyModel(load_file)
model.to(device)

#-------------------------------------------------------------------------------------#
# predict
print("---- Predicting ----")
top_p, top_class, top_name = myPredict(image_file, model, device, cat_to_name, topk=top_k)

#-------------------------------------------------------------------------------------#
# print
for i, p in enumerate(top_p):
    print("rank={0} : probability={1:.3f} , name={2}".format( (i+1), p, top_name[i] ) )