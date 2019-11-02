import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from load_test_data import load_test_data
from sklearn.metrics import confusion_matrix
from model_result import model_result
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import pickle
import os
import matplotlib

basewidth = 300
hsize = 300
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=10,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 10, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(5),  # output shape (32, 7, 7)
        )
        self.linear1 = nn.Linear(128, 500)
        self.linear2 = nn.Linear(500, 30)
        self.out = nn.Linear(30, 6)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        output = self.out(x)
        return output, x  # return x for visualization
# model_name = input('Please input model name:')
# model_name = model_name + '.pkl'
# model_name = '1.pkl'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dirName = 'Projection_images'
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory ", dirName, " Created ")
except FileExistsError:
    print("Directory ", dirName, " already exists")

for i in range(10):
    model_name = str(i+1) + '.pkl'
    net = torch.load(model_name)
    model_name_pro = str(i+1)
    dirName = 'Projection_images/' + 'model_' + model_name_pro + '_first_layer_pro'
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")
    train_x_file = ''
    if train_x_file == '':
        # train_x_file = 'train_table_1_4'
        train_x_file = 'training'
    train_x_file = train_x_file + '.pkl'
    data = pickle.load(open(train_x_file, "rb"))

    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)

    for img_num in range(data.shape[0]):
        img = data[img_num, :, :, :]
        img = img.reshape(1, 1, basewidth, hsize)
        img = img.to(device)
        layer_1 = net.conv1(img)
        layer_1 = layer_1.cpu()  # 16 filters

        for i in range(layer_1.shape[1]):
            layer_1_1 = layer_1[0, i, :, :]
            layer_1_1 = layer_1_1.detach().numpy()
            matplotlib.use('Agg')
            fig, ax = plt.subplots()
            filename = dirName + '/' + 'image_' + str(img_num+1) + '_filer_' + str(i+1) + '.jpg'
            plt.imshow(layer_1_1)
            # plt.show()
            fig.savefig(filename)
            plt.close()

