import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
import random
import torch.nn as nn
import datetime
import time
import seaborn as sn
from sklearn.metrics import confusion_matrix
from test_images import test_images
from load_train_data import load_train_data
from model_result import model_result
from load_test_data import load_test_data
from train_folder import input_train_data_folder
from test_folder import input_test_data_folder
from misclassify_class import misclassify_class
import matplotlib.image as mpimg
import PIL
import os
import torch.utils.data as Data
import torchvision
from matplotlib import cm
import collections


# train_data_folder = input_train_data_folder()
# test_data_folder = input_test_data_folder()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=8,  # n_filters
                kernel_size=10,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(8, 16, 10, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(5),  # output shape (32, 7, 7)
        )
        self.linear1 = nn.Linear(1600, 512)
        self.linear2 = nn.Linear(512, 32)
        self.out = nn.Linear(32, 6)
        # self.sigmoid = nn.Sigmoid()
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


print('Current time', str(datetime.datetime.now()))
start = time.time()
model_name = input('Please input creating model name(hit enter, a random name(1000) will be generated):')
if model_name == '':
    model_name = '1000'

print('Constructing model:', model_name)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

print('Input trainig datset and testing dataset information:')
train_x_file = ''
train_y_file = ''
test_x_file = ''
test_y_file = ''
dataset_num = ''
# train_x_file = input('Input training x file name:')
# train_y_file = input('Input training y file name:')
# test_x_file = input('Input testing x file name:')
# test_y_file = input('Input testing y file name:')
# dataset_num = input('Input dataset number (2 for 2 tables)')

if train_x_file == '':
    train_x_file = 'training_table_1'
    # train_x_file = 'training'

train_x_file = train_x_file + '.pkl'

if train_y_file == '':
    train_y_file = 'cluster_chemistry_result.csv'

if test_x_file == '':
    test_x_file = 'testing_table_1'
    # test_x_file = 'testing'

test_x_file = test_x_file + '.pkl'

if test_y_file == '':
    test_y_file = 'test_cluster_chemistry_result.csv'

if dataset_num == '':
    dataset_num = 1

# Hyper parameter
# table 1 hyperparameters
sample_num = 30
train_numImages = 120
train_rep = 4
test_numImages = 30
test_rep = 1
dataset_num = 1

# table 1 and 2 hyperparameters
# sample_num = 60
# train_numImages = 180
# train_rep = 3
# test_numImages = 120
# test_rep = 2
# dataset_num = 2

accuracy_num = 100
# img = Image.open('training_images_table1/1.jpg')
basewidth = 300
hsize = 300
X = np.zeros(shape=(train_numImages, basewidth, basewidth))
test_X = np.zeros(shape=(test_numImages, basewidth, basewidth))
# wpercent = (basewidth / float(img.size[0]))
# hsize = int((float(img.size[1]) * float(wpercent)))
EPOCH = 200  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 5
LR = 0.0001  # learning rate
stable_factor = 0  # control model stability

X, Y = load_train_data(train_numImages, train_x_file, train_y_file, train_rep, dataset_num, device)
test_X, test_Y = load_test_data(test_numImages, test_x_file, test_y_file, test_rep, dataset_num, device)

print('Epoch:', EPOCH)
print('Learning rate:', LR)
print('batch size:', BATCH_SIZE)
print('Training data replicates:', train_rep)
print('Testing data replicates:', test_rep)

weight = torch.zeros(6)
weight[0] = 1/6/12/train_rep
weight[1] = 1/6/1/train_rep
weight[2] = 1/6/3/train_rep
weight[3] = 1/6/6/train_rep
weight[4] = 1/6/6/train_rep
weight[5] = 1/6/2/train_rep

probability_dist = np.ones(train_numImages)
probability_dist = probability_dist/sum(probability_dist)

# sample weight initialization
# probability_dist[0:6] = weight[0]
# probability_dist[6:9] = weight[1]
# probability_dist[9:12] = weight[2]
# probability_dist[12:15] = weight[3]
# probability_dist[15:18] = weight[4]
# probability_dist[18:24] = weight[2]
# probability_dist[24:30] = weight[0]
# probability_dist[30:33] = weight[3]
# probability_dist[33:39] = weight[5]
# probability_dist[39:42] = weight[4]
# probability_dist[42:51] = weight[0]
# probability_dist[51:54] = weight[4]
# probability_dist[54:66] = weight[3]
# probability_dist[66:72] = weight[0]
# probability_dist[72:75] = weight[4]
# probability_dist[75:84] = weight[0]
# probability_dist[84:90] = weight[4]

cnn = CNN()
cnn.to(device)
optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)  # optimize all cnn parameters
weight = weight.to(device)
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
print(cnn)
print('Start training...')
x = list()
test_accuracy_list = np.zeros(EPOCH)
plt.ion()
epoch = 0

misclassified_images = np.zeros((test_numImages, EPOCH))

for epoch in range(EPOCH):
    # x.append(epoch)
    # samples = np.zeros(150)

    train_accuracy, prediction, train_tot_loss = test_images(X, Y, train_numImages, cnn, basewidth, loss_func, device)

    test_accuracy, test_prediction, test_tot_loss = test_images(test_X, test_Y, test_numImages, cnn, basewidth,
                                                                loss_func, device)

    if epoch % 1 == 0:
        print('Epoch: ', epoch, '| Training loss: %.8f' % train_tot_loss, '| training accuracy: %.2f' %
              train_accuracy, '%', '| Testing loss: %.8f' % test_tot_loss, '| testing accuracy: %.2f' % test_accuracy,
              '%')

    if epoch >= 0:
        mis_classify = misclassify_class(test_Y, test_numImages, test_prediction)
        test_accuracy_list[epoch] = test_accuracy
        for i in range(len(mis_classify)):
            misclassified_images[i, epoch] = mis_classify[i]

    # randomly by weight selecting samples to train model
    for i in range(int(train_numImages/BATCH_SIZE)):
        index = np.random.choice(range(len(Y)), BATCH_SIZE, p=probability_dist, replace=False)
        b_x = X[index, :, :, :]
        b_x = b_x.reshape(BATCH_SIZE, 1, basewidth, basewidth)
        b_x = b_x.to(device)
        output = cnn(b_x)[0]  # cnn output
        b_y = Y[index]
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradientsd
        # samples[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = b_y.cpu()

        # based on prediction result to uopdate weights
        pred_y = torch.max(output.cpu(), 1)[1].data.numpy()
        for i in range(len(b_y)):
            if pred_y[i] == b_y.cpu().data.numpy()[i]:
                probability_dist[index[i]] = probability_dist[index[i]]*0.95
                probability_dist = probability_dist/sum(probability_dist)
                if probability_dist[index[i]] <= 1/train_numImages*0.75:
                    probability_dist[index[i]] = 1/train_numImages*0.75
                probability_dist = probability_dist / sum(probability_dist)
                # probability_dist = [x/sum(probability_dist) for x in probability_dist]
            else:
                probability_dist[index[i]] = probability_dist[index[i]]*1.05
                probability_dist = probability_dist / sum(probability_dist)
                if probability_dist[index[i]] >= 1/train_numImages*1.25:
                    probability_dist[index[i]] = 1/train_numImages*1.25
                probability_dist = probability_dist / sum(probability_dist)
                # probability_dist = [x/sum(probability_dist) for x in probability_dist]


misclassified_images = misclassified_images.astype(int)
misclassified_images_filename = 'mis_classify_' + model_name + '_.csv'

np.savetxt(misclassified_images_filename, misclassified_images, delimiter=',')
np.savetxt('test_accuracy_' + model_name + '_.csv', test_accuracy_list, delimiter=',')

done = time.time()
elapsed = (done - start)/60
print('Programming running time(mins):', elapsed)
torch.save(cnn, model_name+'.pkl')  # save entire net

# print('True labels:')
# print(Y.cpu())
# print('Prediction labels:')
# print(prediction.astype(int))

# export training mis_classified images
mis_class_train = list()
y = Y.cpu()
for i in range(train_numImages):
    if y[i] != prediction[i]:
        mis_class_train.append(i)

# calculate accuracy of training each class
# print('Training dataset result:')
# model_result(model_name, Y, train_numImages, prediction)
# print('Testing dataset result:')
# mis_classify = model_result(model_name, test_Y, test_numImages, test_prediction)


