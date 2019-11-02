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

basewidth = 300
hsize = 300
# img = Image.open('training_images_table1/1.jpg')
# wpercent = (basewidth / float(img.size[0]))
# hsize = int((float(img.size[1]) * float(wpercent)))
rep = 2
sample_num = 60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# test_numImages = input('Please input number of testing images:') # 120
test_numImages = 120
test_numImages = int(test_numImages)

# model_name = input('Please input model name:')
model_name = '1.pkl'
net = torch.load(model_name)
# pkl_file = open('test_X.pkl', 'rb')
# test_X = pickle.load(pkl_file)
# pkl_file.close()
test_X = np.zeros(shape=(test_numImages, basewidth, basewidth))

test_x_file = input('Input test x file name:')
if test_x_file == '':
    test_x_file = 'testing.pkl'
test_y_file = input('Input test y file name:')
if test_y_file == '':
    test_y_file = 'test_cluster_chemistry_result.csv'
dataset_num = input('Input dataset number:')
if dataset_num == '':
    dataset_num = 2

test_X, test_Y = load_test_data(test_numImages, test_x_file, test_y_file, rep, dataset_num, device)
X = test_X.to(device)
output = net(X)[0]
pred_y = torch.max(output.cpu(), 1)[1].data.numpy()

accuracy = 0
for i in range(len(test_Y)):
    if test_Y[i] == pred_y[i]:
        accuracy = accuracy + 1
accuracy = accuracy/test_numImages*100
print('Accuracy:', accuracy)

# confusion matrix
model_name = model_name.replace('.pkl', '')
labels = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
con_mat = confusion_matrix(test_Y.cpu(), pred_y)
fig = plt.figure()
ax = fig.add_subplot(111)
df_cm = pd.DataFrame(con_mat, range(6), range(6))
sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 11}, cmap="Blues")
plt.xlabel('Class assigned by CNN model', fontsize=12)
plt.ylabel('Class assigned by cluster analysis', fontsize=12)
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticklabels(labels, fontsize=12)
# plt.title('Model prediction vs True class confusion matrix', fontsize=16)
plt.savefig(model_name + '_confusion_matrix.jpg', dpi = 600)
plt.show()
plt.close('all')
