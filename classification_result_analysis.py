import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


misclassification_list = list()
model_num = 10
iteration_num = 100
dataset_num = 2
rep = 2
test_images = 120

for i in range(model_num):
    filename = 'mis_classify_' + str(i+1) + '_.csv'
    data_pd = pd.read_csv(filename, sep=',', header=None)
    data = data_pd.values
    data = data[:, 100:200]
    misclassification_list.append(data)

misclassify_dict = dict()
for i in range(len(misclassification_list)):
    unique, counts = np.unique(misclassification_list[i], return_counts=True)
    dic = dict(zip(unique, counts))
    misclassify_dict = {x: misclassify_dict.get(x, 0) + dic.get(x, 0) for x in set(misclassify_dict).union(dic)}

total_mis_classify = np.zeros((test_images*model_num, iteration_num))

for i in range(model_num):
    total_mis_classify[i*test_images:(i+1)*test_images, :] = misclassification_list[i]

num_mis_classify_images = np.count_nonzero(total_mis_classify)

for i in range(test_images):
    if i+1 in misclassify_dict:
        pass
    else:
        misclassify_dict[i+1] = 0

mis_list = np.zeros(test_images)
print('Misclassified images percentage is calculated by for specific image, the misclassified'
      ' times divided total classification number')
for i in range(test_images):
    mis_percent = misclassify_dict[i+1]/(model_num*iteration_num)*100
    mis_list[i] = mis_percent
    # print('Misclassified images percentage for image', str(i+1), 'is: %.4f' % mis_percent, '%')

# mis classification above 70%
for i in range(len(mis_list)):
    if mis_list[i] >= 70:
        print('Mis classification percentage (above 70%) for image {} is {}'.format(int(i+1), mis_list[i]))

cluster_chemistry_result_dataframe = pd.read_csv('test_cluster_chemistry_result.csv', header=None)
cluster_chemistry_result = cluster_chemistry_result_dataframe.values
cluster_chemistry_result = cluster_chemistry_result.reshape(1, int(test_images / (dataset_num*2)))
cluster_chemistry_result = np.repeat(cluster_chemistry_result, rep)
Y = np.copy(cluster_chemistry_result)
for i in range(dataset_num-1):
    Y = np.concatenate((Y, cluster_chemistry_result))

# calculate accuracy of each class
y_1 = 0
y_2 = 0
y_3 = 0
y_4 = 0
y_5 = 0
y_6 = 0

for i in range(len(Y)):
    if Y[i] == 1:
        y_1 = y_1 + 1

for i in range(len(Y)):
    if Y[i] == 2:
        y_2 = y_2 + 1

for i in range(len(Y)):
    if Y[i] == 3:
        y_3 = y_3 + 1

for i in range(len(Y)):
    if Y[i] == 4:
        y_4 = y_4 + 1

for i in range(len(Y)):
    if Y[i] == 5:
        y_5 = y_5 + 1

for i in range(len(Y)):
    if Y[i] == 6:
        y_6 = y_6 + 1

y_1 = y_1 * 100 * 10
y_2 = y_2 * 100 * 10
y_3 = y_3 * 100 * 10
y_4 = y_4 * 100 * 10
y_5 = y_5 * 100 * 10
y_6 = y_6 * 100 * 10

y_1_mis = 0
y_2_mis = 0
y_3_mis = 0
y_4_mis = 0
y_5_mis = 0
y_6_mis = 0

for row in range(total_mis_classify.shape[0]):
    for col in range(total_mis_classify.shape[1]):
        mis_sample = total_mis_classify[row, col]
        if mis_sample != 0:
            mis_sample = mis_sample - 1
            mis_sample_class = Y[int(mis_sample)]
            if mis_sample_class == 1:
                y_1_mis = y_1_mis + 1
            if mis_sample_class == 2:
                y_2_mis = y_2_mis + 1
            if mis_sample_class == 3:
                y_3_mis = y_3_mis + 1
            if mis_sample_class == 4:
                y_4_mis = y_4_mis + 1
            if mis_sample_class == 5:
                y_5_mis = y_5_mis + 1
            if mis_sample_class == 6:
                y_6_mis = y_6_mis + 1


# accuracy for group 1
y_1_acc = y_1_mis / y_1
y_1_acc = (1 - y_1_acc) * 100

# accuracy for group 2
y_2_acc = y_2_mis / y_2
y_2_acc = (1 - y_2_acc) * 100

# accuracy for group 3
y_3_acc = y_3_mis / y_3
y_3_acc = (1 - y_3_acc) * 100

# accuracy for group 4
y_4_acc = y_4_mis / y_4
y_4_acc = (1 - y_4_acc) * 100

# accuracy for group 5
y_5_acc = y_5_mis / y_5
y_5_acc = (1 - y_5_acc) * 100

# accuracy for group 6
y_6_acc = y_6_mis / y_6
y_6_acc = (1 - y_6_acc) * 100

x = [1, 2, 3, 4, 5, 6]
test_accuracy_list = list()
test_accuracy_list.append(y_1_acc)
test_accuracy_list.append(y_2_acc)
test_accuracy_list.append(y_3_acc)
test_accuracy_list.append(y_4_acc)
test_accuracy_list.append(y_5_acc)
test_accuracy_list.append(y_6_acc)

# test dataset accuracy of each class
fig_accuracy_plot, ax = plt.subplots()
# ax.set_title('Accuracy of each class')
plt.scatter(x, test_accuracy_list)
plt.xlabel('class number', fontsize=15)
plt.ylabel('Accuracy of each class (percentage)', fontsize=15)
plt.yticks(fontsize=9)
plt.xticks([1, 2, 3, 4, 5, 6], ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6'], fontsize=9)
plt.show()
fig_accuracy_plot.savefig('Test accuracy of each class.jpg')
plt.close(fig_accuracy_plot)

# plot accuracy histogram

for i in range(len(Y)):
    if Y[i] != np.min(Y[i:]):
        index = i + np.where(Y[i:] == np.min(Y[i:]))[0][0]
        Y[i], Y[index] = Y[index], Y[i]
        mis_list[i], mis_list[index] = mis_list[index], mis_list[i]

for i in range(6+1):
    Y_sub = Y[np.where(Y == i)]
    mis_list_sub = mis_list[np.where(Y == i)]
    for num in range(len(mis_list_sub)):
        if mis_list_sub[num] != np.min(mis_list_sub[num:]):
            index = num + np.where(mis_list_sub[num:] == np.min(mis_list_sub[num:]))[0][0]
            mis_list_sub[num], mis_list_sub[index] = mis_list_sub[index], mis_list_sub[num]
            Y_sub[num], Y_sub[index] = Y_sub[index], Y_sub[num]
    # print(mis_list_sub)
    mis_list[np.where(Y == i)] = mis_list_sub
x = []
for i in range(test_images):
    x.append(i)
x_tick = []
for i in range(test_images):
    x_tick.append(str(Y[i]))

color = []
for i in range(test_images):
    if Y[i] == 1:
        color.append('red')
    if Y[i] == 2:
        color.append('green')
    if Y[i] == 3:
        color.append('cyan')
    if Y[i] == 4:
        color.append('yellow')
    if Y[i] == 5:
        color.append('purple')
    if Y[i] == 6:
        color.append('black')

fig, ax = plt.subplots()
x = np.arange(120)
plt.bar(x, height=mis_list, color=color)


x_line = np.linspace(x[0], x[47], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[0]
ax.plot(x_line, y)

x_line = np.linspace(x[48], x[51], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[1]
ax.plot(x_line, y)

x_line = np.linspace(x[52], x[63], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[2]
ax.plot(x_line, y)

x_line = np.linspace(x[64], x[87], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[3]
ax.plot(x_line, y)

x_line = np.linspace(x[88], x[111], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[4]
ax.plot(x_line, y)

x_line = np.linspace(x[112], x[119], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[5]
ax.plot(x_line, y)

plt.xlabel('Test image class number', fontsize=15)
plt.ylabel('Mis-classification percentage', fontsize=15)
# ax.set_title('Mis-classification percentage of each image', fontsize=20)
plt.xticks([23, 49, 62, 77, 101, 117],
           ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6'], fontsize=9)
fig.savefig('Mis-classification percentage color class.jpg')
plt.show()
plt.close(fig)



mis_list_non_zero = []
Y_non_zero = []
color_non_zero = []
x = []
index = 0
for i in range(len(Y)):
    if mis_list[i] != 0:
        x.append(index)
        mis_list_non_zero.append(mis_list[i])
        Y_non_zero.append(Y[i])
        if Y[i] == 1:
            color_non_zero.append('red')
        if Y[i] == 2:
            color_non_zero.append('green')
        if Y[i] == 3:
            color_non_zero.append('cyan')
        if Y[i] == 4:
            color_non_zero.append('yellow')
        if Y[i] == 5:
            color_non_zero.append('purple')
        if Y[i] == 6:
            color_non_zero.append('black')
        index = index + 1

x = np.asarray(x)
mis_list_non_zero = np.asarray(mis_list_non_zero)
Y_non_zero = np.asarray(Y_non_zero)

fig, ax = plt.subplots()
plt.bar(x, height=mis_list_non_zero, color=color_non_zero)
# Y_non_zero
# array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#        4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#        5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6], dtype=int64)
x_line = np.linspace(x[0], x[22], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[0]
ax.plot(x_line, y)

x_line = np.linspace(x[23], x[26], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[1]
ax.plot(x_line, y)

x_line = np.linspace(x[27], x[31], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[2]
ax.plot(x_line, y)

x_line = np.linspace(x[32], x[49], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[3]
ax.plot(x_line, y)

x_line = np.linspace(x[50], x[64], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[4]
ax.plot(x_line, y)

x_line = np.linspace(x[65], x[71], 1000)
y = np.zeros(1000)
y = y + test_accuracy_list[5]
ax.plot(x_line, y)

plt.xlabel('Test image class number', fontsize=13)
plt.ylabel('Mis-classification percentage', fontsize=13)
# ax.set_title('Mis-classification percentage of each image', fontsize=17)
plt.xticks([11, 24, 31, 40, 58, 68],
           ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6'], fontsize=9)
fig.savefig('Mis-classification percentage color class without zero mis-classification.jpg')
plt.show()
plt.close(fig)

