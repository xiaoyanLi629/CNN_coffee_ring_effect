import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


misclassification_list = list()
model_num = 10
iteration_num = 100

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

total_mis_classify = np.zeros((120*model_num, iteration_num))

for i in range(model_num):
    total_mis_classify[i*120:(i+1)*120, :] = misclassification_list[i]

num_mis_classify_images = np.count_nonzero(total_mis_classify)

for i in range(120):
    if i+1 in misclassify_dict:
        pass
    else:
        misclassify_dict[i+1] = 0

mis_list = np.zeros(120)
# print('Misclassified images percentage is calculated by for specific image, the misclassified'
#       ' times divided total classification number')
# for i in range(120):
#     mis_percent = misclassify_dict[i+1]/(model_num*iteration_num)*100
#     mis_list[i] = mis_percent
#     print('Misclassified images percentage for image', str(i+1), 'is: %.4f' % mis_percent, '%')

cluster_chemistry_result_dataframe = pd.read_csv('test_cluster_chemistry_result.csv', header=None)
cluster_chemistry_result = cluster_chemistry_result_dataframe.values
cluster_chemistry_result = cluster_chemistry_result.reshape(1, int(120 / 4))
cluster_chemistry_result = np.repeat(cluster_chemistry_result, 2)
cluster_chemistry_result = np.concatenate((cluster_chemistry_result, cluster_chemistry_result))
Y = cluster_chemistry_result

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

y_1 = y_1 * 100
y_2 = y_2 * 100
y_3 = y_3 * 100
y_4 = y_4 * 100
y_5 = y_5 * 100
y_6 = y_6 * 100

acc_1_list = np.zeros(10)
acc_2_list = np.zeros(10)
acc_3_list = np.zeros(10)
acc_4_list = np.zeros(10)
acc_5_list = np.zeros(10)
acc_6_list = np.zeros(10)

for i in range(model_num):
    mis_y_1 = 0
    mis_y_2 = 0
    mis_y_3 = 0
    mis_y_4 = 0
    mis_y_5 = 0
    mis_y_6 = 0
    for row in range(120):
        for col in range(100):
            num = misclassification_list[i][row, col]
            if num != 0:
                class_index = Y[int(num-1)]
                if class_index == 1:
                    mis_y_1 = mis_y_1 + 1
                if class_index == 2:
                    mis_y_2 = mis_y_2 + 1
                if class_index == 3:
                    mis_y_3 = mis_y_3 + 1
                if class_index == 4:
                    mis_y_4 = mis_y_4 + 1
                if class_index == 5:
                    mis_y_5 = mis_y_5 + 1
                if class_index == 6:
                    mis_y_6 = mis_y_6 + 1

    mis_acc_1 = (1 - mis_y_1 / y_1) * 100
    mis_acc_2 = (1 - mis_y_2 / y_2) * 100
    mis_acc_3 = (1 - mis_y_3 / y_3) * 100
    mis_acc_4 = (1 - mis_y_4 / y_4) * 100
    mis_acc_5 = (1 - mis_y_5 / y_5) * 100
    mis_acc_6 = (1 - mis_y_6 / y_6) * 100

    acc_1_list[i] = mis_acc_1
    acc_2_list[i] = mis_acc_2
    acc_3_list[i] = mis_acc_3
    acc_4_list[i] = mis_acc_4
    acc_5_list[i] = mis_acc_5
    acc_6_list[i] = mis_acc_6

total_acc = np.zeros((10, 6))

total_acc[:, 0] = acc_1_list
total_acc[:, 1] = acc_2_list
total_acc[:, 2] = acc_3_list
total_acc[:, 3] = acc_4_list
total_acc[:, 4] = acc_5_list
total_acc[:, 5] = acc_6_list

fig_box_plot, ax = plt.subplots()
ax.set_title('Accuracy of each class')
ax.boxplot(total_acc)
plt.xlabel('Water sample class', fontsize=15)
plt.ylabel('Test dataset accuracy (percentage)', fontsize=15)
plt.yticks(fontsize=9)
plt.xticks([1, 2, 3, 4, 5, 6], ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6'], fontsize=9)
plt.show()
fig_box_plot.savefig('Test accuracy of each class.jpg')
