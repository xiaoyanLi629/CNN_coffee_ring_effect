import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# accuracy plot of all 10 runs

EPOCH = 200
test_accuracy_list = np.zeros((EPOCH, 10))
model_num = 10
for i in range(model_num):
    filename = 'test_accuracy_' + str(i+1) + '_.csv'
    data_pd = pd.read_csv(filename, sep=',', header=None)
    data = data_pd.values
    data = data.reshape(EPOCH)
    test_accuracy_list[:, i] = data

test_accuracy_list_last_100 = test_accuracy_list[100:200, :]
fig_box_plot, ax = plt.subplots()
# ax.set_title('Accuracy of each independent model')
ax.boxplot(test_accuracy_list_last_100)
plt.xlabel('CNN model', fontsize=15)
plt.ylabel('Test dataset accuracy (percentage)', fontsize=15)
plt.yticks(fontsize=9)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], fontsize=9)
plt.show()
fig_box_plot.savefig('Test accuracy figure' + '.jpg', dpi = 1000)
