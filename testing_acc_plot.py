import pandas as pd
import matplotlib.pyplot as plt

filename = 'test_accuracy_1_.csv'
accuracy_list = pd.read_csv(filename, header=None)
accuracy_list = accuracy_list.values

fig, ax = plt.subplots()
plt.plot(accuracy_list)
plt.xlabel('Epoch number', fontsize=15)
plt.ylabel('testing accuracy percentage', fontsize=15)
# ax.set_title('Testing dataset accuracy percentage in model training process', fontsize=20)
# fig.savefig('Mis-classification percentage.jpg')
plt.show()
plt.close(fig)
