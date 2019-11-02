from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def model_result(model_name, Y, num, prediction):
    # export training mis_classified images
    mis_class = list()
    y = Y.cpu()
    for i in range(num):
        if y[i] != prediction[i]:
            mis_class.append(i+1)

    print('Mis_classified images (number):', mis_class)

    # calculate accuracy of each class
    y_0 = 0
    y_1 = 0
    y_2 = 0
    y_3 = 0
    y_4 = 0
    y_5 = 0

    for i in range(num):
        if y[i] == 0:
            y_0 = y_0 + 1

    for i in range(num):
        if y[i] == 1:
            y_1 = y_1 + 1

    for i in range(num):
        if y[i] == 2:
            y_2 = y_2 + 1

    for i in range(num):
        if y[i] == 3:
            y_3 = y_3 + 1

    for i in range(num):
        if y[i] == 4:
            y_4 = y_4 + 1

    for i in range(num):
        if y[i] == 5:
            y_5 = y_5 + 1

    y_0_mis = 0
    y_1_mis = 0
    y_2_mis = 0
    y_3_mis = 0
    y_4_mis = 0
    y_5_mis = 0

    # accuracy for group 0
    for i in range(num):
        if y[i] == 0 and y[i] == prediction[i]:
            y_0_mis = y_0_mis + 1
    y_0_acc = y_0_mis/y_0

    # accuracy for group 1
    for i in range(num):
        if y[i] == 1 and y[i] == prediction[i]:
            y_1_mis = y_1_mis + 1
    y_1_acc = y_1_mis/y_1

    # accuracy for group 2
    for i in range(num):
        if y[i] == 2 and y[i] == prediction[i]:
            y_2_mis = y_2_mis + 1
    y_2_acc = y_2_mis/y_2

    # accuracy for group 3
    for i in range(num):
        if y[i] == 3 and y[i] == prediction[i]:
            y_3_mis = y_3_mis + 1
    y_3_acc = y_3_mis/y_3

    # accuracy for group 4
    for i in range(num):
        if y[i] == 4 and y[i] == prediction[i]:
            y_4_mis = y_4_mis + 1
    y_4_acc = y_4_mis/y_4

    # accuracy for group 5
    for i in range(num):
        if y[i] == 5 and y[i] == prediction[i]:
            y_5_mis = y_5_mis + 1
    y_5_acc = y_5_mis/y_5

    print('Accuracy of class 0:', y_0_acc, 'Accuracy of class 1:', y_1_acc, 'Accuracy of class 2:', y_2_acc,
          'Accuracy of class 3:', y_3_acc,'Accuracy of class 4:', y_4_acc, 'Accuracy of class 5:', y_5_acc,)
    return mis_class