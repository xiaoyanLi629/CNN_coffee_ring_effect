def misclassify_class(Y, num, prediction):
    # export training mis_classified images
    mis_class = list()
    y = Y.cpu()
    for i in range(num):
        if y[i] != prediction[i]:
            mis_class.append(i+1)

    return mis_class