import matplotlib.image as mpimg
import numpy as np
import PIL
from PIL import Image
import os
import pandas as pd
import torch
import pickle


def load_test_data(numImages, x_file, y_file, rep, dataset_num, device):
    """ Loading training data... """
    """ Return training data, training data labels"""

    data = pickle.load(open(x_file, "rb"))
    cluster_chemistry_result_dataframe = pd.read_csv(y_file, header=None)
    cluster_chemistry_result = cluster_chemistry_result_dataframe.values
    cluster_chemistry_result = cluster_chemistry_result - 1
    cluster_chemistry_result = cluster_chemistry_result.reshape(1, int(numImages / (rep * dataset_num)))
    cluster_chemistry_result = np.repeat(cluster_chemistry_result, rep)
    y = np.copy(cluster_chemistry_result)
    for i in range(dataset_num - 1):
        y = np.concatenate((y, cluster_chemistry_result))

    data = torch.from_numpy(data)
    y = torch.from_numpy(y)
    data = data.type(torch.FloatTensor)
    # data = data.reshape((data.shape[0], 1, data.shape[1], data.shape[2]))
    y = y.to(device)
    return data, y
