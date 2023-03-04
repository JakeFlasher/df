import torch
import random, math, time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from models import MLP, LeNet5, CAN, MLP_MNIST_Loader, LeNet5_MNIST_Loader, CAN_MNIST_Loader
from train import apply_models 


def main():
    # data preparation
    data_path = "./raw_data/MNIST"
    model_dump_path = "./models"
    if not os.path.exists(model_dump_path):
        os.makedirs(model_dump_path)
        os.makedirs(os.path.join(model_dump_path, "MLP"))
        os.makedirs(os.path.join(model_dump_path, "LENET5"))
        os.makedirs(os.path.join(model_dump_path, "CAN")) 
    # data exploration
    train_imgs = np.fromfile(os.path.join(data_path, "train-images.idx3-ubyte"), dtype=np.uint8)[0x10:].reshape((-1,28,28))
    train_labs = np.fromfile(os.path.join(data_path, "train-labels.idx1-ubyte"), dtype=np.uint8)[0x08:]
    test_imgs = np.fromfile(os.path.join(data_path, "t10k-images.idx3-ubyte"), dtype=np.uint8)[0x10:].reshape((-1,28,28))
    test_labs = np.fromfile(os.path.join(data_path, "t10k-labels.idx1-ubyte"), dtype=np.uint8)[0x08:]

    # test for printing the dataset
    idx1 = int(train_imgs.shape[0] * random.random())
    idx2 = int(test_imgs.shape[0] * random.random())
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(train_imgs[idx1].reshape((28,28)), cmap='gray', vmin=0, vmax=255)
    ax[0].set_title(str(train_labs[idx1]), fontsize=12)
    ax[1].imshow(test_imgs[idx2].reshape((28,28)), cmap='gray', vmin=0, vmax=255)
    ax[1].set_title(str(test_labs[idx2]), fontsize=12)
    # plt.show()

    # setting the parameters
    # n_hidden = var, determined when doing experiments
    n_class = 10
    n_feature = 32
    n_epochs = 20 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # create models
    # model_KNN = KNeighborsClassifier
    # # model_MLP = MLP(n_hidden, n_class)
    # model_LeNet5 = LeNet5(n_class)
    # model_CAN = CAN(n_feature)


    # training MLP
    n = 4
    while n <= 256:
        model_MLP = MLP(n, n_class)
        n *= 2
        train_MLP = MLP_MNIST_Loader(train_imgs, train_labs)
        test_MLP = MLP_MNIST_Loader(test_imgs, test_labs)
        apply_models(train_MLP, test_MLP, model_MLP, "MLP_"+str(n), device)


if __name__ == "__main__":
    main()