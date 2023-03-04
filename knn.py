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


def flatten_imgs(imgs):
    flat = []
    for i, data in enumerate(imgs):
        flat.append(data.flatten())
    return np.array(flat)

def KNN(n_neighbors, train_imgs, train_labs, test_imgs, test_labs):
    # images cleaning
    train_imgs = train_imgs.reshape((-1, 28*28))
    test_imgs = test_imgs.reshape((-1, 28*28))

    # p=1 in minkowski metric means using manhattan distance, which is SAD
    KNN = KNeighborsClassifier(n_neighbors=n_neighbors, p=1, n_jobs=-1) 
    KNN.fit(train_imgs, train_labs)

    start = time.time()
    preds = KNN.predict(test_imgs)
    dt = time.time() - start
    acc = np.sum(preds == test_labs)
    acc = acc / len(test_imgs)
    print("Validation took {} s\n Test Accuracy: {}".format(dt, acc))

    return acc, dt, KNN


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
    accuracy = []
    time = []
    for i in range(10):
        acc, dt, knn = KNN(i+1, train_imgs, train_labs, test_imgs, test_labs)
        accuracy.append(acc)
        time.append(dt)
    plt.plot(accuracy)
    plt.xticks(range(10), range(1,11))
    plt.show()
    plt.plot(time)
    plt.show()
    # print(accuracy)

if __name__ == "__main__":
    main()