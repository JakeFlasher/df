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
 

# train on training set
def unit_training(model, optimizer, loss_function, loader, device):
    model.train()
    avg_loss = 0
    iterations = 0
    correct = 0
    image_cnt = 0

    for i, data in enumerate(loader):
        images, labels = data
        images = images.to(device=device)
        labels = labels.to(device=device) 

        optimizer.zero_grad()
        outputs = model(images)

        # print(outputs)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        # preds = torch.argmax(outputs, dim=0)
        preds = outputs.argmax().item()
        avg_loss += loss.item()
        # print(labels)
        # print(preds)
        correct += int(preds==int(labels))
        image_cnt += 1
        iterations += 1
        
    avg_loss /= iterations
    avg_accuracy = correct / iterations
    # print(avg_loss, avg_accuracy)


    return avg_loss, avg_accuracy

# validate on testing set
def unit_validation(model, loss_function, loader, device):
    model.eval()
    avg_loss = 0
    iterations = 0
    correct = 0
    image_cnt = 0
    for i, data in enumerate(loader):
        # data = [d.to(device) for d in data]
        # images, labels = data
        images, labels = data
        images = images.to(device=device)
        labels = labels.to(device=device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=0)

        loss = loss_function(outputs, labels)

        avg_loss += loss.item()
        correct += int(preds==int(labels))
        image_cnt += 1
        iterations += 1

    avg_loss /= iterations
    avg_accuracy = correct / iterations

    return avg_loss, avg_accuracy
# def apply_KNN(train_loader, test_loader, n_neighbors, device):

def apply_models(train_loader, test_loader, model, modelname, device):
    # since the training setting are the same, we omit the redundant parameters
    lr = 0.01
    batch_size = 64
    epochs = 20
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    model_state_dict = {"epoch":0, "model_state_dict":model.state_dict(), "loss":0}
    best_acc = 0
    out_path = os.path.join("./results")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for epoch in range(epochs):
        train_loss, train_acc = unit_training(model, optimizer, loss_function, train_loader, device)
        print("\tEpoch {}, Train Loss {:.5f}, Training Accuracy {:.5f}".format(epoch, train_loss, train_acc))
        if epoch % 5 == 0:
            validate_loss, validate_acc = unit_validation(model, loss_function, test_loader, device)
            print("\tEpoch {}, Validation Loss {:.5f}, Validation Accuracy {:.5f}".format(epoch, validate_loss, validate_acc))

            if validate_acc > best_acc and epoch > epochs * 0.1:
                best_acc = validate_acc
    			# save the model parameters for quick use
                # model_state_dict["epoch"] = epoch
                # model_state_dict["loss"] = best_acc
                # model_state_dict["model_state_dict"] = model.state_dict()
                # torch.save(model_state_dict, os.path.join(os.path.join(out_path, modelname), "_best.pt"))

    fout = os.path.join(out_path, modelname) + "_result.txt"
    with open(fout, "w+") as f:
        f.write("lr: {}\nepoch: {}\nbest_val: {}\nbatch_size: {}\n".format(lr, epochs, best_acc, batch_size))
        if modelname == "MLP":
            f.write("num_neurons: {}\n".format(num_neurons))    
        elif modelname == "CNN":
            f.write("act: {}\n".format(act)) 
        elif modelname == "CAN":
            f.write("feat_dim: {}\n".format(feat_dim))
        # elif modelname == MODE_KNN:
        #     f.write("best_acc: {}\nnum_neighbors: {}\ninference_time: {}".format(best_acc, num_neighbors, dt))
        f.close()

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

    import sys
    with open('file', 'a') as sys.stdout:
        # print('test')
    # training MLP
        n = 4
        while n <= 256:
            model_MLP = MLP(n, n_class)
            n *= 2
            train_MLP = MLP_MNIST_Loader(train_imgs, train_labs)
            test_MLP = MLP_MNIST_Loader(test_imgs, test_labs)
            apply_models(train_MLP, test_MLP, model_MLP, "MLP_"+str(n), device)
                # training LeNet5
        model = LeNet5(act_func="tanh") 
        training_set = LeNet5_MNIST_Loader(train_imgs, train_labs)
        testing_set = LeNet5_MNIST_Loader(test_imgs, test_labs)
        apply_models(training_set, testing_set, model, "LeNet5_"+str(n), device)


        model = LeNet5(act_func="relu") 
        training_set = LeNet5_MNIST_Loader(train_imgs, train_labs)
        testing_set = LeNet5_MNIST_Loader(test_imgs, test_labs)
        apply_models(training_set, testing_set, model, "LeNet5_"+str(n), device)

        n = 4
        while n <= 64:
            model = CAN(n)
            n *= 2
            train_MLP = CAN_MNIST_Loader(train_imgs, train_labs)
            test_MLP = CAN_MNIST_Loader(test_imgs, test_labs)
            apply_models(train_MLP, test_MLP, model, "CAN_"+str(n), device)



if __name__ == "__main__":
    main()