# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
import numpy as np
# Define networks
class MLP(nn.Module):
    def __init__(self, n_hidden, n_class):    
        super(MLP, self).__init__()

        self.layer = torch.nn.Sequential(
            nn.Linear(28*28, n_hidden, bias=True),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_class, bias=False)
            )

    def forward(self, x):
        x = self.layer(x)
        return x

class MLP_MNIST_Loader(Dataset):
    def __init__(self, images, labels) -> None:
        super().__init__()
        self.images = torch.from_numpy((images.reshape((-1, 28*28))/255).astype(np.float32))
        self.labels = torch.from_numpy(labels).long()
        self.len = images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.len

# training_set = MLP_MNIST_Loader(train_images, train_labels)
# validation_set = MLP_MNIST_Loader(test_images, test_labels)


# Define networks
class LeNet5(nn.Module):
    def __init__(self, act_func="tanh"):
        super(LeNet5, self).__init__()
        if act_func == "tanh":
            self.feature_extract = nn.Sequential(
                nn.Conv2d(1, 6, 1),
                nn.Tanh(),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5),
                nn.Tanh(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 120, 5),
                nn.Tanh()
                )
        else:
            self.feature_extract = nn.Sequential(
                nn.Conv2d(1, 6, 1),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 120, 5),
                nn.ReLU(True),
                )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, n_class)
            )

    def forward(self, x):
            x = self.feature_extract(x)
            x = torch.squeeze(x)
            x = self.classifier(x)
            out = func.softmax(x)
            return out

class LeNet5_MNIST_Loader(Dataset):
    def __init__(self, images, labels) -> None:
        super().__init__()
        self.images = torch.from_numpy((images.reshape((-1, 1, 28, 28))/255).astype(np.float32))
        self.labels = torch.from_numpy(labels).long()
        self.len = images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.len

# training_set = MNIST_CNN_Dataset(train_images, train_labels)
# validation_set = MNIST_CNN_Dataset(test_images, test_labels)

# Define networks
class CAN(nn.Module):
    def __init__(self, n_feature=32):
        super(CAN, self).__init__()

        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.n_feature, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=self.n_feature, out_channels=self.n_feature, kernel_size=3, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=self.n_feature, out_channels=self.n_feature, kernel_size=3, dilation=4)
        self.conv4 = nn.Conv2d(in_channels=self.n_feature, out_channels=self.n_feature, kernel_size=3, padding=3, dilation=8)
        self.conv5 = nn.Conv2d(in_channels=self.n_feature, out_channels=10, kernel_size=3, dilation=1)

        self.avg_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = func.leaky_relu(x)
        x = self.conv2(x)
        x = func.leaky_relu(x)
        x= self.conv3(x)
        x = func.leaky_relu(x)
        x = self.conv4(x)
        x = func.leaky_relu(x)
        x = self.conv5(x)
        # x = nn.ReLU(x)
        x = self.avg_pool(x)
        out = torch.squeeze(x)
        # out = torch.mean(x, dim=(2,3))

        return out


class CAN_MNIST_Loader(Dataset):
    def __init__(self, images, labels) -> None:
        super().__init__()
        self.images = torch.from_numpy((images.reshape((-1, 1, 28, 28))/255).astype(np.float32))
        self.labels = torch.from_numpy(labels).long()
        self.len = images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.len