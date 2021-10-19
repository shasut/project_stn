



from torchvision import datasets, transforms
import torch
def my_data_loader_mnist(data_path, batch_size):
    # Training dataset
    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=data_path, train=True, download=True, transform=my_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=data_path, train=False, transform=my_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)
    return train_loader, test_loader



def my_data_loader_svhn(data_path, batch_size):
    # Training dataset
    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=data_path, split="train", download=True, transform=my_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=data_path, split="test", download=True, transform=my_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)
    return train_loader, test_loader



def my_data_loader_stl10(data_path, batch_size):
    # Training dataset
    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.STL10(root=data_path, split="train", download=False, transform=my_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.STL10(root=data_path, split="test", download=False, transform=my_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)
    return train_loader, test_loader



#
# import scipy.io
# import wget
# import os
# from torchvision import transforms
# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
#
#
# class MyDatasetColorDigits(Dataset):
#     """custom dataset for .mat images"""
#     def __init__(self, root, transform=None):
#         self.url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
#         self.data_path = root
#
#         if not os.path.exists(self.data_path + "train_32x32.mat"):
#             print("Downloading the dataset....")
#             wget.download(self.url, out=self.data_path)
#             print("Dataset successfully download.")
#
#         # data = torch.tensor(self.indicator.iloc[idx, :],dtype=torch.float)
#
#         self.mat = scipy.io.loadmat(self.data_path + "train_32x32.mat")
#         self.X = self.mat["X"].transpose(3, 0, 1, 2)
#         self.y = self.mat["y"].astype(np.int64)
#         self.transform = transform
#         # print(self.X.shape)
#
#     def __len__(self):
#         return len(self.X)
#         # pass
#
#     def __getitem__(self, index):
#         image = self.X[index]
#         label = self.y[index][0]
#         if self.transform:
#             image = self.transform(image)
#         return image, label
#
#
# def my_data_loader_custom_color_digit(data_path, batch_size):
#     my_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#
#     dataset=MyDatasetColorDigits(root=data_path, transform=my_transform)
#     length = int(len(dataset) * 0.8)
#     train_set, test_set = torch.utils.data.random_split(dataset, [length, len(dataset) - length])
#
#     train_loader = torch.utils.data.DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=2)
#     # Test dataset
#     test_loader = torch.utils.data.DataLoader(
#         test_set,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=2)
#     return train_loader, test_loader
