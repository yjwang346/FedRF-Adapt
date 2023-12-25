import numpy as np

import torch
from torch.utils.data import Sampler
from torch.utils.data import DataLoader, Dataset

import pickle
from torchvision import datasets, transforms
from torch.utils.data import Dataset 

# from DAN.py
def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)

class MyDataset(Dataset):
    def __init__(self, visda_dir, domain_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            info_csv: path to the csv file containing image indexes
                with corresponding labels.
            image_list: path to the txt file contains image names to training/validation set
            transform: optional transform to be applied on a sample.
        """
        f = open(visda_dir + domain_file, 'rb')
        data1 = pickle.load(f)
        f.close()
        # self.data = data1['feature']
        # self.label = data1['label']
        self.data = data1['train']
        self.label = data1['train_label']
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        return self.transform(self.data[index]), torch.from_numpy(self.label[index]).float()

    def __len__(self):
        return len(self.label)

def load_train(root_dir, domain, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    image_folder = datasets.ImageFolder(root=root_dir + domain, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=image_folder, batch_size=batch_size,
                                              shuffle=True, num_workers=1, drop_last=True)
    return data_loader

class numpyDataset(Dataset):
    def __init__(self, data_numpy, label_numpy):
        self.data = data_numpy
        self.label = label_numpy

    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.label)


        