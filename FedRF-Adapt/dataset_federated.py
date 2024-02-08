
from torchvision import datasets, transforms
from office_dataset import get_pair_dataset, get_digit_five_dataset, get_office_10_dataset
from visda_dataset import get_dataset

def dataset_federated(dataset_name, source, target):
    
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    train_transform = transform

    if dataset_name == "office31" or dataset_name == "office_caltech_10":
        if dataset_name == "office_caltech_10":
            class_num = 10
        elif dataset_name == "office31":
            class_num = 31
        source_dataset, target_dataset, num_classes, class_names = \
            get_pair_dataset(source, target, train_transform, val_transform=train_transform, dataset=dataset_name)
    elif dataset_name == "visda":
        root = "/data/dataset/visda-2017"
        source = "train"
        target = "validation"
        source_dataset, target_dataset, num_classes, class_names = \
        get_dataset(root, source, target, train_transform, val_transform=train_transform)
    elif dataset_name == "digit_five":
        num_classes = 10
        class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        source_dataset = digit_five_dataset(source, transform=transform)
        target_dataset = digit_five_dataset(target, transform=transform)

    return source_dataset, target_dataset, num_classes, class_names

def office_10_dataset_federated(source):
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
    source_dataset = get_office_10_dataset(source, transform)
    return source_dataset

def digit_five_dataset_federated(source):
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    source_dataset = digit_five_dataset(source, transform=transform)
    return source_dataset

def digit_five_dataset(dataset_name, transform):
    if dataset_name == "mnist":
        transform = transforms.Compose([ transforms.Grayscale(3), transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.5],std=[0.5])])
        data_train = datasets.MNIST(root = "/data/dataset/digit_five",
                            transform=transform,
                            train = True,
                            download = False)
    elif dataset_name == "usps":
        transform = transforms.Compose([ transforms.Grayscale(3), transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.5],std=[0.5])])
        data_train = datasets.USPS(root = "/data/dataset/digit_five",
                            transform=transform,
                            train = True,
                            download = False)
    elif dataset_name == "svhn":
        data_train = datasets.SVHN(root = "/data/dataset/digit_five",
                            transform=transform,
                            split = 'train',
                            download = False)
    elif dataset_name == "mnist-m" or dataset_name == "synthetic_digits":
        data_train = get_digit_five_dataset(dataset_name, transform)

    return data_train