import torch
import os
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
import argparse, sys
import glob

def load_PACS_data(batchsize):
    
    transform_art = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.555, 0.5085, 0.4579), (0.2291, 0.2182, 0.2209)),
        transforms.Resize((227,227))
    ])
    transform_car = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.8077, 0.7829, 0.7358), (0.2315, 0.2538, 0.2986)),
        transforms.Resize((227,227))
    ])
    transform_pho = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5085, 0.4832, 0.4396), (0.244, 0.236, 0.2457)),
        transforms.Resize((227,227))
    ])
    transform_ske = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.9566, 0.9566, 0.9566), (0.183, 0.183, 0.183)),
        transforms.Resize((227,227))
    ])
    
    art_painting = datasets.ImageFolder(root='./data/PACS/art_painting', transform=transform_art)
    cartoon = datasets.ImageFolder(root='./data/PACS/cartoon', transform=transform_car)
    photo = datasets.ImageFolder(root='./data/PACS/photo', transform=transform_pho)
    sketch = datasets.ImageFolder(root='./data/PACS/sketch', transform=transform_ske)

    def data_split(dataset):
        dataset_size = len(dataset)
        tr_size = int(dataset_size * 0.8)
        ts_size = dataset_size-tr_size
        train_dataset, test_dataset = random_split(dataset, [tr_size, ts_size])
        return train_dataset, test_dataset

    tr_art, ts_art = data_split(art_painting)
    tr_cartoon, ts_cartoon = data_split(cartoon)
    tr_photo, ts_photo = data_split(photo)
    tr_sketch, ts_sketch = data_split(sketch)

    return tr_art, ts_art, tr_cartoon, ts_cartoon, tr_photo, ts_photo, tr_sketch, ts_sketch


def load_VLCS_data(batchsize):
    
    transform_cal = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5121, 0.4792, 0.4509), (0.2402, 0.2231, 0.2266)),
        transforms.Resize((227,227))
    ])
    transform_label = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4607, 0.4706, 0.4573), (0.2334, 0.2319, 0.2463)),
        transforms.Resize((227,227))
    ])
    transform_pascal = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4607, 0.4706, 0.4573), (0.2368, 0.23, 0.232)),
        transforms.Resize((227,227))
    ])
    transform_sun = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4607, 0.4706, 0.4573), (0.2343, 0.232, 0.2377)),
        transforms.Resize((227,227))
    ])

    tr_Caltech = datasets.ImageFolder(root='./data/VLCS/CALTECH/train', transform=transform_cal)
    tr_Labelme = datasets.ImageFolder(root='./data/VLCS/LABELME/train', transform=transform_label)
    tr_Pascal = datasets.ImageFolder(root='./data/VLCS/PASCAL/train', transform=transform_pascal)
    tr_Sun = datasets.ImageFolder(root='./data/VLCS/SUN/train', transform=transform_sun)
    ts_Caltech = datasets.ImageFolder(root='./data/VLCS/CALTECH/test', transform=transform_cal)
    ts_Labelme = datasets.ImageFolder(root='./data/VLCS/LABELME/test', transform=transform_label)
    ts_Pascal = datasets.ImageFolder(root='./data/VLCS/PASCAL/test', transform=transform_pascal)
    ts_Sun = datasets.ImageFolder(root='./data/VLCS/SUN/test', transform=transform_sun)


    return tr_Caltech, ts_Caltech, tr_Labelme, ts_Labelme, tr_Pascal, ts_Pascal, tr_Sun, ts_Sun


def load_OfficeHome_data(batchsize):
    
    transform_art = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5156, 0.4827, 0.4455), (0.2391, 0.2348, 0.2285)),
        transforms.Resize((500,500))
    ])

    transform_clipart = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5983, 0.577, 0.5513), (0.297, 0.295, 0.2997)),
        transforms.Resize((500,500))
    ])

    transform_product = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.7472, 0.7348, 0.7261), (0.2727, 0.2789, 0.2841)),
        transforms.Resize((500,500))
    ])

    transform_real = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.6068, 0.5755, 0.5425), (0.2401, 0.2408, 0.2459)),
        transforms.Resize((500,500))
    ])

    Art = datasets.ImageFolder(root='./data/OfficeHomeDataset/Art', transform=transform_art)
    Clipart = datasets.ImageFolder(root='./data/OfficeHomeDataset/Clipart', transform=transform_clipart)
    Product = datasets.ImageFolder(root='./data/OfficeHomeDataset/Product', transform=transform_product)
    Real_World = datasets.ImageFolder(root='./data/OfficeHomeDataset/Real World', transform=transform_real)


    def data_split(dataset):
        dataset_size = len(dataset)
        tr_size = int(dataset_size * 0.8)
        ts_size = dataset_size-tr_size
        train_dataset, test_dataset = random_split(dataset, [tr_size, ts_size])
        return train_dataset, test_dataset

    tr_art, ts_art = data_split(Art)
    tr_clipart, ts_clipart = data_split(Clipart)
    tr_product, ts_product = data_split(Product)
    tr_real, ts_real = data_split(Real_World)

    return tr_art, ts_art, tr_clipart, ts_clipart, tr_product, ts_product, tr_real, ts_real

