import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import random
from load import load_PACS_data
from train_test import training, testing

import argparse, sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=20, help='the number of epoches')
parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-lamda', type=float, default=0.0, help='lambda for penelty')
parser.add_argument('-batch_size', type=int, default=64, help='batch size')
parser.add_argument('-random_seed', type=int, default='42')
args = parser.parse_args()

def main(args):
	epochs = args.epoch
	lr = args.lr
	lamda = args.lamda
	batchsize = args.batch_size
	seed = args.random_seed

	# fix random seed
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# pl.seed_everything(seed)


	# GPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Using device: {device}')
	
	# data load
	tr_art, ts_art, tr_cartoon, ts_cartoon, tr_photo, ts_photo, tr_sketch, ts_sketch = load_PACS_data(batchsize)

	# Art, Cartoon, Sketch -> Photo
	train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([tr_art, tr_cartoon, tr_sketch]), batch_size=batchsize, shuffle=True)
	val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([ts_art, ts_cartoon, ts_sketch]), batch_size=batchsize, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=ts_photo, batch_size=batchsize, shuffle=True)

	model = torchvision.models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(in_features=512, out_features=7, bias=True)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	model = training(model, device, criterion, optimizer, lamda, epochs, train_loader)
	val_acc = testing(model, val_loader, device)
	test_acc = testing(model, test_loader, device)
	print(f'Validation_Acc: {val_acc:.4f} || Test_Acc: {test_acc:.4f}')


	# Photo, Cartoon, Sketch -> Art
	train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([tr_photo, tr_cartoon, tr_sketch]), batch_size=batchsize, shuffle=True)
	val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([ts_photo, ts_cartoon, ts_sketch]), batch_size=batchsize, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=ts_art, batch_size=batchsize, shuffle=True)

	model = torchvision.models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(in_features=512, out_features=7, bias=True)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=lr)

	model = training(model, device, criterion, optimizer, lamda, epochs, train_loader)
	val_acc = testing(model, val_loader, device)
	test_acc = testing(model, test_loader, device)
	print(f'Validation_Acc: {val_acc:.4f} || Test_Acc: {test_acc:.4f}')


	# Photo, Art, Sketch -> Cartoon
	train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([tr_art, tr_photo, tr_sketch]), batch_size=batchsize, shuffle=True)
	val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([ts_art, ts_photo, ts_sketch]), batch_size=batchsize, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=ts_cartoon, batch_size=batchsize, shuffle=True)

	model = torchvision.models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(in_features=512, out_features=7, bias=True)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=lr)

	model = training(model, device, criterion, optimizer, lamda, epochs, train_loader)
	val_acc = testing(model, val_loader, device)
	test_acc = testing(model, test_loader, device)
	print(f'Validation_Acc: {val_acc:.4f} || Test_Acc: {test_acc:.4f}')


	# Photo, Art, Cartoon -> Sketch
	train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([tr_photo, tr_art, tr_cartoon]), batch_size=batchsize, shuffle=True)
	val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([ts_photo, ts_art, ts_cartoon]), batch_size=batchsize, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=ts_sketch, batch_size=batchsize, shuffle=True)

	model = torchvision.models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(in_features=512, out_features=7, bias=True)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=lr)

	model = training(model, device, criterion, optimizer, lamda, epochs, train_loader)
	val_acc = testing(model, val_loader, device)
	test_acc = testing(model, test_loader, device)
	print(f'Validation_Acc: {val_acc:.4f} || Test_Acc: {test_acc:.4f}')



if __name__ == '__main__':
	argv = sys.argv
	main(args)
