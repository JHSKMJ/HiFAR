import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import random
from load import load_VLCS_data
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
	tr_Caltech, ts_Caltech, tr_Labelme, ts_Labelme, tr_Pascal, ts_Pascal, tr_Sun, ts_Sun = load_VLCS_data(batchsize)

	# LabelMe, Caltech, Sun -> Pascal
	train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([tr_Labelme, tr_Caltech, tr_Sun]), batch_size=batchsize, shuffle=True)
	val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([ts_Labelme, ts_Caltech, ts_Sun]), batch_size=batchsize, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=ts_Pascal, batch_size=batchsize, shuffle=True)

	model = torchvision.models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(in_features=512, out_features=5, bias=True)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	model = training(model, device, criterion, optimizer, lamda, epochs, train_loader)
	val_acc = testing(model, val_loader, device)
	test_acc = testing(model, test_loader, device)
	print(f'Validation_Acc: {val_acc:.4f} || Test_Acc: {test_acc:.4f}')


	# Pascal, Caltech, Sun -> LabelMe
	train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([tr_Pascal, tr_Caltech, tr_Sun]), batch_size=batchsize, shuffle=True)
	val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([ts_Pascal, ts_Caltech, ts_Sun]), batch_size=batchsize, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=ts_Labelme, batch_size=batchsize, shuffle=True)

	model = torchvision.models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(in_features=512, out_features=5, bias=True)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=lr)

	model = training(model, device, criterion, optimizer, lamda, epochs, train_loader)
	val_acc = testing(model, val_loader, device)
	test_acc = testing(model, test_loader, device)
	print(f'Validation_Acc: {val_acc:.4f} || Test_Acc: {test_acc:.4f}')


	# Pascal, LabelMe, Sun -> Caltech
	train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([tr_Pascal, tr_Labelme, tr_Sun]), batch_size=batchsize, shuffle=True)
	val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([ts_Pascal, ts_Labelme, ts_Sun]), batch_size=batchsize, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=ts_Caltech, batch_size=batchsize, shuffle=True)

	model = torchvision.models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(in_features=512, out_features=5, bias=True)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=lr)

	model = training(model, device, criterion, optimizer, lamda, epochs, train_loader)
	val_acc = testing(model, val_loader, device)
	test_acc = testing(model, test_loader, device)
	print(f'Validation_Acc: {val_acc:.4f} || Test_Acc: {test_acc:.4f}')


	# Pascal, LabelMe, Caltech -> Sun
	train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([tr_Pascal, tr_Labelme, tr_Caltech]), batch_size=batchsize, shuffle=True)
	val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([ts_Pascal, ts_Labelme, ts_Caltech]), batch_size=batchsize, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=ts_Sun, batch_size=batchsize, shuffle=True)

	model = torchvision.models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(in_features=512, out_features=5, bias=True)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=lr)

	model = training(model, device, criterion, optimizer, lamda, epochs, train_loader)
	val_acc = testing(model, val_loader, device)
	test_acc = testing(model, test_loader, device)
	print(f'Validation_Acc: {val_acc:.4f} || Test_Acc: {test_acc:.4f}')

	

if __name__ == '__main__':
	argv = sys.argv
	main(args)
